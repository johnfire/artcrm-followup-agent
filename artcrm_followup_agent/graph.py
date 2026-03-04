from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END

from .protocols import (
    AgentMission, LanguageModel, InboxFetcher, ContactMatcher,
    InteractionLogger, OptOutSetter, InboxMessageMarker,
    OverdueFetcher, EmailSender, RunStarter, RunFinisher,
)
from .state import FollowupState
from .prompts import classify_reply_prompt, draft_reply_prompt, draft_followup_prompt
from ._utils import parse_json_response


def create_followup_agent(
    llm: LanguageModel,
    fetch_inbox: InboxFetcher,
    match_contact: ContactMatcher,
    log_interaction: InteractionLogger,
    set_opt_out: OptOutSetter,
    mark_message_processed: InboxMessageMarker,
    fetch_overdue: OverdueFetcher,
    send_email: EmailSender,
    start_run: RunStarter,
    finish_run: RunFinisher,
    mission: AgentMission,
    overdue_days: int = 90,
):
    """
    Build and return a compiled LangGraph follow-up agent.

    Processes two work streams per run:
      1. Inbox replies — classifies each, logs interactions, flags opt-outs,
         and drafts replies for interested contacts.
      2. Overdue contacts — finds contacts with no reply after `overdue_days`
         and drafts brief follow-up emails.

    All drafted emails are sent autonomously. A daily report (visible at
    /activity/ in the supervisor UI) is the human checkpoint.

    Usage:
        agent = create_followup_agent(llm=..., ...)
        result = agent.invoke({})
        print(result["summary"])
    """

    def init(state: FollowupState) -> dict:
        run_id = start_run("followup_agent", {})
        return {
            "run_id": run_id,
            "inbox_messages": [],
            "classified_replies": [],
            "overdue_contacts": [],
            "emails_to_send": [],
            "errors": [],
            "sent_count": 0,
            "opt_out_count": 0,
            "summary": "",
        }

    def fetch_inbox_messages(state: FollowupState) -> dict:
        try:
            messages = fetch_inbox()
        except Exception as e:
            return {"errors": state["errors"] + [f"fetch_inbox: {e}"]}
        return {"inbox_messages": messages}

    def classify_replies(state: FollowupState) -> dict:
        """
        For each inbox message:
        - Try to match to a contact by sender email
        - Classify the reply with the LLM
        - If opt_out: flag immediately
        - If interested: draft a reply
        - Log the interaction regardless
        - Mark message as processed
        """
        classified = []
        emails_to_send = list(state.get("emails_to_send", []))
        opt_out_count = 0

        for msg in state.get("inbox_messages", []):
            contact = None
            try:
                contact = match_contact(msg["from_email"])
            except Exception:
                pass

            # Classify the reply
            system, user = classify_reply_prompt(mission, msg)
            classification = "other"
            reasoning = ""
            try:
                response = llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
                result = parse_json_response(response.content)
                classification = result.get("classification", "other")
                reasoning = result.get("reasoning", "")
            except Exception as e:
                classification = "other"
                reasoning = f"classification error: {e}"

            entry = {
                "inbox_message_id": msg["id"],
                "contact_id": contact["id"] if contact else None,
                "from_email": msg["from_email"],
                "classification": classification,
                "reasoning": reasoning,
            }

            # Handle opt-out immediately
            if classification == "opt_out" and contact:
                try:
                    set_opt_out(contact["id"])
                    opt_out_count += 1
                except Exception as e:
                    entry["error"] = f"set_opt_out: {e}"

            # Log the interaction if we matched a contact
            if contact:
                outcome_map = {
                    "interested": "interested",
                    "rejected": "rejected",
                    "opt_out": "no_reply",
                    "other": "no_reply",
                }
                try:
                    log_interaction(
                        contact_id=contact["id"],
                        method="email",
                        direction="inbound",
                        summary=f"{classification}: {msg.get('subject', '')}",
                        outcome=outcome_map.get(classification, "no_reply"),
                    )
                except Exception:
                    pass

            # Draft a reply for interested contacts
            if classification == "interested" and contact and contact.get("email"):
                language = contact.get("preferred_language") or mission.language_default
                sys_p, usr_p = draft_reply_prompt(mission, contact, msg, language)
                try:
                    draft_resp = llm.invoke([SystemMessage(content=sys_p), HumanMessage(content=usr_p)])
                    draft = parse_json_response(draft_resp.content)
                    emails_to_send.append({
                        "contact_id": contact["id"],
                        "to_email": contact["email"],
                        "subject": draft.get("subject", ""),
                        "body": draft.get("body", ""),
                        "type": "reply",
                    })
                    entry["draft_queued"] = True
                except Exception as e:
                    entry["draft_error"] = str(e)

            # Mark the inbox message as processed
            try:
                mark_message_processed(msg["id"], contact["id"] if contact else None)
            except Exception:
                pass

            classified.append(entry)

        return {
            "classified_replies": classified,
            "emails_to_send": emails_to_send,
            "opt_out_count": opt_out_count,
        }

    def fetch_overdue_contacts(state: FollowupState) -> dict:
        try:
            overdue = fetch_overdue(days=overdue_days)
        except Exception as e:
            return {"errors": state["errors"] + [f"fetch_overdue: {e}"], "overdue_contacts": []}
        return {"overdue_contacts": overdue}

    def draft_followup_emails(state: FollowupState) -> dict:
        emails_to_send = list(state.get("emails_to_send", []))
        for contact in state.get("overdue_contacts", []):
            if not contact.get("email"):
                continue
            language = contact.get("preferred_language") or mission.language_default
            days_since = contact.get("days_since_contact", overdue_days)
            original_subject = contact.get("last_subject", "")
            system, user = draft_followup_prompt(mission, contact, days_since, language, original_subject)
            try:
                response = llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
                draft = parse_json_response(response.content)
                emails_to_send.append({
                    "contact_id": contact["id"],
                    "to_email": contact["email"],
                    "subject": draft.get("subject", ""),
                    "body": draft.get("body", ""),
                    "type": "followup",
                })
            except Exception:
                pass  # individual draft failures don't stop the run
        return {"emails_to_send": emails_to_send}

    def send_all_emails(state: FollowupState) -> dict:
        sent = 0
        for email in state.get("emails_to_send", []):
            if not email.get("to_email") or not email.get("body"):
                continue
            try:
                success = send_email(
                    to_email=email["to_email"],
                    subject=email.get("subject", ""),
                    body=email["body"],
                )
                if success and email.get("contact_id"):
                    log_interaction(
                        contact_id=email["contact_id"],
                        method="email",
                        direction="outbound",
                        summary=email.get("subject", "follow-up sent"),
                        outcome="no_reply",
                    )
                if success:
                    sent += 1
            except Exception:
                pass
        return {"sent_count": sent}

    def generate_report(state: FollowupState) -> dict:
        inbox_count = len(state.get("inbox_messages", []))
        overdue_count = len(state.get("overdue_contacts", []))
        sent = state.get("sent_count", 0)
        opt_outs = state.get("opt_out_count", 0)
        errs = state.get("errors", [])

        summary = (
            f"followup_agent: {inbox_count} inbox messages processed, "
            f"{overdue_count} overdue contacts, "
            f"{sent} emails sent, {opt_outs} opt-outs recorded"
        )
        if errs:
            summary += f", {len(errs)} error(s)"

        finish_run(
            state.get("run_id", 0),
            "completed",
            summary,
            {
                "inbox_processed": inbox_count,
                "overdue_handled": overdue_count,
                "sent": sent,
                "opt_outs": opt_outs,
            },
        )
        return {"summary": summary}

    graph = StateGraph(FollowupState)
    graph.add_node("init", init)
    graph.add_node("fetch_inbox_messages", fetch_inbox_messages)
    graph.add_node("classify_replies", classify_replies)
    graph.add_node("fetch_overdue_contacts", fetch_overdue_contacts)
    graph.add_node("draft_followup_emails", draft_followup_emails)
    graph.add_node("send_all_emails", send_all_emails)
    graph.add_node("generate_report", generate_report)

    graph.set_entry_point("init")
    graph.add_edge("init", "fetch_inbox_messages")
    graph.add_edge("fetch_inbox_messages", "classify_replies")
    graph.add_edge("classify_replies", "fetch_overdue_contacts")
    graph.add_edge("fetch_overdue_contacts", "draft_followup_emails")
    graph.add_edge("draft_followup_emails", "send_all_emails")
    graph.add_edge("send_all_emails", "generate_report")
    graph.add_edge("generate_report", END)

    return graph.compile()
