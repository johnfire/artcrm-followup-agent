from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END

from .protocols import (
    AgentMission, LanguageModel, InboxFetcher, ContactMatcher,
    InteractionLogger, OptOutSetter, InboxMessageMarker,
    OverdueFetcher, EmailSender, ApprovalQueuer, RunStarter, RunFinisher,
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
    queue_for_approval: ApprovalQueuer,
    start_run: RunStarter,
    finish_run: RunFinisher,
    mission: AgentMission,
    overdue_days: int = 90,
):
    """
    Build and return a compiled LangGraph follow-up agent.

    Processes two work streams per run:
      1. Inbox replies — reads unread emails, skips any that don't match a known
         contact, classifies the rest, logs interactions, flags opt-outs, and
         sends replies autonomously for interested contacts.
      2. Overdue contacts — finds contacted contacts with no reply after
         `overdue_days` days and queues a brief nudge for human approval.

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
            "queued_count": 0,
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
        - Skip (mark processed) if no match — it's not one of our contacts
        - Classify the reply with the LLM
        - If opt_out: flag immediately
        - If interested: draft a reply and send autonomously
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

            # Not one of our contacts — mark processed and skip entirely
            # Don't waste LLM tokens classifying newsletters or personal email
            if contact is None:
                try:
                    mark_message_processed(msg["id"], None)
                except Exception:
                    pass
                continue

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
                "contact_id": contact["id"],
                "from_email": msg["from_email"],
                "classification": classification,
                "reasoning": reasoning,
            }

            # Handle opt-out immediately
            if classification == "opt_out":
                try:
                    set_opt_out(contact["id"])
                    opt_out_count += 1
                except Exception as e:
                    entry["error"] = f"set_opt_out: {e}"

            # Log the interaction
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

            # Draft and send a reply for interested contacts
            # These are time-sensitive — someone just said yes — so send autonomously
            if classification == "interested" and contact.get("email"):
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
                mark_message_processed(msg["id"], contact["id"])
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

    def queue_followup_drafts(state: FollowupState) -> dict:
        """
        Draft follow-up nudges for overdue contacts and put them in the
        approval queue — not sent autonomously. You review and approve.
        """
        run_id = state.get("run_id", 0)
        queued = 0
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
                queue_for_approval(
                    contact_id=contact["id"],
                    run_id=run_id,
                    subject=draft.get("subject", ""),
                    body=draft.get("body", ""),
                )
                queued += 1
            except Exception:
                pass
        return {"queued_count": queued}

    def send_all_emails(state: FollowupState) -> dict:
        """Send replies to interested contacts. Overdue nudges go through approval instead."""
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
                        summary=email.get("subject", "reply sent"),
                        outcome="no_reply",
                    )
                if success:
                    sent += 1
            except Exception:
                pass
        return {"sent_count": sent}

    def generate_report(state: FollowupState) -> dict:
        inbox_count = len(state.get("classified_replies", []))
        overdue_count = len(state.get("overdue_contacts", []))
        sent = state.get("sent_count", 0)
        queued = state.get("queued_count", 0)
        opt_outs = state.get("opt_out_count", 0)
        errs = state.get("errors", [])

        summary = (
            f"followup_agent: {inbox_count} replies processed, "
            f"{overdue_count} overdue contacts, "
            f"{sent} replies sent, {queued} follow-ups queued for approval, "
            f"{opt_outs} opt-outs recorded"
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
                "queued": queued,
                "opt_outs": opt_outs,
            },
        )
        return {"summary": summary}

    graph = StateGraph(FollowupState)
    graph.add_node("init", init)
    graph.add_node("fetch_inbox_messages", fetch_inbox_messages)
    graph.add_node("classify_replies", classify_replies)
    graph.add_node("fetch_overdue_contacts", fetch_overdue_contacts)
    graph.add_node("queue_followup_drafts", queue_followup_drafts)
    graph.add_node("send_all_emails", send_all_emails)
    graph.add_node("generate_report", generate_report)

    graph.set_entry_point("init")
    graph.add_edge("init", "fetch_inbox_messages")
    graph.add_edge("fetch_inbox_messages", "classify_replies")
    graph.add_edge("classify_replies", "fetch_overdue_contacts")
    graph.add_edge("fetch_overdue_contacts", "queue_followup_drafts")
    graph.add_edge("queue_followup_drafts", "send_all_emails")
    graph.add_edge("send_all_emails", "generate_report")
    graph.add_edge("generate_report", END)

    return graph.compile()
