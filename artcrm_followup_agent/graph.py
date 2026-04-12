import logging
import re
from langchain_core.messages import SystemMessage, HumanMessage

logger = logging.getLogger(__name__)
from langgraph.graph import StateGraph, END

from .protocols import (
    AgentMission, LanguageModel, InboxFetcher, ContactMatcher,
    InteractionLogger, OptOutSetter, BounceHandler, VisitFlagSetter,
    InboxClassificationSaver, OverdueFetcher, ApprovalQueuer, RunStarter, RunFinisher,
    WarmOutcomeRecorder,
)

# Patterns that indicate a delivery failure notification.
_BOUNCE_SENDERS = re.compile(
    r"(mailer-daemon|postmaster|delivery.notification|"
    r"mail-daemon|noreply\+bounce|mailerdaemon)",
    re.IGNORECASE,
)
_BOUNCE_SUBJECTS = re.compile(
    r"(undelivered mail|delivery (status notification|failed|failure)|"
    r"returned mail|failure notice|mail delivery failed|"
    r"unzustellbar|zustellungs(fehler|benachrichtigung)|"
    r"nicht zugestellt)",
    re.IGNORECASE,
)
_EMAIL_RE = re.compile(r"[\w.+\-]+@[\w.\-]+\.[a-z]{2,}", re.IGNORECASE)


def _is_bounce(msg: dict) -> bool:
    """Return True if the message looks like a delivery failure notification."""
    return bool(
        _BOUNCE_SENDERS.search(msg.get("from_email", ""))
        or _BOUNCE_SUBJECTS.search(msg.get("subject", ""))
    )


def _extract_recipient_emails(msg: dict) -> list[str]:
    """Extract all email addresses from the bounce body — one is likely the failed recipient."""
    body = msg.get("body", "")
    return list(dict.fromkeys(_EMAIL_RE.findall(body)))  # deduplicated, order preserved
from .state import FollowupState
from .prompts import (
    classify_reply_prompt, draft_interested_reply_prompt,
    draft_warm_reply_prompt, draft_followup_prompt,
)
from ._utils import parse_json_response

# Statuses that mean we have already sent outreach to this contact.
# Only process inbox replies from contacts in these states.
POST_OUTREACH_STATUSES = {
    "contacted", "meeting", "networking_visit", "dormant",
    "on_hold", "bad_email", "proposal",
}


def create_followup_agent(
    llm: LanguageModel,
    fetch_inbox: InboxFetcher,
    match_contact: ContactMatcher,
    log_interaction: InteractionLogger,
    set_opt_out: OptOutSetter,
    handle_bounce: BounceHandler,
    record_warm_outcome: WarmOutcomeRecorder,
    set_visit_when_nearby: VisitFlagSetter,
    save_inbox_classification: InboxClassificationSaver,
    fetch_overdue: OverdueFetcher,
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
         post-outreach contact, classifies the rest, and queues reply drafts for
         human approval. All outgoing email requires approval — nothing is sent
         autonomously. Warm replies flag the contact for a personal visit.
      2. Overdue contacts — finds contacted contacts with no reply after
         `overdue_days` days and queues a brief nudge for human approval.
    """

    def init(state: FollowupState) -> dict:
        run_id = start_run("followup_agent", {})
        return {
            "run_id": run_id,
            "inbox_messages": [],
            "classified_replies": [],
            "overdue_contacts": [],
            "errors": [],
            "queued_count": 0,
            "opt_out_count": 0,
            "warm_count": 0,
            "bounce_count": 0,
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
        - Match to a contact by sender email
        - Skip if no match — not one of our contacts
        - Skip if contact status is pre-outreach (cold/candidate) — we didn't email them
        - Classify the reply
        - Handle opt_out: flag contact immediately
        - Handle warm: set visit_when_nearby = True
        - Queue reply draft for human approval (interested and warm get a draft)
        - Log interaction
        - Save classification back to inbox_messages row
        """
        classified = []
        queued = state.get("queued_count", 0)
        opt_out_count = 0
        warm_count = 0
        bounce_count = 0

        for msg in state.get("inbox_messages", []):

            # ── Bounce detection — no LLM needed ─────────────────────────────
            if _is_bounce(msg):
                # Extract emails from bounce body, try to match a known contact
                candidate_emails = _extract_recipient_emails(msg)
                bounced_contact = None
                for email in candidate_emails:
                    try:
                        c = match_contact(email)
                        if c and c.get("status") in POST_OUTREACH_STATUSES:
                            bounced_contact = c
                            break
                    except Exception:
                        pass

                if bounced_contact:
                    try:
                        handle_bounce(bounced_contact["id"])
                        bounce_count += 1
                    except Exception as e:
                        pass
                    try:
                        save_inbox_classification(
                            msg["id"], bounced_contact["id"], "bounce",
                            f"Delivery failure for {bounced_contact.get('email', '')}",
                        )
                    except Exception:
                        pass
                else:
                    # Bounce but no matching contact found — just mark it processed
                    try:
                        save_inbox_classification(
                            msg["id"], None, "bounce", "No matching contact found in bounce body",
                        )
                    except Exception:
                        pass
                continue  # skip LLM classification for bounces
            # ─────────────────────────────────────────────────────────────────
            contact = None
            try:
                contact = match_contact(msg["from_email"])
            except Exception:
                pass

            # Not one of our contacts — mark and skip
            if contact is None:
                try:
                    save_inbox_classification(msg["id"], None, "skipped", "no matching contact")
                except Exception:
                    pass
                continue

            # Contact exists but we never emailed them — leave it alone
            if contact.get("status") not in POST_OUTREACH_STATUSES:
                try:
                    save_inbox_classification(
                        msg["id"], contact["id"], "skipped",
                        f"contact status '{contact.get('status')}' is pre-outreach"
                    )
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

            # Handle opt_out
            if classification == "opt_out":
                try:
                    set_opt_out(contact["id"])
                    opt_out_count += 1
                except Exception as e:
                    entry["error"] = f"set_opt_out: {e}"

            # Handle warm — flag for personal visit
            if classification == "warm":
                try:
                    set_visit_when_nearby(contact["id"])
                    warm_count += 1
                    entry["visit_flagged"] = True
                except Exception as e:
                    entry["error"] = f"set_visit_when_nearby: {e}"

            # Log the interaction
            outcome_map = {
                "interested":    "interested",
                "warm":          "warm",
                "not_interested": "rejected",
                "not_possible":  "not_possible",
                "opt_out":       "opt_out",
                "other":         "no_reply",
            }
            interaction_logged = False
            try:
                log_interaction(
                    contact_id=contact["id"],
                    method="email",
                    direction="inbound",
                    summary=f"{classification}: {msg.get('subject', '')}",
                    outcome=outcome_map.get(classification, "no_reply"),
                )
                interaction_logged = True
            except Exception as e:
                logger.warning("log_interaction failed: contact_id=%s error=%s", contact.get("id"), e)

            # Record warm signal for outreach quality loop — only if interaction was committed
            if interaction_logged and classification in ("interested", "warm"):
                try:
                    record_warm_outcome(contact["id"])
                except Exception as e:
                    logger.warning("record_warm_outcome failed: contact_id=%s error=%s", contact.get("id"), e)

            # Draft a reply for interested and warm contacts — queue for approval
            if classification in ("interested", "warm") and contact.get("email"):
                language = contact.get("preferred_language") or mission.language_default
                if classification == "interested":
                    sys_p, usr_p = draft_interested_reply_prompt(mission, contact, msg, language)
                else:
                    sys_p, usr_p = draft_warm_reply_prompt(mission, contact, msg, language)
                try:
                    draft_resp = llm.invoke([SystemMessage(content=sys_p), HumanMessage(content=usr_p)])
                    draft = parse_json_response(draft_resp.content)
                    queue_for_approval(
                        contact_id=contact["id"],
                        run_id=state.get("run_id", 0),
                        subject=draft.get("subject", ""),
                        body=draft.get("body", ""),
                    )
                    queued += 1
                    entry["reply_queued"] = True
                except Exception as e:
                    entry["draft_error"] = str(e)

            # Save classification to inbox_messages
            try:
                save_inbox_classification(msg["id"], contact["id"], classification, reasoning)
            except Exception:
                pass

            classified.append(entry)

        return {
            "classified_replies": classified,
            "queued_count": queued,
            "opt_out_count": opt_out_count,
            "warm_count": warm_count,
            "bounce_count": bounce_count,
        }

    def fetch_overdue_contacts(state: FollowupState) -> dict:
        try:
            overdue = fetch_overdue(days=overdue_days)
        except Exception as e:
            return {"errors": state["errors"] + [f"fetch_overdue: {e}"], "overdue_contacts": []}
        return {"overdue_contacts": overdue}

    def queue_followup_drafts(state: FollowupState) -> dict:
        """Draft follow-up nudges for overdue contacts. All go to approval queue."""
        run_id = state.get("run_id", 0)
        queued = state.get("queued_count", 0)
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

    def generate_report(state: FollowupState) -> dict:
        inbox_count = len(state.get("classified_replies", []))
        overdue_count = len(state.get("overdue_contacts", []))
        queued = state.get("queued_count", 0)
        opt_outs = state.get("opt_out_count", 0)
        warm = state.get("warm_count", 0)
        bounces = state.get("bounce_count", 0)
        errs = state.get("errors", [])

        summary = (
            f"followup_agent: {inbox_count} replies processed, "
            f"{overdue_count} overdue contacts, "
            f"{queued} drafts queued for approval, "
            f"{warm} warm replies flagged for visit, "
            f"{opt_outs} opt-outs recorded, "
            f"{bounces} bounces marked as bad_email"
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
                "queued": queued,
                "warm": warm,
                "opt_outs": opt_outs,
                "bounces": bounces,
            },
        )
        return {"summary": summary}

    graph = StateGraph(FollowupState)
    graph.add_node("init", init)
    graph.add_node("fetch_inbox_messages", fetch_inbox_messages)
    graph.add_node("classify_replies", classify_replies)
    graph.add_node("fetch_overdue_contacts", fetch_overdue_contacts)
    graph.add_node("queue_followup_drafts", queue_followup_drafts)
    graph.add_node("generate_report", generate_report)

    graph.set_entry_point("init")
    graph.add_edge("init", "fetch_inbox_messages")
    graph.add_edge("fetch_inbox_messages", "classify_replies")
    graph.add_edge("classify_replies", "fetch_overdue_contacts")
    graph.add_edge("fetch_overdue_contacts", "queue_followup_drafts")
    graph.add_edge("queue_followup_drafts", "generate_report")
    graph.add_edge("generate_report", END)

    return graph.compile()
