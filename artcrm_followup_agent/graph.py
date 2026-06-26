"""
Followup agent.

Processes two work streams per run:
  1. Inbox replies — reads unread emails, matches them to known post-outreach
     contacts, classifies via LLM, and queues reply drafts for human approval.
  2. Overdue contacts — finds contacted contacts with no reply after
     `overdue_days` days and queues a brief nudge for human approval.

Nothing is sent until a human approves via the UI.

Pipeline position: research → enrich → scout → outreach → followup
"""
import logging
import re
from dataclasses import dataclass

from langchain_core.messages import SystemMessage, HumanMessage

from .protocols import (
    AgentMission, LanguageModel, InboxFetcher, ContactMatcher,
    InteractionLogger, OptOutSetter, BounceHandler, VisitFlagSetter,
    InboxClassificationSaver, OverdueFetcher, ApprovalQueuer, RunStarter, RunFinisher,
    WarmOutcomeRecorder,
)
from .prompts import (
    classify_reply_prompt, draft_interested_reply_prompt,
    draft_warm_reply_prompt, draft_followup_prompt,
    REPLY_CLASSIFICATIONS,
)
from ._utils import parse_json_response

logger = logging.getLogger(__name__)

# Statuses that mean we have already sent outreach to this contact.
POST_OUTREACH_STATUSES = {
    "contacted", "meeting", "networking_visit", "dormant",
    "on_hold", "bad_email", "proposal",
}

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

_OUTCOME_MAP = {
    "interested":     "interested",
    "warm":           "warm",
    "not_interested": "rejected",
    "not_possible":   "not_possible",
    "opt_out":        "opt_out",
    "other":          "no_reply",
}


@dataclass
class _ReplyTotals:
    """Side-effect counters accumulated while classifying inbox replies (L-3)."""
    queued: int = 0
    opt_outs: int = 0
    warm: int = 0
    bounces: int = 0


def _is_bounce(msg: dict) -> bool:
    return bool(
        _BOUNCE_SENDERS.search(msg.get("from_email", ""))
        or _BOUNCE_SUBJECTS.search(msg.get("subject", ""))
    )


def _extract_recipient_emails(msg: dict) -> list[str]:
    """Deduplicated emails from bounce body — one is likely the failed recipient."""
    return list(dict.fromkeys(_EMAIL_RE.findall(msg.get("body", ""))))


class _FollowupAgent:
    def __init__(
        self,
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
        self._llm = llm
        self._fetch_inbox = fetch_inbox
        self._match_contact = match_contact
        self._log_interaction = log_interaction
        self._set_opt_out = set_opt_out
        self._handle_bounce = handle_bounce
        self._record_warm_outcome = record_warm_outcome
        self._set_visit_when_nearby = set_visit_when_nearby
        self._save_inbox_classification = save_inbox_classification
        self._fetch_overdue = fetch_overdue
        self._queue_for_approval = queue_for_approval
        self._start_run = start_run
        self._finish_run = finish_run
        self._mission = mission
        self._overdue_days = overdue_days

    def invoke(self, inputs: dict) -> dict:
        run_id = self._start_run("followup_agent", {})
        errors = []

        inbox_messages = self._fetch_inbox_messages(errors)
        classified_replies, totals = self._classify_replies(inbox_messages, run_id)

        overdue_contacts = self._fetch_overdue_contacts(errors)
        queued_count = self._queue_followup_drafts(overdue_contacts, run_id, totals.queued)

        inbox_count = len(classified_replies)
        overdue_count = len(overdue_contacts)
        summary = (
            f"followup_agent: {inbox_count} replies processed, "
            f"{overdue_count} overdue contacts, "
            f"{queued_count} drafts queued for approval, "
            f"{totals.warm} warm replies flagged for visit, "
            f"{totals.opt_outs} opt-outs recorded, "
            f"{totals.bounces} bounces marked as bad_email"
        )
        if errors:
            summary += f", {len(errors)} error(s)"

        self._finish_run(
            run_id, "completed", summary,
            {
                "inbox_processed": inbox_count,
                "overdue_handled": overdue_count,
                "queued": queued_count,
                "warm": totals.warm,
                "opt_outs": totals.opt_outs,
                "bounces": totals.bounces,
            },
        )
        logger.info(summary)
        return {
            "summary": summary,
            "queued_count": queued_count,
            "opt_out_count": totals.opt_outs,
            "warm_count": totals.warm,
            "bounce_count": totals.bounces,
        }

    def _fetch_inbox_messages(self, errors: list) -> list[dict]:
        try:
            return self._fetch_inbox()
        except Exception as e:
            errors.append(f"fetch_inbox: {e}")
            return []

    def _fetch_overdue_contacts(self, errors: list) -> list[dict]:
        try:
            return self._fetch_overdue(days=self._overdue_days)
        except Exception as e:
            errors.append(f"fetch_overdue: {e}")
            return []

    def _classify_replies(
        self, inbox_messages: list[dict], run_id: int
    ) -> tuple[list[dict], _ReplyTotals]:
        """Classify each inbox message, accumulating side-effect counts in `_ReplyTotals`.

        Returns the per-message classification entries and the aggregate totals.
        """
        totals = _ReplyTotals()
        classified = []
        for msg in inbox_messages:
            entry = self._classify_one_reply(msg, run_id, totals)
            if entry is not None:
                classified.append(entry)
        return classified, totals

    def _classify_one_reply(self, msg: dict, run_id: int, totals: _ReplyTotals) -> dict | None:
        """Handle a single inbox message; mutates `totals`. Returns its classification
        entry, or None when the message is a bounce or is skipped (no entry recorded)."""
        if _is_bounce(msg):
            totals.bounces += self._handle_bounce_message(msg)
            return None

        contact = None
        try:
            contact = self._match_contact(msg["from_email"])
        except Exception as e:
            logger.warning("followup: match_contact failed for %s: %s", msg.get("from_email"), e)

        if contact is None:
            self._save_classification(msg["id"], None, "skipped", "no matching contact")
            return None

        if contact.get("status") not in POST_OUTREACH_STATUSES:
            self._save_classification(
                msg["id"], contact["id"], "skipped",
                f"contact status '{contact.get('status')}' is pre-outreach",
            )
            return None

        classification, reasoning = self._classify_message(msg)
        entry = {
            "inbox_message_id": msg["id"],
            "contact_id": contact["id"],
            "from_email": msg["from_email"],
            "classification": classification,
            "reasoning": reasoning,
        }

        self._apply_reply_side_effects(msg, contact, classification, run_id, totals, entry)

        self._save_classification(msg["id"], contact["id"], classification, reasoning)
        return entry

    def _apply_reply_side_effects(
        self, msg: dict, contact: dict, classification: str,
        run_id: int, totals: _ReplyTotals, entry: dict,
    ) -> None:
        """Run the DB side-effects gated by a reply's classification, recording counts
        on `totals` and outcomes/errors on `entry`."""
        if classification == "opt_out":
            try:
                self._set_opt_out(contact["id"])
                totals.opt_outs += 1
            except Exception as e:
                entry["error"] = f"set_opt_out: {e}"

        if classification == "warm":
            try:
                self._set_visit_when_nearby(contact["id"])
                totals.warm += 1
                entry["visit_flagged"] = True
            except Exception as e:
                entry["error"] = f"set_visit_when_nearby: {e}"

        interaction_logged = False
        try:
            self._log_interaction(
                contact_id=contact["id"],
                method="email",
                direction="inbound",
                summary=f"{classification}: {msg.get('subject', '')}",
                outcome=_OUTCOME_MAP.get(classification, "no_reply"),
            )
            interaction_logged = True
        except Exception as e:
            logger.warning("log_interaction failed: contact_id=%s error=%s", contact.get("id"), e)

        if interaction_logged and classification in ("interested", "warm"):
            try:
                self._record_warm_outcome(contact["id"])
            except Exception as e:
                logger.warning("record_warm_outcome failed: contact_id=%s error=%s", contact.get("id"), e)

        if classification in ("interested", "warm") and contact.get("email"):
            self._draft_and_queue_reply(msg, contact, classification, run_id, totals, entry)

    def _draft_and_queue_reply(
        self, msg: dict, contact: dict, classification: str,
        run_id: int, totals: _ReplyTotals, entry: dict,
    ) -> None:
        """Draft an interested/warm reply via the LLM and queue it for human approval."""
        language = contact.get("preferred_language") or self._mission.language_default
        if classification == "interested":
            sys_p, usr_p = draft_interested_reply_prompt(self._mission, contact, msg, language)
        else:
            sys_p, usr_p = draft_warm_reply_prompt(self._mission, contact, msg, language)
        try:
            draft_resp = self._llm.invoke([SystemMessage(content=sys_p), HumanMessage(content=usr_p)])
            draft = parse_json_response(draft_resp.content)
            self._queue_for_approval(
                contact_id=contact["id"],
                run_id=run_id,
                subject=draft.get("subject", ""),
                body=draft.get("body", ""),
            )
            totals.queued += 1
            entry["reply_queued"] = True
        except Exception as e:
            entry["draft_error"] = str(e)

    def _save_classification(self, message_id, contact_id, classification: str, reasoning: str) -> None:
        """Persist an inbox classification, logging (not raising) on failure."""
        try:
            self._save_inbox_classification(message_id, contact_id, classification, reasoning)
        except Exception as e:
            logger.warning("followup: save_inbox_classification failed for msg %s: %s", message_id, e)

    def _classify_message(self, msg: dict) -> tuple[str, str]:
        """Ask the LLM to classify an inbox reply. Returns (classification, reasoning).

        The classification gates unattended DB side-effects (opt_out / visit flags), so
        any value not in the strict enum is coerced to "other" (no side-effects). This,
        with the delimiter fencing in the prompt, blocks prompt-injection from the
        untrusted email body flipping a CRM decision (H-3).
        """
        system, user = classify_reply_prompt(self._mission, msg)
        try:
            response = self._llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
            result = parse_json_response(response.content)
            classification = result.get("classification", "other")
            if classification not in REPLY_CLASSIFICATIONS:
                logger.warning("followup: off-enum classification %r coerced to 'other'", classification)
                classification = "other"
            return classification, result.get("reasoning", "")
        except Exception as e:
            return "other", f"classification error: {e}"

    def _handle_bounce_message(self, msg: dict) -> int:
        """Process a bounce notification. Returns 1 if a contact was marked bad_email, else 0."""
        candidate_emails = _extract_recipient_emails(msg)
        bounced_contact = None
        for email in candidate_emails:
            try:
                c = self._match_contact(email)
                if c and c.get("status") in POST_OUTREACH_STATUSES:
                    bounced_contact = c
                    break
            except Exception as e:
                logger.warning("followup: match_contact failed for bounce %s: %s", email, e)

        if bounced_contact:
            try:
                self._handle_bounce(bounced_contact["id"])
            except Exception as e:
                logger.warning("followup: handle_bounce failed for contact %s: %s", bounced_contact.get("id"), e)
            try:
                self._save_inbox_classification(
                    msg["id"], bounced_contact["id"], "bounce",
                    f"Delivery failure for {bounced_contact.get('email', '')}",
                )
            except Exception as e:
                logger.warning("followup: save_inbox_classification failed for msg %s: %s", msg.get("id"), e)
            return 1
        else:
            try:
                self._save_inbox_classification(
                    msg["id"], None, "bounce", "No matching contact found in bounce body",
                )
            except Exception as e:
                logger.warning("followup: save_inbox_classification failed for msg %s: %s", msg.get("id"), e)
            return 0

    def _queue_followup_drafts(
        self, overdue_contacts: list[dict], run_id: int, queued_count: int
    ) -> int:
        for contact in overdue_contacts:
            if not contact.get("email"):
                continue
            language = contact.get("preferred_language") or self._mission.language_default
            days_since = contact.get("days_since_contact", self._overdue_days)
            original_subject = contact.get("last_subject", "")
            system, user = draft_followup_prompt(self._mission, contact, days_since, language, original_subject)
            try:
                response = self._llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
                draft = parse_json_response(response.content)
                self._queue_for_approval(
                    contact_id=contact["id"],
                    run_id=run_id,
                    subject=draft.get("subject", ""),
                    body=draft.get("body", ""),
                )
                queued_count += 1
            except Exception as e:
                logger.warning("followup: failed to queue followup draft for contact %s: %s", contact.get("id"), e)
        return queued_count


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
) -> _FollowupAgent:
    """
    Build and return a followup agent.

    Processes inbox replies (classify + draft) and overdue contacts (nudge).
    All outgoing email requires human approval — nothing is sent autonomously.

    Usage:
        agent = create_followup_agent(llm=..., fetch_inbox=..., ...)
        result = agent.invoke({})
        print(result["summary"])
    """
    return _FollowupAgent(
        llm=llm,
        fetch_inbox=fetch_inbox,
        match_contact=match_contact,
        log_interaction=log_interaction,
        set_opt_out=set_opt_out,
        handle_bounce=handle_bounce,
        record_warm_outcome=record_warm_outcome,
        set_visit_when_nearby=set_visit_when_nearby,
        save_inbox_classification=save_inbox_classification,
        fetch_overdue=fetch_overdue,
        queue_for_approval=queue_for_approval,
        start_run=start_run,
        finish_run=finish_run,
        mission=mission,
        overdue_days=overdue_days,
    )
