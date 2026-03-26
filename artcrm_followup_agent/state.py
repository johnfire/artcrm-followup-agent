from typing import TypedDict


class FollowupState(TypedDict):
    # --- working state ---
    run_id: int
    inbox_messages: list[dict]      # raw unprocessed messages from inbox

    # [{message_id, contact_id, classification, draft_subject?, draft_body?, to_email?}]
    classified_replies: list[dict]

    overdue_contacts: list[dict]

    # [{contact_id, to_email, subject, body}] — both reply drafts and proactive follow-ups
    emails_to_send: list[dict]

    errors: list[str]

    # --- output ---
    sent_count: int
    queued_count: int
    opt_out_count: int
    summary: str
