from typing import Any, Protocol


class AgentMission(Protocol):
    goal: str
    identity: str
    targets: str
    fit_criteria: str
    outreach_style: str
    language_default: str


class LanguageModel(Protocol):
    """Any LangChain-compatible chat model satisfies this."""
    def invoke(self, messages: list) -> Any:
        """Returns an object with a .content (str) attribute."""
        ...


class InboxFetcher(Protocol):
    """
    Read unprocessed incoming emails.
    Returns list of message dicts: {id, message_id, from_email, subject, body, received_at}.
    """
    def __call__(self) -> list[dict]: ...


class ContactMatcher(Protocol):
    """
    Given a sender email address, find the matching contact in the database.
    Returns contact dict or None if no match found.
    """
    def __call__(self, from_email: str) -> dict | None: ...


class InteractionLogger(Protocol):
    """Log a contact interaction to the database."""
    def __call__(
        self,
        contact_id: int,
        method: str,
        direction: str,
        summary: str,
        outcome: str,
    ) -> None: ...


class OptOutSetter(Protocol):
    """Mark a contact as opted out. No further outreach will be sent."""
    def __call__(self, contact_id: int) -> None: ...


class VisitFlagSetter(Protocol):
    """Set visit_when_nearby = True on a contact. Used for warm replies."""
    def __call__(self, contact_id: int) -> None: ...


class InboxClassificationSaver(Protocol):
    """Persist the LLM classification result back to the inbox_messages row."""
    def __call__(self, inbox_message_id: int, contact_id: int | None, classification: str, reasoning: str) -> None: ...


class OverdueFetcher(Protocol):
    """
    Return contacts that are overdue for follow-up.
    A contact is overdue if they were contacted more than `days` ago with no reply.
    """
    def __call__(self, days: int = 90) -> list[dict]: ...


class ApprovalQueuer(Protocol):
    """Insert a drafted email into the approval queue. Returns queue item id."""
    def __call__(self, contact_id: int, run_id: int, subject: str, body: str) -> int: ...


class RunStarter(Protocol):
    """Log the start of an agent run. Returns run_id."""
    def __call__(self, agent_name: str, input_data: dict) -> int: ...


class RunFinisher(Protocol):
    """Log the completion of an agent run."""
    def __call__(self, run_id: int, status: str, summary: str, output_data: dict) -> None: ...
