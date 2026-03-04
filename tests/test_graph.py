"""
Tests use dummy implementations of every Protocol — no real LLM, DB, or network.
"""
from dataclasses import dataclass
from langchain_core.messages import AIMessage
from artcrm_followup_agent import create_followup_agent


@dataclass(frozen=True)
class DummyMission:
    goal: str = "Find art venues"
    identity: str = "Test Artist"
    targets: str = "galleries, cafes"
    fit_criteria: str = "contemporary art friendly"
    outreach_style: str = "personal"
    language_default: str = "de"


class FakeLLM:
    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self._index = 0

    def invoke(self, messages):
        response = self._responses[self._index % len(self._responses)]
        self._index += 1
        return AIMessage(content=response)


SAMPLE_MESSAGE = {
    "id": 1,
    "message_id": "<abc@proton.me>",
    "from_email": "gallery@example.com",
    "subject": "Re: Ausstellungsanfrage",
    "body": "Vielen Dank für Ihre Anfrage. Wir würden uns sehr freuen...",
    "received_at": "2026-03-01T10:00:00",
}

SAMPLE_CONTACT = {
    "id": 10,
    "name": "Galerie Nord",
    "city": "Munich",
    "type": "gallery",
    "email": "gallery@example.com",
    "preferred_language": "de",
}

OVERDUE_CONTACT = {
    "id": 20,
    "name": "Galerie Süd",
    "city": "Augsburg",
    "type": "gallery",
    "email": "sued@example.com",
    "preferred_language": "de",
    "days_since_contact": 95,
    "last_subject": "Ausstellungsanfrage",
}

DRAFT = '{"subject": "Re: Danke", "body": "Vielen Dank für Ihr Interesse..."}'
FOLLOWUP_DRAFT = '{"subject": "Kurze Nachfrage", "body": "Ich wollte kurz nachfragen..."}'


def make_tools(
    inbox=None,
    contact_for_email=None,
    overdue=None,
    send_result=True,
):
    opt_outs = []
    interactions = []
    marked = []
    sent = []
    runs = {}

    def fetch_inbox():
        return inbox if inbox is not None else [SAMPLE_MESSAGE]

    def match_contact(from_email):
        return contact_for_email

    def log_interaction(contact_id, method, direction, summary, outcome):
        interactions.append({"contact_id": contact_id, "direction": direction, "outcome": outcome})

    def set_opt_out(contact_id):
        opt_outs.append(contact_id)

    def mark_message_processed(inbox_message_id, contact_id):
        marked.append(inbox_message_id)

    def fetch_overdue(days=90):
        return overdue if overdue is not None else []

    def send_email(to_email, subject, body):
        sent.append({"to": to_email, "subject": subject})
        return send_result

    def start_run(agent_name, input_data):
        run_id = len(runs) + 1
        runs[run_id] = {"status": "running"}
        return run_id

    def finish_run(run_id, status, summary, output_data):
        runs[run_id]["status"] = status

    return (
        fetch_inbox, match_contact, log_interaction, set_opt_out,
        mark_message_processed, fetch_overdue, send_email,
        start_run, finish_run,
        opt_outs, interactions, marked, sent, runs,
    )


def make_agent(llm, **tool_overrides):
    (
        fetch_inbox, match_contact, log_interaction, set_opt_out,
        mark_message_processed, fetch_overdue, send_email,
        start_run, finish_run,
        opt_outs, interactions, marked, sent, runs,
    ) = make_tools(**tool_overrides)

    agent = create_followup_agent(
        llm=llm,
        fetch_inbox=fetch_inbox,
        match_contact=match_contact,
        log_interaction=log_interaction,
        set_opt_out=set_opt_out,
        mark_message_processed=mark_message_processed,
        fetch_overdue=fetch_overdue,
        send_email=send_email,
        start_run=start_run,
        finish_run=finish_run,
        mission=DummyMission(),
    )
    return agent, opt_outs, interactions, marked, sent


def test_interested_reply_logs_and_sends():
    llm = FakeLLM([
        '{"classification": "interested", "reasoning": "They want to meet"}',
        DRAFT,   # draft_reply response
    ])
    agent, opt_outs, interactions, marked, sent = make_agent(
        llm=llm,
        contact_for_email=SAMPLE_CONTACT,
    )

    result = agent.invoke({})

    # interaction logged as inbound/interested
    assert any(i["direction"] == "inbound" and i["outcome"] == "interested" for i in interactions)
    # reply email sent
    assert result["sent_count"] == 1
    assert sent[0]["to"] == "gallery@example.com"
    # no opt-out
    assert opt_outs == []
    assert result["opt_out_count"] == 0


def test_opt_out_reply_sets_flag_and_does_not_send():
    llm = FakeLLM(['{"classification": "opt_out", "reasoning": "Asked to be removed"}'])
    agent, opt_outs, interactions, marked, sent = make_agent(
        llm=llm,
        contact_for_email=SAMPLE_CONTACT,
    )

    result = agent.invoke({})

    assert 10 in opt_outs
    assert result["opt_out_count"] == 1
    assert result["sent_count"] == 0
    assert sent == []


def test_rejected_reply_logs_and_does_not_send():
    llm = FakeLLM(['{"classification": "rejected", "reasoning": "Not interested"}'])
    agent, opt_outs, interactions, marked, sent = make_agent(
        llm=llm,
        contact_for_email=SAMPLE_CONTACT,
    )

    result = agent.invoke({})

    assert any(i["outcome"] == "rejected" for i in interactions)
    assert result["sent_count"] == 0
    assert opt_outs == []


def test_overdue_contact_gets_followup_sent():
    llm = FakeLLM([
        # No inbox messages so classify_replies makes no LLM calls
        FOLLOWUP_DRAFT,  # draft_followup response
    ])
    agent, opt_outs, interactions, marked, sent = make_agent(
        llm=llm,
        inbox=[],
        overdue=[OVERDUE_CONTACT],
    )

    result = agent.invoke({})

    assert result["sent_count"] == 1
    assert sent[0]["to"] == "sued@example.com"


def test_empty_inbox_and_no_overdue():
    llm = FakeLLM(["{}"])
    agent, opt_outs, interactions, marked, sent = make_agent(
        llm=llm,
        inbox=[],
        overdue=[],
    )

    result = agent.invoke({})

    assert result["sent_count"] == 0
    assert result["opt_out_count"] == 0
    assert "0 inbox messages" in result["summary"]


def test_unmatched_inbox_message_still_processed():
    """Message from unknown sender: classified but no interaction logged, no opt-out."""
    llm = FakeLLM(['{"classification": "interested", "reasoning": "Interested"}'])
    agent, opt_outs, interactions, marked, sent = make_agent(
        llm=llm,
        contact_for_email=None,  # no contact match
    )

    result = agent.invoke({})

    # No interaction logged (no contact to link to)
    assert interactions == []
    # No email sent (no contact email to reply to)
    assert result["sent_count"] == 0
    # Message still marked as processed
    assert 1 in marked


def test_send_failure_does_not_crash():
    llm = FakeLLM([
        '{"classification": "interested", "reasoning": "Interested"}',
        DRAFT,
    ])
    agent, opt_outs, interactions, marked, sent = make_agent(
        llm=llm,
        contact_for_email=SAMPLE_CONTACT,
        send_result=False,  # SMTP fails
    )

    result = agent.invoke({})

    # Attempted but failed — no crash, sent_count stays 0
    assert result["sent_count"] == 0
