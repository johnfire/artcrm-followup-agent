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
    website: str = "https://example.com"


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
    "status": "contacted",
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
    "status": "contacted",
}

DRAFT = '{"subject": "Re: Danke", "body": "Vielen Dank für Ihr Interesse..."}'
FOLLOWUP_DRAFT = '{"subject": "Kurze Nachfrage", "body": "Ich wollte kurz nachfragen..."}'


def make_tools(
    inbox=None,
    contact_for_email=None,
    overdue=None,
):
    opt_outs = []
    interactions = []
    warm_outcomes = []
    visit_flags = []
    classifications = []
    queued = []
    runs = {}

    def fetch_inbox():
        return inbox if inbox is not None else [SAMPLE_MESSAGE]

    def match_contact(from_email):
        return contact_for_email

    def log_interaction(contact_id, method, direction, summary, outcome):
        interactions.append({"contact_id": contact_id, "direction": direction, "outcome": outcome})

    def set_opt_out(contact_id):
        opt_outs.append(contact_id)

    def handle_bounce(contact_id):
        pass

    def record_warm_outcome(contact_id):
        warm_outcomes.append(contact_id)

    def set_visit_when_nearby(contact_id):
        visit_flags.append(contact_id)

    def save_inbox_classification(inbox_message_id, contact_id, classification, reasoning):
        classifications.append({"inbox_message_id": inbox_message_id, "classification": classification})

    def fetch_overdue(days=90):
        return overdue if overdue is not None else []

    def queue_for_approval(contact_id, run_id, subject, body):
        queued.append({"contact_id": contact_id, "subject": subject})
        return len(queued)

    def start_run(agent_name, input_data):
        run_id = len(runs) + 1
        runs[run_id] = {"status": "running"}
        return run_id

    def finish_run(run_id, status, summary, output_data):
        runs[run_id]["status"] = status

    return (
        fetch_inbox, match_contact, log_interaction, set_opt_out,
        handle_bounce, record_warm_outcome, set_visit_when_nearby,
        save_inbox_classification, fetch_overdue, queue_for_approval,
        start_run, finish_run,
        opt_outs, interactions, warm_outcomes, visit_flags, queued, runs,
    )


def make_agent(llm, **tool_overrides):
    (
        fetch_inbox, match_contact, log_interaction, set_opt_out,
        handle_bounce, record_warm_outcome, set_visit_when_nearby,
        save_inbox_classification, fetch_overdue, queue_for_approval,
        start_run, finish_run,
        opt_outs, interactions, warm_outcomes, visit_flags, queued, runs,
    ) = make_tools(**tool_overrides)

    agent = create_followup_agent(
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
        mission=DummyMission(),
    )
    return agent, opt_outs, interactions, warm_outcomes, visit_flags, queued


def test_interested_reply_logs_and_queues():
    llm = FakeLLM([
        '{"classification": "interested", "reasoning": "They want to meet"}',
        DRAFT,
    ])
    agent, opt_outs, interactions, warm_outcomes, visit_flags, queued = make_agent(
        llm=llm,
        contact_for_email=SAMPLE_CONTACT,
    )

    result = agent.invoke({})

    assert any(i["direction"] == "inbound" and i["outcome"] == "interested" for i in interactions)
    assert result["queued_count"] == 1
    assert queued[0]["contact_id"] == 10
    assert opt_outs == []
    assert result["opt_out_count"] == 0
    # warm outcome recorded for interested
    assert 10 in warm_outcomes


def test_warm_reply_flags_visit_and_queues():
    llm = FakeLLM([
        '{"classification": "warm", "reasoning": "Friendly but not ready"}',
        DRAFT,
    ])
    agent, opt_outs, interactions, warm_outcomes, visit_flags, queued = make_agent(
        llm=llm,
        contact_for_email=SAMPLE_CONTACT,
    )

    result = agent.invoke({})

    assert any(i["outcome"] == "warm" for i in interactions)
    assert 10 in visit_flags
    assert result["warm_count"] == 1
    assert result["queued_count"] == 1
    # warm outcome recorded
    assert 10 in warm_outcomes


def test_opt_out_reply_sets_flag_and_does_not_queue():
    llm = FakeLLM(['{"classification": "opt_out", "reasoning": "Asked to be removed"}'])
    agent, opt_outs, interactions, warm_outcomes, visit_flags, queued = make_agent(
        llm=llm,
        contact_for_email=SAMPLE_CONTACT,
    )

    result = agent.invoke({})

    assert 10 in opt_outs
    assert result["opt_out_count"] == 1
    assert result["queued_count"] == 0
    assert queued == []
    # no warm outcome for opt_out
    assert warm_outcomes == []


def test_not_interested_reply_logs_and_does_not_queue():
    llm = FakeLLM(['{"classification": "not_interested", "reasoning": "Not interested"}'])
    agent, opt_outs, interactions, warm_outcomes, visit_flags, queued = make_agent(
        llm=llm,
        contact_for_email=SAMPLE_CONTACT,
    )

    result = agent.invoke({})

    assert any(i["outcome"] == "rejected" for i in interactions)
    assert result["queued_count"] == 0
    assert opt_outs == []
    assert warm_outcomes == []


def test_overdue_contact_gets_followup_queued():
    llm = FakeLLM([FOLLOWUP_DRAFT])
    agent, opt_outs, interactions, warm_outcomes, visit_flags, queued = make_agent(
        llm=llm,
        inbox=[],
        overdue=[OVERDUE_CONTACT],
    )

    result = agent.invoke({})

    assert result["queued_count"] == 1
    assert queued[0]["contact_id"] == 20


def test_empty_inbox_and_no_overdue():
    llm = FakeLLM(["{}"])
    agent, opt_outs, interactions, warm_outcomes, visit_flags, queued = make_agent(
        llm=llm,
        inbox=[],
        overdue=[],
    )

    result = agent.invoke({})

    assert result["queued_count"] == 0
    assert result["opt_out_count"] == 0
    assert "0 replies processed" in result["summary"]


def test_unmatched_inbox_message_skipped():
    """Message from unknown sender: skipped, no interaction logged."""
    llm = FakeLLM(['{"classification": "interested", "reasoning": "Interested"}'])
    agent, opt_outs, interactions, warm_outcomes, visit_flags, queued = make_agent(
        llm=llm,
        contact_for_email=None,
    )

    result = agent.invoke({})

    assert interactions == []
    assert result["queued_count"] == 0
    assert warm_outcomes == []
