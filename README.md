# artcrm-followup-agent

LangGraph agent that monitors the inbox for replies and sends follow-up emails to overdue contacts. Fully autonomous — the daily activity feed in the UI is the human checkpoint.

## What it does

**Stream 1 — Inbox replies:**
- Reads unprocessed inbox messages
- Matches each message to a contact by sender email
- Classifies the reply: `interested` / `rejected` / `opt_out` / `other`
- Logs the interaction to the database
- If `opt_out`: flags the contact immediately, no further outreach ever
- If `interested`: drafts and sends a warm reply
- Marks each message as processed

**Stream 2 — Proactive follow-ups:**
- Fetches contacts overdue for follow-up (no reply after N days, default 90)
- Drafts a brief, non-pushy follow-up in the contact's preferred language
- Sends it and logs the interaction

## Usage

```python
from artcrm_followup_agent import create_followup_agent

agent = create_followup_agent(
    llm=your_llm,
    fetch_inbox=your_inbox_fn,
    match_contact=your_match_fn,
    log_interaction=your_log_fn,
    set_opt_out=your_opt_out_fn,
    mark_message_processed=your_mark_fn,
    fetch_overdue=your_overdue_fn,
    send_email=your_send_fn,
    start_run=your_start_run_fn,
    finish_run=your_finish_run_fn,
    mission=your_mission,
    overdue_days=90,    # optional, default 90
)

result = agent.invoke({})
print(result["summary"])
# "followup_agent: 3 inbox messages processed, 2 overdue contacts, 4 emails sent, 1 opt-outs recorded"
```

## Protocols

| Parameter | Protocol | Description |
|---|---|---|
| `llm` | `LanguageModel` | Any LangChain `BaseChatModel` |
| `fetch_inbox` | `InboxFetcher` | `() -> list[dict]` |
| `match_contact` | `ContactMatcher` | `(from_email: str) -> dict \| None` |
| `log_interaction` | `InteractionLogger` | `(contact_id, method, direction, summary, outcome) -> None` |
| `set_opt_out` | `OptOutSetter` | `(contact_id: int) -> None` |
| `mark_message_processed` | `InboxMessageMarker` | `(inbox_message_id, contact_id) -> None` |
| `fetch_overdue` | `OverdueFetcher` | `(days: int) -> list[dict]` |
| `send_email` | `EmailSender` | `(to_email, subject, body) -> bool` |
| `start_run` | `RunStarter` | `(agent_name, input_data) -> int` |
| `finish_run` | `RunFinisher` | `(run_id, status, summary, output_data) -> None` |
| `mission` | `AgentMission` | Any object with the six mission fields |

## Testing

```bash
uv run pytest -v
```

7 tests covering: opt-out handling, interested reply flow, rejected reply logging, overdue follow-up drafting, unmatched senders, empty inbox, and SMTP failure resilience.

## Support

If you find this useful, a small donation helps keep projects like this going:
[Donate via PayPal](https://paypal.me/christopherrehm001)
