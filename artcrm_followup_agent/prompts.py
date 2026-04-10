from .protocols import AgentMission

# Classifications the LLM can assign to an incoming reply.
REPLY_CLASSIFICATIONS = ("interested", "warm", "not_interested", "not_possible", "opt_out", "other")

OPT_OUT_LINE = {
    "de": "Wenn Sie keine weiteren Nachrichten wünschen, antworten Sie bitte mit 'Abmelden'.",
    "en": "If you'd prefer not to receive further messages, please reply with 'Unsubscribe'.",
    "fr": "Si vous ne souhaitez plus recevoir de messages, répondez 'Désabonner'.",
    "cs": "Pokud si nepřejete dostávat další zprávy, odpovězte prosím 'Odhlásit'.",
    "nl": "Als u geen verdere berichten wenst te ontvangen, antwoord dan met 'Afmelden'.",
    "es": "Si prefiere no recibir más mensajes, responda con 'Cancelar suscripción'.",
    "it": "Se non desidera ricevere ulteriori messaggi, risponda con 'Annulla iscrizione'.",
}


def classify_reply_prompt(mission: AgentMission, message: dict) -> tuple[str, str]:
    system = (
        f"You are processing email replies for {mission.identity}.\n"
        f"You help manage outreach for: {mission.goal}"
    )
    user = (
        f"Classify this incoming email reply.\n\n"
        f"From: {message.get('from_email')}\n"
        f"Subject: {message.get('subject')}\n"
        f"Body:\n{message.get('body', '')}\n\n"
        f"Classify as exactly one of:\n"
        f'- "interested" — clear yes: they want to meet, show work, discuss next steps, or explicitly express strong interest\n'
        f'- "warm" — friendly and positive but no concrete commitment: they like the idea, say maybe, ask for more info, or suggest a future possibility without committing\n'
        f'- "not_interested" — polite or direct no: they decline, say it\'s not for them, or are not interested\n'
        f'- "not_possible" — logistical barrier: venue is closed, wrong person, already have artists, under renovation, or circumstances make it impossible right now\n'
        f'- "opt_out" — they explicitly ask to stop receiving messages or to be removed\n'
        f'- "other" — unclear, out of office, automated, or requires human judgement\n\n'
        f"Return JSON: {{\"classification\": \"...\", \"reasoning\": \"one sentence\"}}\n"
        f"Return ONLY the JSON object, no other text."
    )
    return system, user


def draft_interested_reply_prompt(mission: AgentMission, contact: dict, message: dict, language: str) -> tuple[str, str]:
    opt_out = OPT_OUT_LINE.get(language, OPT_OUT_LINE["en"])
    system = (
        f"You are {mission.identity}.\n"
        f"Outreach style: {mission.outreach_style}"
    )
    user = (
        f"Write a warm, enthusiastic reply to this clearly interested message from {contact.get('name')} "
        f"({contact.get('city')}).\n"
        f"Write entirely in language code: {language}\n\n"
        f"Their message:\nSubject: {message.get('subject')}\n{message.get('body', '')}\n\n"
        f"The reply should:\n"
        f"- Respond warmly and match their energy\n"
        f"- Propose a concrete next step (a visit, video call, or sending portfolio)\n"
        f"- Be specific — mention their venue type or city\n"
        f"- Sign off with your name and website: {mission.website}\n"
        f'- End with this opt-out line (verbatim): "{opt_out}"\n\n'
        f"Return JSON: {{\"subject\": \"...\", \"body\": \"...\"}}\n"
        f"Return ONLY the JSON object, no other text."
    )
    return system, user


def draft_warm_reply_prompt(mission: AgentMission, contact: dict, message: dict, language: str) -> tuple[str, str]:
    opt_out = OPT_OUT_LINE.get(language, OPT_OUT_LINE["en"])
    system = (
        f"You are {mission.identity}.\n"
        f"Outreach style: {mission.outreach_style}"
    )
    user = (
        f"Write a gentle, low-pressure reply to this friendly-but-uncommitted message from {contact.get('name')} "
        f"({contact.get('city')}).\n"
        f"Write entirely in language code: {language}\n\n"
        f"Their message:\nSubject: {message.get('subject')}\n{message.get('body', '')}\n\n"
        f"The reply should:\n"
        f"- Be warm and appreciative — don't push for commitment\n"
        f"- Keep the door open naturally (e.g. mention you'll be in the area, or invite them to reach out whenever)\n"
        f"- Be brief — 2-3 sentences max\n"
        f"- Sign off with your name and website: {mission.website}\n"
        f'- End with this opt-out line (verbatim): "{opt_out}"\n\n'
        f"Return JSON: {{\"subject\": \"...\", \"body\": \"...\"}}\n"
        f"Return ONLY the JSON object, no other text."
    )
    return system, user


def draft_followup_prompt(
    mission: AgentMission,
    contact: dict,
    days_since: int,
    language: str,
    original_subject: str = "",
) -> tuple[str, str]:
    opt_out = OPT_OUT_LINE.get(language, OPT_OUT_LINE["en"])
    system = (
        f"You are {mission.identity}.\n"
        f"Outreach style: {mission.outreach_style}"
    )
    original_ref = f"\nOriginal outreach: {original_subject}" if original_subject else ""
    user = (
        f"Write a brief, non-pushy follow-up email to {contact.get('name')} "
        f"({contact.get('type', 'venue')} in {contact.get('city')}).\n"
        f"You haven't heard back in {days_since} days.{original_ref}\n"
        f"Write entirely in language code: {language}\n\n"
        f"The email should:\n"
        f"- Be 2-3 sentences only — brief and light\n"
        f"- Not feel like pressure or a reminder\n"
        f"- Offer a natural out if they're not interested\n"
        f"- Sign off with your name and website: {mission.website}\n"
        f'- End with this opt-out line (verbatim): "{opt_out}"\n\n'
        f"Return JSON: {{\"subject\": \"...\", \"body\": \"...\"}}\n"
        f"Return ONLY the JSON object, no other text."
    )
    return system, user
