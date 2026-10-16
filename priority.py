import re


HIGH_PRIORITY_CUES = (
    "urgent",
    "critical",
    "outage",
    "security breach",
    "data loss",
    "system down",
    "system is down",
    "production down",
    "production is down",
    "service down",
    "service is down",
)

MEDIUM_PRIORITY_CUES = (
    "error",
    "failed",
    "failure",
    "unable",
    "blocked",
    "not working",
    "cannot access",
    "can't access",
    "issue",
    "unavailable",
    "not responding",
)

NEGATIONS = {
    "no",
    "not",
    "never",
    "without",
    "isn't",
    "aren't",
    "wasn't",
    "weren't",
}


def _contains_active_cue(tokens: list[str], cue: str) -> bool:
    cue_tokens = cue.split()
    last_start = len(tokens) - len(cue_tokens)

    for start in range(last_start + 1):
        if tokens[start : start + len(cue_tokens)] != cue_tokens:
            continue

        context_start = max(0, start - 3)
        if not NEGATIONS.intersection(tokens[context_start:start]):
            return True

    return False


def classify_priority(ticket: str) -> tuple[str, str]:
    """Estimate priority from explicit urgency and impact phrases."""
    normalized_text = ticket.casefold().replace("’", "'")
    tokens = re.findall(r"[a-z0-9]+(?:'[a-z]+)?", normalized_text)

    for cue in HIGH_PRIORITY_CUES:
        if _contains_active_cue(tokens, cue):
            return "High", f"High-priority signal detected: '{cue}'."

    for cue in MEDIUM_PRIORITY_CUES:
        if _contains_active_cue(tokens, cue):
            return "Medium", f"Medium-priority signal detected: '{cue}'."

    return "Low", "No high- or medium-priority signal was detected."
