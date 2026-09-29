"""
Thin wrapper around the Hindsight client. This module is the "brain" of the
Deal Intelligence Agent -- everything a rep has ever said about a deal, and
every cross-deal lesson about what tactics work, lives in Hindsight memory
banks, not in our own database.

Two kinds of banks:
  - deal:<deal_id>   -> per-deal memory. Raw call notes, objections,
                        stakeholder concerns, pricing talk for ONE deal.
  - playbook         -> org-wide memory. Distilled "what worked" tactics
                        pulled from every deal, so the agent gets smarter
                        across the whole team's deal history, not just one
                        account.

Structured memory: wherever possible, what we retain is written as a labelled
block -- Customer, Stakeholder, Objection, Competitor, Tactic, Outcome,
Reason, Date, Deal stage -- followed by a plain-English sentence, and is also
tagged (tags=[...]) and given metadata. The banks themselves are unchanged.
"""
import json
import os
import re
from hindsight_client import Hindsight

_client = None


def get_client() -> Hindsight:
    global _client
    if _client is None:
        base_url = os.environ.get("HINDSIGHT_BASE_URL", "https://api.hindsight.vectorize.io")
        api_key = os.environ.get("HINDSIGHT_API_KEY")
        _client = Hindsight(base_url=base_url, api_key=api_key)
    return _client


def deal_bank(deal_id: str) -> str:
    return f"deal:{deal_id}"


PLAYBOOK_BANK = "playbook"


# ---------- Structured-memory helpers ----------

STRUCTURED_FIELDS = [
    ("customer", "Customer"),
    ("stakeholder", "Stakeholder"),
    ("objection", "Objection"),
    ("competitor", "Competitor"),
    ("tactic", "Tactic"),
    ("outcome", "Outcome"),
    ("reason", "Reason"),
    ("date", "Date"),
    ("deal_stage", "Deal stage"),
]


def slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", (value or "").lower()).strip("-")


def structured_header(fields: dict) -> str:
    """Render the labelled block. Empty fields are omitted -- we never write
    'Competitor: unknown', because that would teach the memory a non-fact."""
    lines = [f"{label}: {fields[key]}" for key, label in STRUCTURED_FIELDS
             if fields.get(key)]
    return "\n".join(lines)


def _clean_meta(meta: dict) -> dict:
    return {k: str(v) for k, v in (meta or {}).items() if v not in (None, "")}


def _retain(bank_id: str, content: str, context: str = None, timestamp=None,
            tags: list = None, metadata: dict = None, document_id: str = None):
    """retain() with graceful degradation: older hindsight-client versions do
    not accept tags/metadata/document_id, so retry with the core arguments."""
    client = get_client()
    kwargs = {"bank_id": bank_id, "content": content}
    if context:
        kwargs["context"] = context
    if timestamp:
        kwargs["timestamp"] = timestamp
    extras = {}
    if tags:
        extras["tags"] = tags
    if metadata:
        extras["metadata"] = _clean_meta(metadata)
    if document_id:
        extras["document_id"] = document_id
    try:
        return client.retain(**kwargs, **extras)
    except TypeError:
        return client.retain(**kwargs)


def retain_call_note(deal_id: str, content: str, context: str, timestamp: str = None,
                     structured: dict = None):
    """Push a raw call note / email / meeting summary into this deal's memory.

    If `structured` is given (customer, stakeholder, deal_stage, objection...),
    a labelled header is prepended so the extracted memories carry those
    entities explicitly. The raw note is always included verbatim."""
    body = content
    tags = None
    if structured:
        body = structured_header(structured) + "\n\n" + content
        tags = ["record:call-note"]
        if structured.get("objection"):
            tags.append(f"objection:{slug(structured['objection'])}")
    return _retain(deal_bank(deal_id), body, context, timestamp, tags=tags,
                   metadata=structured)


def retain_playbook_lesson(content: str, context: str):
    """Push a distilled, cross-deal lesson (e.g. 'annual-plan discount overcame
    a pricing objection at a mid-market SaaS company') into the shared playbook."""
    return _retain(PLAYBOOK_BANK, content, context)


# ---------- Outcome records (the learning loop's write path) ----------

def outcome_record_text(exp: dict) -> str:
    """One outcome, as structured fields + a natural-language sentence.
    Hindsight's extraction reads prose best; the labelled block keeps every
    entity explicit and greppable."""
    objection = exp["objection_type"]
    if exp.get("objection_text"):
        objection = f'{exp["objection_type"]} -- "{exp["objection_text"]}"'
    header = structured_header({
        "customer": exp.get("customer"),
        "stakeholder": exp.get("stakeholder"),
        "objection": objection,
        "competitor": exp.get("competitor"),
        "tactic": exp.get("tactic_used"),
        "outcome": exp.get("outcome"),
        "reason": exp.get("reason"),
        "date": (exp.get("date") or "")[:10],
        "deal_stage": exp.get("deal_stage"),
    })
    extra = []
    if exp.get("recommended_tactic"):
        extra.append(f"Tactic recommended by the agent: {exp['recommended_tactic']}")
    if exp.get("actual_action"):
        extra.append(f"Actual action taken: {exp['actual_action']}")
    if exp.get("result"):
        extra.append(f"Result: {exp['result']}")

    who = exp.get("stakeholder") or "the buyer"
    sentence = (
        f"At {exp.get('customer')}"
        + (f" ({exp['deal_stage']} stage)" if exp.get("deal_stage") else "")
        + f", {who} raised a {exp['objection_type']} objection"
        + (f" involving competitor {exp['competitor']}" if exp.get("competitor") else "")
        + f". The rep used the tactic '{exp.get('tactic_used')}'. "
        + f"The outcome was {exp['outcome']}"
        + (f" because {exp['reason']}" if exp.get("reason") else "")
        + "."
    )
    parts = ["[STRUCTURED OUTCOME RECORD]", header]
    if extra:
        parts.append("\n".join(extra))
    parts.append("Summary: " + sentence)
    return "\n".join(parts)


def retain_outcome(exp: dict, to_deal_bank: bool = True) -> dict:
    """Retain one recorded outcome into Hindsight: the deal's own bank (so the
    deal brief knows what happened) and the shared playbook bank (so every
    other deal can learn from it). document_id makes re-runs idempotent."""
    content = outcome_record_text(exp)
    tags = ["record:outcome", f"objection:{exp['objection_type']}",
            f"outcome:{exp['outcome']}", f"tactic:{slug(exp.get('tactic_used'))}",
            f"customer:{slug(exp.get('customer'))}"]
    if exp.get("competitor"):
        tags.append(f"competitor:{slug(exp['competitor'])}")
    meta = {k: exp.get(k) for k in (
        "customer", "stakeholder", "objection_type", "competitor", "tactic_used",
        "outcome", "reason", "date", "deal_stage")}
    context = f"outcome-record objection:{exp['objection_type']} outcome:{exp['outcome']}"
    ts = exp.get("timestamp")
    doc_id = f"outcome-{exp['id']}"

    banks = []
    if to_deal_bank and exp.get("deal_id"):
        _retain(deal_bank(exp["deal_id"]), content, context, ts, tags, meta, doc_id)
        banks.append(deal_bank(exp["deal_id"]))
    _retain(PLAYBOOK_BANK, content, context, ts, tags, meta, doc_id)
    banks.append(PLAYBOOK_BANK)
    return {"retained_to": banks}


# ---------- Recall / reflect helpers for the recommendation pipeline ----------

def _to_items(result, limit: int) -> list:
    items = []
    for r in (getattr(result, "results", None) or [])[:limit]:
        text = getattr(r, "text", None) or str(r)
        items.append({
            "text": text if len(text) <= 420 else text[:417] + "...",
            "type": getattr(r, "type", None),
            "date": getattr(r, "occurred_start", None) or getattr(r, "mentioned_at", None),
        })
    return items


def recall_deal_memories(deal_id: str, query: str, limit: int = 6) -> list:
    client = get_client()
    return _to_items(client.recall(bank_id=deal_bank(deal_id), query=query), limit)


def recall_playbook_memories(objection_type: str, query: str, limit: int = 6,
                             competitor: str = None) -> list:
    """Prefer outcome records tagged for this exact situation, narrowing as far
    as the tags allow and widening again when nothing comes back:

      1. objection + competitor tags (only when a competitor is actually named)
      2. objection tag alone
      3. plain semantic recall over the whole playbook

    Step 3 is what keeps older, untagged memories usable."""
    client = get_client()

    def _tagged(tags):
        try:
            return _to_items(client.recall(bank_id=PLAYBOOK_BANK, query=query,
                                           tags=tags, tags_match="any_strict"), limit)
        except TypeError:
            return []

    items = []
    if competitor:
        items = _tagged([f"objection:{objection_type}", f"competitor:{slug(competitor)}"])
    if not items:
        items = _tagged([f"objection:{objection_type}"])
    if not items:
        items = _to_items(client.recall(bank_id=PLAYBOOK_BANK, query=query), limit)
    return items


def _parse_json_object(text: str):
    if not text:
        return None
    cleaned = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    try:
        obj = json.loads(cleaned)
        return obj if isinstance(obj, dict) else None
    except ValueError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if match:
            try:
                obj = json.loads(match.group(0))
                return obj if isinstance(obj, dict) else None
            except ValueError:
                return None
    return None


def reflect_json(bank_id: str, query: str, schema: dict) -> dict:
    """reflect() asking for a JSON object. Uses response_schema when the client
    supports it, otherwise parses JSON out of the text. Raises if neither works."""
    client = get_client()
    try:
        result = client.reflect(bank_id=bank_id, query=query, response_schema=schema)
    except TypeError:
        result = client.reflect(bank_id=bank_id, query=query)
    structured = getattr(result, "structured_output", None)
    if isinstance(structured, dict) and structured:
        return structured
    parsed = _parse_json_object(getattr(result, "text", "") or "")
    if parsed:
        return parsed
    raise ValueError("reflect() did not return usable JSON")


def generate_call_brief(deal_id: str) -> str:
    """The core 'wow' moment: ask Hindsight to reflect across every interaction
    logged for this deal and produce a rep-ready call brief."""
    client = get_client()
    query = (
        "Prepare a concise pre-call brief for a sales rep about to get on a call "
        "with this prospect. Cover: (1) every objection raised so far and whether "
        "it was resolved, (2) competitors that have been mentioned, (3) stakeholder "
        "concerns and who holds influence, (4) pricing/budget discussion history, "
        "and (5) 2-3 recommended talking points for this next call based on that "
        "history. Be specific and reference what was actually said, not generic advice."
    )
    result = client.reflect(bank_id=deal_bank(deal_id), query=query)
    return getattr(result, "text", str(result))


def ask_about_deal(deal_id: str, question: str) -> str:
    client = get_client()
    result = client.reflect(bank_id=deal_bank(deal_id), query=question)
    return getattr(result, "text", str(result))


def recall_deal_raw(deal_id: str, query: str):
    client = get_client()
    result = client.recall(bank_id=deal_bank(deal_id), query=query)
    return getattr(result, "results", [])


def playbook_insight(objection_type: str) -> str:
    """Cross-deal learning: what has historically worked against this kind of
    objection, across every deal the team has ever logged?"""
    client = get_client()
    query = (
        f"Across all deals in memory, what tactics have worked best for handling "
        f"a '{objection_type}' objection? Summarize the pattern and give a concrete "
        f"recommended approach a rep could use today. If you don't have enough "
        f"history yet, say so plainly instead of guessing."
    )
    result = client.reflect(bank_id=PLAYBOOK_BANK, query=query)
    return getattr(result, "text", str(result))


def no_memory_baseline(question: str) -> str:
    """Simulates what a stateless assistant would say with zero memory, purely
    for the demo's before/after contrast. No Hindsight call at all."""
    return (
        "I don't have any prior context about this prospect or deal. "
        "I'd need you to re-explain the history, past objections, and "
        "stakeholders before I can give useful advice."
    )
