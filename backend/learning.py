"""
Outcome-aware learning loop.

    MEMORY -> EVIDENCE -> CONTEXT MATCH -> RECOMMENDATION -> OUTCOME
           -> LEARNING -> BETTER FUTURE RECOMMENDATION

This is outcome-aware memory + reasoning, not reinforcement learning. Nothing
is trained. The agent gets better because every recorded outcome is (a) retained
into Hindsight as a structured memory and (b) mirrored into the experience
ledger in store.py, and the next recommendation is computed from those records.

Division of labour -- this is what keeps the agent from inventing evidence:

  * CODE (this file) decides everything that is a fact or a number: which prior
    experiences are relevant, which of them actually share context with the
    deal in front of the rep, how many succeeded / failed, which tactic is
    recommended, how confident we are, and what the cautions are.
  * HINDSIGHT recalls the relevant memories (shown in the MEMORY step) and its
    reflect() writes the human-readable reasoning and next action, but it is
    handed the verified evidence and told not to add any of its own.
  * If reflect() is unavailable, a deterministic explanation built from the
    same evidence is used, and the response says so.

Three properties this module is careful about:

  1. Context similarity is never invented. A past experience only counts as
     "like this deal" when a structured field it actually stores (customer,
     competitor, stakeholder, deal_stage, objection_text) matches a structured
     field of the current situation. Every match is returned with the field and
     both values so the UI can show why it matched.
  2. Conflicting evidence is surfaced, not averaged away. A tactic with both
     successes and failures is reported as "Mixed evidence" together with the
     conditions recorded on each side, and loses ties against a tactic with a
     clean record.
  3. The learning curve is read back out of the stored recommendation log, so
     the interaction numbers shown in the UI are counts, not storytelling.
"""
import re
from typing import Optional

import store
import hindsight_service as hs

OBJECTION_TYPES = ("pricing", "competitor", "trust", "timing", "authority")
OUTCOMES = ("positive", "negative", "neutral")

# The pre-existing UI/API tagged notes as advanced / stalled / lost.
LEGACY_OUTCOME_MAP = {"advanced": "positive", "stalled": "neutral", "lost": "negative"}

UNSPECIFIED = "(tactic not recorded)"

# What a stateless assistant would say. Used ONLY when the team has no closed
# outcomes for this objection type, and always labelled as generic.
GENERIC_PLAYBOOK = {
    "pricing": {
        "tactic": "Reframe price as return on investment",
        "reasoning": "Standard sales guidance: move the conversation from price to value, "
                     "and quantify the return before discussing discounts.",
        "next_action": "Ask what the current problem costs them per month, then walk through "
                       "the return before offering any concession.",
    },
    "competitor": {
        "tactic": "Differentiate on what the competitor cannot do",
        "reasoning": "Standard sales guidance: avoid feature-by-feature or price-matching "
                     "battles; anchor on the two or three needs the competitor serves poorly.",
        "next_action": "Ask which requirements matter most, then show how you meet those specifically.",
    },
    "trust": {
        "tactic": "Provide proof of credibility",
        "reasoning": "Standard sales guidance: address trust with third-party proof "
                     "(certifications, references, case studies) rather than assurances.",
        "next_action": "Offer relevant certifications or a customer reference and ask "
                       "what evidence their reviewers need.",
    },
    "timing": {
        "tactic": "Acknowledge the timing and agree a follow-up date",
        "reasoning": "Standard sales guidance: respect the constraint, keep the relationship warm, "
                     "and agree when you will next speak.",
        "next_action": "Ask when the budget or priority window opens and schedule a check-in around it.",
    },
    "authority": {
        "tactic": "Identify the decision-maker and how decisions get made",
        "reasoning": "Standard sales guidance: map the buying process early so you are not "
                     "relying on a contact who cannot sign.",
        "next_action": "Ask who else needs to approve this and what their criteria will be.",
    },
}


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def normalize_outcome(value: str) -> str:
    v = (value or "").strip().lower()
    if v in OUTCOMES:
        return v
    if v in LEGACY_OUTCOME_MAP:
        return LEGACY_OUTCOME_MAP[v]
    raise ValueError(f"outcome must be one of {OUTCOMES} (or legacy advanced/stalled/lost)")


def _key(label: str) -> str:
    return " ".join((label or "").lower().split())


def _short(exp: dict) -> dict:
    """The compact, display-safe view of a ledger row used as evidence."""
    return {k: exp.get(k) for k in (
        "id", "customer", "stakeholder", "competitor", "objection_type", "objection_text",
        "tactic_used", "outcome", "reason", "result", "date", "deal_stage", "deal_id")}


# ---------------------------------------------------------------------------
# CONTEXT MATCHING
#
# "Is this past experience like the situation the rep is in right now?"
# Answered only from structured fields that the experience ledger actually
# stores. Nothing here guesses, embeds, or asks an LLM.
# ---------------------------------------------------------------------------

# Role words we are willing to treat as the same kind of stakeholder. This is a
# fixed, explicit vocabulary on purpose: a stakeholder match is always traceable
# to a word that BOTH sides literally wrote in their stakeholder field, so we
# never claim "Head of Procurement is like VP Finance" on a hunch.
ROLE_WORDS = {
    "procurement": "procurement", "purchasing": "procurement", "sourcing": "procurement",
    "cfo": "finance", "finance": "finance", "financial": "finance", "controller": "finance",
    "security": "security", "ciso": "security",
    "compliance": "compliance", "regulatory": "compliance", "legal": "legal",
    "it": "it", "cio": "it", "infrastructure": "it",
    "cto": "engineering", "engineering": "engineering", "engineer": "engineering",
    "operations": "operations", "ops": "operations", "coo": "operations",
    "marketing": "marketing", "sales": "sales", "revenue": "sales",
    "clinical": "clinical", "patient": "clinical", "medical": "clinical",
    "hr": "people", "people": "people",
}

# Weight per matched dimension. Integers, so the score is auditable by eye.
CONTEXT_WEIGHTS = {
    "customer": 3,           # the same account -- the strongest possible signal
    "competitor": 3,         # same competitor in the room
    "stakeholder": 3,        # literally the same person
    "stakeholder_role": 2,   # a different person in a comparable role
    "deal_stage": 2,         # same point in the cycle
    "concern": 1,            # the buyer said overlapping things
}

# A row counts as "highly relevant contextual evidence" at or above this score.
# 3 means one strong dimension (same account / competitor / stakeholder), or two
# softer ones (role + stage). A stage match on its own is deliberately NOT enough.
CONTEXT_MATCH_MIN = 3

# ...and contextual evidence is only allowed to *drive* the recommendation once
# there are at least this many matching rows. Below that we say so and fall back
# to the general record rather than pretending one anecdote is a pattern.
CONTEXT_MIN_ROWS = 2

_STOPWORDS = {
    "about", "after", "again", "against", "already", "always", "another", "anything",
    "because", "before", "being", "between", "could", "different", "during", "either",
    "enough", "every", "going", "instead", "might", "other", "really", "should",
    "since", "something", "their", "there", "these", "thing", "think", "those",
    "through", "until", "using", "value", "where", "which", "while", "would",
}


def _concern_terms(text: str) -> set:
    """Significant words a buyer actually used. Long words only, minus filler,
    so 'we already pay for a bundle' overlaps on 'bundle' rather than 'already'."""
    return {w for w in re.findall(r"[a-z]{5,}", (text or "").lower())} - _STOPWORDS


def _roles(text: str) -> set:
    words = re.findall(r"[a-z]+", (text or "").lower())
    return {ROLE_WORDS[w] for w in words if w in ROLE_WORDS}


def context_of(deal: Optional[dict], stakeholder: str = None, competitor: str = None,
               deal_stage: str = None, objection_text: str = None) -> dict:
    """The current situation, as the same structured fields the ledger stores.
    Only fields we actually know are filled in -- an unknown competitor stays
    None and therefore can never produce a match.

    `stakeholder_roles` is the one derived field. Reps type a person's *name*
    ("Elena Ruiz") while historical rows often store a *role* ("Director of
    Procurement"), so the role has to come from somewhere. We take it from the
    deal's own contact_title -- but only when the named stakeholder IS the
    deal's contact, or when no stakeholder was named at all. If the rep names
    somebody else, we deliberately learn nothing about their role rather than
    borrow the contact's title and invent one."""
    deal = deal or {}
    sh = (stakeholder or "").strip() or None
    contact = (deal.get("contact_name") or "").strip()
    role_text = sh or ""
    if not sh or (contact and _key(sh) == _key(contact)):
        role_text = f"{role_text} {deal.get('contact_title') or ''}"
    return {
        "customer": deal.get("company") or None,
        "competitor": (competitor or "").strip() or None,
        "stakeholder": sh,
        "stakeholder_title": (deal.get("contact_title") or "").strip() or None if (
            not sh or (contact and _key(sh) == _key(contact))) else None,
        "stakeholder_roles": sorted(_roles(role_text)),
        "deal_stage": (deal_stage or deal.get("stage") or "").strip() or None,
        "concern": (objection_text or "").strip() or None,
    }


# Context keys that are plain scalars describing the situation -- the ones safe
# to echo back to a human as "customer: X, competitor: Y".
CONTEXT_QUERY_FIELDS = ("customer", "competitor", "stakeholder", "deal_stage", "concern")


def match_context(exp: dict, ctx: dict) -> tuple:
    """Score one past experience against the current context.

    Returns (score, matches). Every entry in `matches` names the field, the
    current value and the past value, so the UI can justify the claim instead
    of asserting 'similar deal'."""
    matches = []

    def add(field, label, current, past, weight=None):
        matches.append({
            "field": field, "label": label, "current": current, "past": past,
            "weight": weight if weight is not None else CONTEXT_WEIGHTS[field],
        })

    if ctx.get("customer") and exp.get("customer") and _key(ctx["customer"]) == _key(exp["customer"]):
        add("customer", "Same account", ctx["customer"], exp["customer"])

    if ctx.get("competitor") and exp.get("competitor") and _key(ctx["competitor"]) == _key(exp["competitor"]):
        add("competitor", "Same competitor", ctx["competitor"], exp["competitor"])

    cur_sh, past_sh = ctx.get("stakeholder"), exp.get("stakeholder")
    if cur_sh and past_sh and _key(cur_sh) == _key(past_sh):
        add("stakeholder", "Same stakeholder", cur_sh, past_sh)
    elif past_sh:
        shared_roles = set(ctx.get("stakeholder_roles") or []) & _roles(past_sh)
        if shared_roles:
            add("stakeholder_role", f"Comparable role ({'/'.join(sorted(shared_roles))})",
                cur_sh or ctx.get("stakeholder_title") or "", past_sh)

    if (ctx.get("deal_stage") and exp.get("deal_stage")
            and _key(ctx["deal_stage"]) == _key(exp["deal_stage"])):
        add("deal_stage", "Same deal stage", ctx["deal_stage"], exp["deal_stage"])

    shared_terms = _concern_terms(ctx.get("concern")) & _concern_terms(exp.get("objection_text"))
    if len(shared_terms) >= 2:
        # The match IS the shared wording, so that is what we display -- echoing
        # the whole past sentence back would overstate how much lines up.
        terms = ", ".join(sorted(shared_terms)[:4])
        add("concern", "Overlapping concern", terms, terms)

    return sum(m["weight"] for m in matches), matches


def _context_phrase(matches: list) -> str:
    """'same competitor Ionic Sales Cloud, comparable role (procurement)' --
    built only from dimensions that really matched."""
    seen, parts = set(), []
    for m in matches:
        if m["field"] in seen:
            continue
        seen.add(m["field"])
        if m["field"] == "customer":
            parts.append(f"the same account ({m['past']})")
        elif m["field"] == "competitor":
            parts.append(f"the same competitor ({m['past']})")
        elif m["field"] == "stakeholder":
            parts.append(f"the same stakeholder ({m['past']})")
        elif m["field"] == "stakeholder_role":
            parts.append(f"a comparable stakeholder role ({m['past']})")
        elif m["field"] == "deal_stage":
            parts.append(f"the same deal stage ({m['past']})")
        elif m["field"] == "concern":
            parts.append(f"an overlapping stated concern ({m['past']})")
    return ", ".join(parts)


# ---------------------------------------------------------------------------
# EVIDENCE: pure functions over ledger rows
# ---------------------------------------------------------------------------

def _condition(e: dict) -> dict:
    """The circumstances attached to one outcome -- used to explain when a
    tactic worked and when the same tactic did not."""
    return {k: e.get(k) for k in
            ("customer", "stakeholder", "competitor", "deal_stage", "reason", "date")}


def condition_text(c: dict) -> str:
    """'Coldstream Robotics (Head of Procurement, Negotiation stage, vs Ionic
    Sales Cloud) -- she was evaluating total cost'. Empty fields are skipped, so
    we never render 'Competitor: unknown'."""
    where = [v for v in (c.get("stakeholder"), c.get("deal_stage") and f"{c['deal_stage']} stage") if v]
    if c.get("competitor"):
        where.append(f"vs {c['competitor']}")
    head = c.get("customer") or "an unnamed account"
    if where:
        head += " (" + ", ".join(where) + ")"
    return head + (f" — {c['reason']}" if c.get("reason") else "")


def tactic_stats(relevant: list) -> list:
    """Per-tactic tally. score is a Laplace-smoothed success rate,
    (successes + 1) / (attempts + 2): one lucky win does not outrank a tactic
    with a longer, mostly-good record. Neutral counts as not-a-success.

    Each row also carries its contradiction picture: a `status` and the
    conditions recorded on the successful and unsuccessful attempts, so the
    caller can explain a conflict instead of hiding it behind an average."""
    stats = {}
    for e in relevant:
        label = (e.get("tactic_used") or "").strip() or UNSPECIFIED
        s = stats.setdefault(_key(label), {
            "tactic": label, "successful": 0, "unsuccessful": 0, "neutral": 0,
            "n": 0, "customers": [], "last_date": "",
            "success_conditions": [], "failure_conditions": []})
        outcome = e.get("outcome")
        if outcome == "positive":
            s["successful"] += 1
            s["success_conditions"].append(_condition(e))
        elif outcome == "negative":
            s["unsuccessful"] += 1
            s["failure_conditions"].append(_condition(e))
        else:
            s["neutral"] += 1
        s["n"] += 1
        if e.get("customer") and e["customer"] not in s["customers"]:
            s["customers"].append(e["customer"])
        s["last_date"] = max(s["last_date"], e.get("date") or "")
    out = list(stats.values())
    for s in out:
        s["score"] = round((s["successful"] + 1) / (s["n"] + 2), 3)
        if s["successful"] and s["unsuccessful"]:
            s["status"] = "mixed"
        elif s["successful"]:
            s["status"] = "consistent"
        elif s["unsuccessful"]:
            s["status"] = "failing"
        else:
            s["status"] = "inconclusive"
    # Contradiction-aware ordering: at a comparable score, a tactic with a clean
    # record beats one that has already failed somewhere. `reverse=True` means
    # the 1 (not mixed) sorts ahead of the 0 (mixed).
    out.sort(key=lambda s: (s["score"], 0 if s["status"] == "mixed" else 1,
                            s["n"], s["last_date"]), reverse=True)
    return out


def _stat_for(stats: list, tactic: str) -> Optional[dict]:
    for s in stats:
        if _key(s["tactic"]) == _key(tactic):
            return s
    return None


def _confidence(top: Optional[dict], overall: Optional[dict] = None) -> str:
    """none | weak | mixed | moderate | strong -- judged on the recommended
    tactic's own record, because that is the claim we are making.

    `overall` is the same tactic's record across ALL relevant experiences. When
    the contextual slice looks clean but the full record is contradictory, we
    report mixed: a clean sub-sample is not a licence to hide a known failure."""
    if top is None:
        return "none"
    if overall and overall["successful"] > 0 and overall["unsuccessful"] > 0:
        return "mixed"
    if top["successful"] > 0 and top["unsuccessful"] > 0:
        return "mixed"
    if top["n"] == 1:
        return "weak"
    if top["n"] >= 3 and top["unsuccessful"] == 0 and top["neutral"] == 0:
        return "strong"
    return "moderate"


def conflict_view(stat: Optional[dict]) -> Optional[dict]:
    """The explicit 'Mixed evidence' block: what the record is, the conditions
    on each side, and the one thing to check before using the tactic anyway."""
    if not stat or not (stat["successful"] > 0 and stat["unsuccessful"] > 0):
        return None
    failure_reasons = [c["reason"] for c in stat["failure_conditions"] if c.get("reason")]
    if failure_reasons:
        check = ("Before using it here, confirm none of these applies: "
                 + "; ".join(dict.fromkeys(failure_reasons)) + ".")
    else:
        who = ", ".join(c.get("customer") or "an unnamed account"
                        for c in stat["failure_conditions"])
        check = (f"Before using it here, check what was different about the "
                 f"{stat['unsuccessful']} recorded failure(s) at {who} — no reason was logged for them.")
    return {
        "label": "Mixed evidence",
        "tactic": stat["tactic"],
        "successful": stat["successful"],
        "unsuccessful": stat["unsuccessful"],
        "neutral": stat["neutral"],
        "n": stat["n"],
        "worked_when": [condition_text(c) for c in stat["success_conditions"]],
        "failed_when": [condition_text(c) for c in stat["failure_conditions"]],
        "check_before_using": check,
    }


def avoid_list(stats: list) -> list:
    """Tactics with failures and no successes on record. Two or more failures
    reads as 'avoid'; a single one is only 'use caution' -- one bad day is not
    a verdict."""
    rows = []
    for s in stats:
        if s["tactic"] == UNSPECIFIED or s["successful"] > 0 or s["unsuccessful"] == 0:
            continue
        rows.append({
            "tactic": s["tactic"],
            "unsuccessful": s["unsuccessful"],
            "neutral": s["neutral"],
            "n": s["n"],
            "severity": "avoid" if s["unsuccessful"] >= 2 else "use caution",
            "customers": s["customers"],
            "failed_when": [condition_text(c) for c in s["failure_conditions"]],
        })
    rows.sort(key=lambda r: (r["unsuccessful"], r["n"]), reverse=True)
    return rows


def build_recommendation(deal: dict, objection_type: str, experiences: list,
                         competitor: Optional[str] = None, stakeholder: Optional[str] = None,
                         deal_stage: Optional[str] = None,
                         objection_text: Optional[str] = None) -> dict:
    """Deterministic core of the recommendation. Returns the evidence, the
    context match, the chosen tactic, confidence, conflicts, cautions and a
    fallback explanation. No LLM involved."""
    relevant = [e for e in experiences if e.get("objection_type") == objection_type]
    generic = GENERIC_PLAYBOOK.get(objection_type, GENERIC_PLAYBOOK["pricing"])

    # ---- CONTEXT MATCH: score every relevant experience against this deal ----
    ctx = context_of(deal, stakeholder, competitor, deal_stage, objection_text)
    scored = []
    for e in relevant:
        score, matches = match_context(e, ctx)
        scored.append({"exp": e, "score": score, "matches": matches})
    scored.sort(key=lambda t: (t["score"], t["exp"].get("date") or ""), reverse=True)

    contextual = [t for t in scored if t["score"] >= CONTEXT_MATCH_MIN]
    contextual_rows = [t["exp"] for t in contextual]

    general_stats = tactic_stats(relevant)
    context_stats = tactic_stats(contextual_rows)

    general_candidates = [s for s in general_stats
                          if s["successful"] > 0 and s["tactic"] != UNSPECIFIED]
    context_candidates = [s for s in context_stats
                          if s["successful"] > 0 and s["tactic"] != UNSPECIFIED]

    use_context = len(contextual_rows) >= CONTEXT_MIN_ROWS and bool(context_candidates)

    # Why contextual evidence was NOT used -- stated plainly, so the UI never
    # has to imply a similarity that the data does not support.
    if use_context:
        context_note = None
    elif not contextual_rows:
        context_note = ("No recorded experience shares enough structured context with this deal "
                        "(account, competitor, stakeholder, stage or stated concern), so this "
                        "recommendation rests on the general record for this objection type.")
    elif len(contextual_rows) < CONTEXT_MIN_ROWS:
        context_note = (f"Only {len(contextual_rows)} past experience matches this deal's context — "
                        f"not enough to prefer it over the general record, so it is shown as "
                        f"supporting evidence only.")
    else:
        context_note = (f"{len(contextual_rows)} context-matched experience(s) exist but none of them "
                        f"succeeded, so the general record is used to choose the tactic instead.")

    def _row(t):
        row = _short(t["exp"])
        row["context_score"] = t["score"]
        row["context_matches"] = t["matches"]
        row["contextual"] = t["score"] >= CONTEXT_MATCH_MIN
        # kept for backwards compatibility with the previous response shape
        row["same_competitor"] = any(m["field"] == "competitor" for m in t["matches"])
        return row

    items = [_row(t) for t in scored[:8]]
    context_items = [_row(t) for t in contextual[:8]]

    summary = {
        "relevant_total": len(relevant),
        "successful": sum(1 for e in relevant if e.get("outcome") == "positive"),
        "unsuccessful": sum(1 for e in relevant if e.get("outcome") == "negative"),
        "neutral": sum(1 for e in relevant if e.get("outcome") == "neutral"),
    }
    context_summary = {
        "relevant_total": len(contextual_rows),
        "successful": sum(1 for e in contextual_rows if e.get("outcome") == "positive"),
        "unsuccessful": sum(1 for e in contextual_rows if e.get("outcome") == "negative"),
        "neutral": sum(1 for e in contextual_rows if e.get("outcome") == "neutral"),
    }
    # Which dimensions actually did the matching, across the contextual rows.
    matched_dimensions = []
    for t in contextual:
        for m in t["matches"]:
            label = f"{m['label']}: {m['past']}"
            if label not in matched_dimensions:
                matched_dimensions.append(label)

    avoid_detail = avoid_list(general_stats)
    avoid = [a["tactic"] for a in avoid_detail]   # legacy shape, still a list of strings

    # ---- already tried in THIS deal? ----
    this_deal = [e for e in relevant if e.get("deal_id") == deal["id"]]
    tried_here = [f"'{e['tactic_used']}' ({e['outcome']})" for e in this_deal if e.get("tactic_used")]

    # Cautions are tagged by kind. The UI renders 'conflict' and 'avoid' in
    # their own dedicated panels, so tagging lets it show each caveat exactly
    # once instead of repeating it in a catch-all box. The legacy flat
    # `caution` string still contains every one of them, in order.
    cautions = []

    def caution(kind, text):
        cautions.append({"kind": kind, "text": text})

    conflict = None
    top = None
    top_overall = None

    if use_context:
        top = context_candidates[0]
        top_overall = _stat_for(general_stats, top["tactic"])
        basis = "learned-contextual"
    elif general_candidates:
        top = general_candidates[0]
        top_overall = top
        basis = "learned"
    else:
        basis = "generic"

    if top is None:
        confidence = "none"
        recommendation = generic["tactic"]
        reasoning = generic["reasoning"]
        next_action = generic["next_action"]
        if not relevant:
            caution("evidence",
                    f"No outcomes have been recorded for {objection_type} objections yet, so this is "
                    "standard sales guidance, not something learned from your team's history.")
        else:
            caution("evidence",
                    f"{len(relevant)} relevant experience(s) exist but none succeeded, so no tactic can "
                    "be recommended from evidence. Falling back to generic guidance.")
    else:
        confidence = _confidence(top, top_overall)
        recommendation = top["tactic"]
        conflict = conflict_view(top_overall)

        # The winning experiences behind the number we are about to quote.
        winners = [e for e in (contextual_rows if use_context else relevant)
                   if _key(e.get("tactic_used")) == _key(top["tactic"]) and e["outcome"] == "positive"]
        best = sorted(winners, key=lambda e: e.get("date") or "", reverse=True)[0]

        if use_context:
            winner_matches = []
            for t in contextual:
                if _key(t["exp"].get("tactic_used")) == _key(top["tactic"]) and t["exp"]["outcome"] == "positive":
                    winner_matches.extend(t["matches"])
            phrase = _context_phrase(winner_matches)
            reasoning = (
                f"'{top['tactic']}' worked {top['successful']} of {top['n']} time(s) in situations that "
                f"share this deal's context"
                + (f" — {phrase}" if phrase else "")
                + (f", most recently at {best['customer']}" if best.get("customer") else "")
                + (f": {best['reason']}" if best.get("reason") else "")
                + "."
                + (f" Across all {objection_type} objections it is "
                   f"{top_overall['successful']} of {top_overall['n']}."
                   if top_overall and top_overall["n"] != top["n"] else "")
            )
        else:
            reasoning = (
                f"'{top['tactic']}' worked {top['successful']} of {top['n']} time(s) it was tried on "
                f"{objection_type} objections"
                + (f" (most recently at {best['customer']}"
                   + (f": {best['reason']}" if best.get("reason") else "") + ")."
                   if best.get("customer") else ".")
            )

        next_action = (f"Repeat what worked at {best['customer']}: {best['actual_action']}"
                       if best.get("actual_action") and best.get("customer")
                       else f"Apply '{top['tactic']}' and record what happens.")

        if confidence == "mixed" and conflict:
            caution("conflict",
                    f"Mixed evidence: '{conflict['tactic']}' succeeded {conflict['successful']} time(s) and "
                    f"failed {conflict['unsuccessful']} time(s). " + conflict["check_before_using"])
        elif confidence == "weak":
            caution("confidence",
                    "Weak evidence: this rests on a single experience"
                    + (f" at {top['customers'][0]}" if top["customers"] else "")
                    + ". Treat it as a hypothesis, not a proven playbook.")
        elif confidence == "moderate":
            caution("confidence",
                    f"Limited sample: only {top['n']} recorded experiences for this tactic.")

        if use_context and context_summary["relevant_total"] < 3:
            caution("context",
                    f"The context match rests on {context_summary['relevant_total']} experience(s); "
                    "more outcomes in comparable situations would firm this up.")

        scope_stats = context_stats if use_context else general_stats
        others = [s for s in scope_stats if s is not top and s["successful"] > 0]
        if others and confidence != "mixed" and abs(others[0]["score"] - top["score"]) < 0.1:
            caution("alternative",
                    f"'{others[0]['tactic']}' has a similar record and is a reasonable alternative.")

    # On the generic path there is no evidence-based claim to qualify, and the
    # caution already says so -- repeating the context note would just be noise.
    # It stays available on rec["context"]["note"] either way.
    if context_note and basis != "generic":
        caution("context", context_note)
    for a in avoid_detail:
        caution("avoid",
                f"{a['severity'].capitalize()}: '{a['tactic']}' has failed {a['unsuccessful']} time(s) "
                f"with no successes on record"
                + (f" — {a['failed_when'][0]}" if a["failed_when"] else "") + ".")
    if tried_here:
        caution("repeat", "Already tried in this deal: " + ", ".join(tried_here) + ".")

    rec = {
        "objection_type": objection_type,
        "basis": basis,
        "confidence": confidence,
        "recommendation": recommendation,
        "reasoning": reasoning,
        "next_action": next_action,
        # Legacy flat string: every caution, in order, exactly as before.
        "caution": " ".join(c["text"] for c in cautions),
        # Structured: [{kind, text}], so a client can render each caveat in the
        # panel that owns it rather than repeating it.
        "cautions": cautions,
        "conflict": conflict,
        "avoid": avoid,
        "avoid_detail": avoid_detail,
        "generic_baseline": {"tactic": generic["tactic"], "reasoning": generic["reasoning"],
                             "next_action": generic["next_action"]},
        "context": {
            "query": {k: ctx[k] for k in CONTEXT_QUERY_FIELDS if ctx.get(k)},
            "stakeholder_roles": ctx.get("stakeholder_roles") or [],
            "used": use_context,
            "note": context_note,
            "threshold": CONTEXT_MATCH_MIN,
            "min_rows": CONTEXT_MIN_ROWS,
            "summary": context_summary,
            "matched_dimensions": matched_dimensions,
            "items": context_items,
            "tactics": context_stats,
            "for_recommended": top if use_context else None,
        },
        "evidence": {
            "summary": summary,
            "tactics": general_stats,
            "items": items,
            "for_recommended": top_overall,
        },
    }
    rec["why"] = build_why(rec)
    return rec


# ---------------------------------------------------------------------------
# "Why this recommendation?" -- the compact, fully deterministic audit card
# ---------------------------------------------------------------------------

def build_why(rec: dict) -> dict:
    s = rec["evidence"]["summary"]
    c = rec["context"]["summary"]
    conflict = rec.get("conflict")
    top = rec["evidence"]["for_recommended"]
    return {
        "relevant_experiences": s["relevant_total"],
        "successful": s["successful"],
        "failed": s["unsuccessful"],
        "neutral": s["neutral"],
        "contextual_matches": c["relevant_total"],
        "contextual_used": rec["context"]["used"],
        "matched_on": rec["context"]["matched_dimensions"],
        "context_note": rec["context"]["note"],
        "recommendation": rec["recommendation"],
        "basis": rec["basis"],
        "confidence": rec["confidence"],
        "tactic_record": (f"{top['successful']} successful / {top['unsuccessful']} unsuccessful / "
                          f"{top['neutral']} neutral across {top['n']} attempt(s)") if top else None,
        "conflicting_evidence": conflict,
        "cautions": rec["cautions"],
        "avoid": rec["avoid_detail"],
        "next_action": rec["next_action"],
    }


# ---------------------------------------------------------------------------
# LEARNING: what changed since the last recommendation, and what changes next
# ---------------------------------------------------------------------------

def _counts(rec: dict) -> dict:
    s = rec["evidence"]["summary"]
    return {"total": s["relevant_total"], "successful": s["successful"],
            "unsuccessful": s["unsuccessful"], "neutral": s["neutral"]}


def _signature(rec: dict) -> str:
    c = _counts(rec)
    return (f"{rec['basis']}|{_key(rec['recommendation'])}|{c['total']}|{c['successful']}"
            f"|{c['unsuccessful']}|{c['neutral']}")


def compute_changes(previous: Optional[dict], rec: dict, experiences: list) -> dict:
    """Deterministic 'what did the agent learn since last time'."""
    if not previous:
        return {"has_previous": False, "changed": False, "previous": None, "new_experiences": [],
                "summary": f"First recommendation for {rec['objection_type']} objections."}
    prev_ids = set(previous.get("experience_ids") or [])
    new = [_short(e) for e in experiences
           if e.get("objection_type") == rec["objection_type"] and e["id"] not in prev_ids]
    if previous.get("signature") == _signature(rec):
        return {"has_previous": True, "changed": False, "previous": _prev_view(previous),
                "new_experiences": [],
                "summary": "No new outcomes recorded since the last recommendation, so nothing new has been learned."}
    pc, cc = previous["counts"], _counts(rec)
    parts = []
    if previous["basis"] != rec["basis"]:
        parts.append(f"moved from {previous['basis']} guidance to {rec['basis']} guidance")
    if _key(previous["tactic"]) != _key(rec["recommendation"]):
        parts.append(f"changed the recommended tactic from '{previous['tactic']}' to '{rec['recommendation']}'")
    parts.append(f"evidence grew from {pc['total']} to {cc['total']} relevant experience(s) "
                 f"({cc['successful']} successful, {cc['unsuccessful']} unsuccessful, {cc['neutral']} neutral)")
    if previous.get("confidence") != rec["confidence"]:
        parts.append(f"confidence went from {previous['confidence']} to {rec['confidence']}")
    return {"has_previous": True, "changed": True, "previous": _prev_view(previous),
            "new_experiences": new, "summary": "Since last time the agent " + "; ".join(parts) + "."}


def _prev_view(p: dict) -> dict:
    return {k: p.get(k) for k in ("version", "basis", "tactic", "confidence", "counts",
                                  "created_at", "first_interaction")}


def build_forecast(rec: dict) -> str:
    """BETTER FUTURE RECOMMENDATION, stated as arithmetic rather than a promise.
    Everything here is computed from the counts already on `rec`."""
    t = rec["objection_type"]
    n = rec["evidence"]["summary"]["relevant_total"]
    top = rec["evidence"]["for_recommended"]
    if top is None:
        return (f"Nothing has succeeded for {t} objections yet, so the agent is still on generic "
                f"guidance. Record one positive outcome and the next recommendation stops being "
                f"generic: it will be computed from real evidence instead.")
    parts = [f"Recording the outcome below makes the next {t} recommendation a count over "
             f"{n + 1} experiences instead of {n}."]
    parts.append(
        f"If '{top['tactic']}' is recorded positive it becomes {top['successful'] + 1} of {top['n'] + 1}"
        + ("" if rec["confidence"] == "strong" else f", which can lift confidence above '{rec['confidence']}'")
        + f"; if negative it becomes {top['successful']} of {top['n'] + 1} and the agent will report it "
          f"as mixed evidence from then on.")
    ctx = rec["context"]
    if ctx["query"]:
        known = ", ".join(f"{k}: {v}" for k, v in ctx["query"].items()
                          if k != "concern" and isinstance(v, str))
        parts.append(f"The outcome is stored with this situation's context ({known}), so it will also "
                     f"join the {ctx['summary']['relevant_total']} context-matched experience(s) the next "
                     f"time a deal looks like this one.")
    return " ".join(parts)


def build_journey(rows: list) -> dict:
    """The learning curve, read back out of the stored recommendation log.

    Every interaction number below is a real count from store.save_recommendation
    (`first_interaction`), never an illustrative milestone. Milestones that have
    not happened yet simply do not appear."""
    if not rows:
        return {"interactions": 0, "distinct_states": 0, "milestones": [],
                "summary": "No recommendations have been requested for this objection type yet."}

    total = sum(r.get("times_asked", 1) for r in rows)

    def mk(row, stage, label):
        return {
            "interaction": row.get("first_interaction") or row.get("version") or 1,
            "version": row.get("version"),
            "stage": stage,
            "label": label,
            "basis": row.get("basis"),
            "tactic": row.get("tactic"),
            "confidence": row.get("confidence"),
            "counts": row.get("counts") or {},
            "created_at": row.get("created_at"),
            # which deal was asking -- the log is shared across deals, so this
            # keeps a milestone attributable rather than anonymous
            "deal_id": row.get("deal_id"),
        }

    milestones, seen = [], set()

    def push(row, stage, label):
        if row is None:
            return
        marker = (row.get("first_interaction"), row.get("version"))
        if marker in seen:
            return
        seen.add(marker)
        milestones.append(mk(row, stage, label))

    first = rows[0]
    c = first.get("counts") or {}
    push(first, "start",
         "Generic guidance — nothing recorded for this objection type yet"
         if first.get("basis") == "generic"
         else f"First recommendation — already backed by {c.get('total', 0)} experience(s)")

    learned = next((r for r in rows if r.get("basis") and r["basis"] != "generic"), None)
    if learned is not first:
        lc = (learned or {}).get("counts") or {}
        push(learned, "personalized",
             f"Personalized — recommendation now computed from {lc.get('total', 0)} recorded "
             f"experience(s) ({lc.get('successful', 0)} successful)")

    contextual = next((r for r in rows if r.get("basis") == "learned-contextual"), None)
    if contextual is not None and contextual is not first:
        push(contextual, "contextual",
             "Context-matched — evidence narrowed to past deals that share this deal's "
             "competitor, stakeholder or stage")

    # A changed recommendation is the most legible form of learning there is --
    # but only when the EVIDENCE changed underneath it. The log is keyed by
    # objection type, so two different deals asking about "pricing" interleave
    # here, and a tactic can flip simply because the second deal has a different
    # competitor. That is context-sensitivity, not learning, so it is not a
    # milestone: we require the counts to have actually moved.
    prev = None
    for row in rows:
        if prev is not None and _key(row.get("tactic")) != _key(prev.get("tactic")):
            rc, pc = row.get("counts") or {}, prev.get("counts") or {}
            if rc != pc:
                push(row, "revised",
                     f"Revised — evidence grew from {pc.get('total', 0)} to {rc.get('total', 0)} "
                     f"experience(s) and changed the recommended tactic from "
                     f"'{prev.get('tactic')}' to '{row.get('tactic')}'")
        prev = row

    strong = next((r for r in rows if r.get("confidence") in ("strong", "moderate")), None)
    if strong is not None:
        sc = strong.get("counts") or {}
        push(strong, "stronger",
             f"Stronger evidence — confidence reached '{strong.get('confidence')}' on "
             f"{sc.get('total', 0)} experience(s)")

    latest = rows[-1]
    lc = latest.get("counts") or {}
    push(latest, "current",
         f"Current — '{latest.get('tactic')}' at {latest.get('confidence')} confidence, "
         f"from {lc.get('total', 0)} experience(s)")

    milestones.sort(key=lambda m: (m["interaction"], m["version"] or 0))

    first_m, last_m = milestones[0], milestones[-1]
    objection = rows[0].get("objection_type")
    if len(milestones) == 1:
        summary = (f"{total} recommendation(s) requested for {objection} objections so far; the agent "
                   f"is still on its first evidence state "
                   f"({first_m['counts'].get('total', 0)} experience(s), {first_m['confidence']} confidence).")
    else:
        moved = []
        if first_m["basis"] != last_m["basis"]:
            moved.append(f"from {first_m['basis']} to {last_m['basis']} guidance")
        if _key(first_m["tactic"] or "") != _key(last_m["tactic"] or ""):
            moved.append(f"from '{first_m['tactic']}' to '{last_m['tactic']}'")
        if first_m["confidence"] != last_m["confidence"]:
            moved.append(f"from {first_m['confidence']} to {last_m['confidence']} confidence")
        moved.append(f"from {first_m['counts'].get('total', 0)} to "
                     f"{last_m['counts'].get('total', 0)} recorded experience(s)")
        summary = (f"Over {total} recommendation(s) for {objection} objections across "
                   f"{len(rows)} distinct evidence state(s), the agent moved "
                   + "; ".join(moved)
                   + f" (interaction {first_m['interaction']} → {last_m['interaction']}).")

    return {"interactions": total, "distinct_states": len(rows),
            "milestones": milestones, "summary": summary}


def journey(objection_type: str) -> dict:
    """Public entry point for GET /api/learning/journey."""
    return build_journey(store.list_recommendations(objection_type))


# ---------------------------------------------------------------------------
# Orchestrators (these are what main.py and seed.py call)
# ---------------------------------------------------------------------------

REFLECT_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "next_action": {"type": "string"},
        "agent_note": {"type": "string"},
    },
    "required": ["reasoning", "next_action"],
}


def _evidence_lines(items: list) -> str:
    lines = []
    for e in items:
        line = (f"- {e['date'] and e['date'][:10]} | {e['customer']} | stakeholder: {e.get('stakeholder') or 'n/a'} "
                f"| stage: {e.get('deal_stage') or 'n/a'} | competitor: {e.get('competitor') or 'n/a'} "
                f"| tactic: {e['tactic_used']} | outcome: {e['outcome']} | reason: {e.get('reason') or 'n/a'}")
        if e.get("context_matches"):
            line += " | context match: " + ", ".join(m["label"] for m in e["context_matches"])
        lines.append(line)
    return "\n".join(lines) or "- (none)"


def _reflect_prompt(deal, rec, objection_text, stakeholder, competitor, playbook_memories) -> str:
    s = rec["evidence"]["summary"]
    c = rec["context"]["summary"]
    mem = "\n".join(f"- {m['text']}" for m in playbook_memories[:4]) or "- (none recalled)"
    conflict = rec.get("conflict")
    conflict_block = "No conflicting evidence on the recommended tactic."
    if conflict:
        conflict_block = (
            f"CONFLICTING EVIDENCE on '{conflict['tactic']}': {conflict['successful']} successful, "
            f"{conflict['unsuccessful']} unsuccessful.\n"
            f"  Worked when: {'; '.join(conflict['worked_when']) or 'n/a'}\n"
            f"  Failed when: {'; '.join(conflict['failed_when']) or 'n/a'}\n"
            f"  {conflict['check_before_using']}")
    context_block = (
        f"CONTEXT MATCH: {c['relevant_total']} of those experiences share verified structured context "
        f"with this deal ({c['successful']} successful, {c['unsuccessful']} unsuccessful, "
        f"{c['neutral']} neutral). Matched on: "
        f"{', '.join(rec['context']['matched_dimensions']) or 'nothing'}.\n"
        + (f"These were used to choose the tactic.\n{_evidence_lines(rec['context']['items'])}\n"
           if rec["context"]["used"]
           else f"NOT used to choose the tactic. {rec['context']['note']}\n"))
    return (
        "You are advising a salesperson. Write a short explanation for the recommendation below.\n"
        f"Deal: {deal['company']} (stage: {deal.get('stage')}). Contact: {deal['contact_name']}, {deal['contact_title']}.\n"
        f"Objection type: {rec['objection_type']}."
        + (f" What was said: \"{objection_text}\"." if objection_text else "")
        + (f" Stakeholder raising it: {stakeholder}." if stakeholder else "")
        + (f" Competitor mentioned: {competitor}." if competitor else "") + "\n\n"
        f"VERIFIED EVIDENCE (computed from the team's recorded outcomes): {s['relevant_total']} relevant experiences: "
        f"{s['successful']} successful, {s['unsuccessful']} unsuccessful, {s['neutral']} neutral.\n"
        f"{_evidence_lines(rec['evidence']['items'])}\n\n"
        f"{context_block}\n"
        f"{conflict_block}\n\n"
        f"Related memories recalled from the playbook:\n{mem}\n\n"
        f"RECOMMENDATION ALREADY CHOSEN: {rec['recommendation']} "
        f"(basis: {rec['basis']}, confidence: {rec['confidence']}).\n\n"
        "Rules: (1) Do NOT invent customers, numbers, quotes, statistics or outcomes -- every count you "
        "state must appear verbatim above, and you may not compute new ones. (2) Do not change the "
        "recommendation. (3) Do not claim two deals are similar unless the CONTEXT MATCH section lists "
        "the dimension that matched; if it was not used, do not imply the recommendation is tailored. "
        "(4) If basis is 'generic', say plainly this is generic guidance. (5) If the evidence is weak or "
        "conflicting, say so and name the condition to check.\n"
        "Return JSON with: reasoning (2-3 sentences, why this fits THIS deal), next_action (one concrete step for "
        "the next conversation with this deal's contact), agent_note (optional, one sentence, only if this deal's own "
        "history adds something important; otherwise empty)."
    )


def recommend(deal: dict, objection_type: str, objection_text: str = None,
              stakeholder: str = None, competitor: str = None,
              deal_stage: str = None) -> dict:
    """Run MEMORY -> EVIDENCE -> CONTEXT MATCH -> RECOMMENDATION, record the
    LEARNING delta, and project the next one."""
    if objection_type not in OBJECTION_TYPES:
        raise ValueError(f"objection_type must be one of {OBJECTION_TYPES}")
    experiences = store.list_experiences()
    rec = build_recommendation(deal, objection_type, experiences, competitor=competitor,
                               stakeholder=stakeholder, deal_stage=deal_stage,
                               objection_text=objection_text)

    hindsight = {"recall_ok": True, "reflect_ok": True, "errors": []}
    query = (f"{objection_type} objection" + (f": {objection_text}" if objection_text else "")
             + (f" competitor {competitor}" if competitor else "")
             + (f" stakeholder {stakeholder}" if stakeholder else ""))

    # ---- MEMORY (Hindsight recall) ----
    memory = {"deal": [], "playbook": [], "deal_bank": hs.deal_bank(deal["id"]),
              "playbook_bank": hs.PLAYBOOK_BANK}
    try:
        memory["deal"] = hs.recall_deal_memories(deal["id"], query)
    except Exception as e:  # keep the pipeline alive; surface the failure
        hindsight["recall_ok"] = False
        hindsight["errors"].append(f"deal recall failed: {type(e).__name__}: {str(e)[:160]}")
    try:
        memory["playbook"] = hs.recall_playbook_memories(objection_type, query, competitor=competitor)
    except Exception as e:
        hindsight["recall_ok"] = False
        hindsight["errors"].append(f"playbook recall failed: {type(e).__name__}: {str(e)[:160]}")

    # ---- RECOMMENDATION wording (Hindsight reflect) ----
    rec["reasoning_source"] = "evidence-only"
    rec["agent_note"] = ""
    try:
        out = hs.reflect_json(hs.deal_bank(deal["id"]),
                              _reflect_prompt(deal, rec, objection_text, stakeholder, competitor, memory["playbook"]),
                              REFLECT_SCHEMA)
        if str(out.get("reasoning", "")).strip():
            rec["reasoning"] = str(out["reasoning"]).strip()
            rec["reasoning_source"] = "hindsight"
        if str(out.get("next_action", "")).strip():
            rec["next_action"] = str(out["next_action"]).strip()
        rec["agent_note"] = str(out.get("agent_note") or "").strip()
    except Exception as e:
        hindsight["reflect_ok"] = False
        hindsight["errors"].append(f"reflect failed: {type(e).__name__}: {str(e)[:160]}")

    # reflect() may have rewritten next_action; keep the audit card in sync with
    # what is actually shown, but never let it rewrite a count.
    rec["why"]["next_action"] = rec["next_action"]

    # ---- LEARNING (compare with the previous recommendation, then log this one) ----
    history_before = store.list_recommendations(objection_type)
    previous = history_before[-1] if history_before else None
    changes = compute_changes(previous, rec, experiences)
    saved = store.save_recommendation({
        "objection_type": objection_type, "deal_id": deal["id"], "basis": rec["basis"],
        "tactic": rec["recommendation"], "confidence": rec["confidence"], "counts": _counts(rec),
        "signature": _signature(rec),
        "experience_ids": [e["id"] for e in experiences if e.get("objection_type") == objection_type],
    })
    rec["id"] = saved["id"]
    rec["version"] = saved["version"]
    rec["interaction"] = saved.get("last_interaction") or saved.get("first_interaction") or saved["version"]

    rows = store.list_recommendations(objection_type)
    history = [{"version": r["version"], "basis": r["basis"], "tactic": r["tactic"],
                "confidence": r["confidence"], "counts": r["counts"], "created_at": r["created_at"],
                "interaction": r.get("first_interaction") or r["version"],
                "times_asked": r.get("times_asked", 1)}
               for r in rows]
    return {
        "deal_id": deal["id"],
        "memory": memory,
        "recommendation": rec,
        "learning": {
            "changes": changes,
            "history": history,
            "journey": build_journey(rows),
            "forecast": build_forecast(rec),
        },
        "hindsight": hindsight,
    }


def record_outcome(deal: Optional[dict], data: dict, retain_deal_bank: bool = True,
                   add_timeline_note: bool = True, customer: str = None) -> dict:
    """Store one outcome. Order matters: retain to Hindsight FIRST, then write
    the ledger, so the ledger never claims something Hindsight was not taught.
    `deal` may be None for historical playbook-only experiences (then `customer`
    must be given)."""
    objection_type = (data.get("objection_type") or "").strip().lower()
    if objection_type not in OBJECTION_TYPES:
        raise ValueError(f"objection_type must be one of {OBJECTION_TYPES}")
    outcome = normalize_outcome(data.get("outcome"))
    tactic = (data.get("tactic_used") or "").strip()
    if not tactic:
        raise ValueError("tactic_used is required (which tactic did the rep actually use?)")

    exp = {
        "id": data.get("id") or __import__("uuid").uuid4().hex[:8],
        "deal_id": deal["id"] if deal else None,
        "customer": (deal["company"] if deal else customer),
        "stakeholder": data.get("stakeholder") or None,
        "competitor": data.get("competitor") or None,
        "objection_type": objection_type,
        "objection_text": data.get("objection_text") or None,
        "recommended_tactic": data.get("recommended_tactic") or None,
        "tactic_used": tactic,
        "actual_action": data.get("actual_action") or None,
        "outcome": outcome,
        "result": data.get("result") or None,
        "reason": data.get("reason") or None,
        "deal_stage": data.get("deal_stage") or (deal.get("stage") if deal else None),
        "recommendation_id": data.get("recommendation_id") or None,
        "date": (data.get("date") or store.now_iso()),
        "source": data.get("source") or "user",
    }
    if not exp["customer"]:
        raise ValueError("customer is required for experiences without a deal")
    exp["timestamp"] = exp["date"]

    # Already in the ledger (a re-run of seed.py, or a replayed request)? Then
    # Hindsight was taught this once already -- do not retain a second copy.
    existing = store.get_experience(exp["id"])
    if existing:
        return {"experience": existing, "hindsight": {"retained_to": [], "skipped": "already recorded"},
                "duplicate": True}

    retained = hs.retain_outcome(exp, to_deal_bank=retain_deal_bank and deal is not None)
    saved = store.add_experience(exp)

    if deal and add_timeline_note:
        text = (f"Outcome recorded ({outcome}). {objection_type.capitalize()} objection"
                + (f" from {exp['stakeholder']}" if exp["stakeholder"] else "")
                + f". Tactic used: {tactic}."
                + (f" Result: {exp['result']}" if exp["result"] else "")
                + (f" Reason: {exp['reason']}" if exp["reason"] else ""))
        store.add_note(deal_id=deal["id"], note_type="outcome", raw_notes=text,
                       objection_type=objection_type, outcome=outcome, tactic_used=tactic,
                       call_date=exp["date"], stakeholder=exp["stakeholder"], competitor=exp["competitor"])
    return {"experience": saved, "hindsight": retained, "duplicate": False}
