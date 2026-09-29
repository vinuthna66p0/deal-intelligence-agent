"""
Self-contained checks for the learning loop. No pytest, no network, no
Hindsight key -- stdlib only, so `requirements.txt` is unchanged.

    cd backend
    python test_learning.py

Hindsight is stubbed at the hindsight_service boundary and reflect_json() is
made to fail, which means every assertion below runs against the deterministic
fallback path. That is the point: if a count, a context match, a conflict or a
confidence level were ever produced by the LLM rather than by code, these tests
would go dark. They do not.

The store is redirected to a temporary file, so running this never touches
your real data.json.
"""
import io
import contextlib
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import store

store.DATA_PATH = os.path.join(tempfile.mkdtemp(prefix="deal-intel-test-"), "data.json")
store.reset_all()

# hindsight_service imports the client at module load. These tests stub the
# service layer above it, so the real package is never called -- stand in a
# placeholder when it isn't installed, and leave it alone when it is.
try:
    import hindsight_client  # noqa: F401
except ModuleNotFoundError:
    import types
    _placeholder = types.ModuleType("hindsight_client")
    _placeholder.Hindsight = type("Hindsight", (), {"__init__": lambda self, **kw: None})
    sys.modules["hindsight_client"] = _placeholder

import hindsight_service as hs

# ---------------------------------------------------------------------------
# Stub Hindsight at the service boundary.
# ---------------------------------------------------------------------------
RETAINED = []


def _retain_outcome(exp, to_deal_bank=True):
    banks = ([hs.deal_bank(exp["deal_id"])] if to_deal_bank and exp.get("deal_id") else [])
    banks.append(hs.PLAYBOOK_BANK)
    RETAINED.append({"id": exp["id"], "banks": banks})
    return {"retained_to": banks}


def _reflect_json(bank_id, query, schema):
    raise ValueError("reflect() did not return usable JSON")


hs.retain_outcome = _retain_outcome
hs.retain_call_note = lambda **kw: {"ok": True}
hs.retain_playbook_lesson = lambda **kw: {"ok": True}
hs.recall_deal_memories = lambda *a, **kw: []
hs.recall_playbook_memories = lambda *a, **kw: []
hs.reflect_json = _reflect_json

import learning
import seed

FAILURES = []


def check(name, condition, detail=""):
    print(("  pass  " if condition else "  FAIL  ") + name + (f"   [{detail}]" if detail else ""))
    if not condition:
        FAILURES.append(name)


def section(title):
    print("\n" + title)
    print("-" * len(title))


def quiet(fn, *a, **kw):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **kw)


# ---------------------------------------------------------------------------
section("1. Seeding is idempotent")
# ---------------------------------------------------------------------------
quiet(seed.main)
first = (len(store.list_deals()), len(store.list_experiences()),
         sum(len(store.list_notes(d["id"])) for d in store.list_deals()))
seed._created = {"deals": 0, "notes": 0, "experiences": 0}
seed._skipped = {"deals": 0, "notes": 0, "experiences": 0}
quiet(seed.main)
second = (len(store.list_deals()), len(store.list_experiences()),
          sum(len(store.list_notes(d["id"])) for d in store.list_deals()))
check("re-seeding creates no duplicate deals/experiences/notes", first == second,
      f"{first} -> {second}")
check("second seed run reports nothing created",
      seed._created == {"deals": 0, "notes": 0, "experiences": 0}, str(seed._created))

deals = {d["company"]: d for d in store.list_deals()}
coldstream, meridian, brightline = (deals["Coldstream Robotics"],
                                    deals["Meridian Logistics"],
                                    deals["Brightline Health"])

COLDSTREAM_PRICING = dict(
    objection_type="pricing",
    objection_text="Ionic bundles this into a suite we already pay for.",
    stakeholder="Elena Ruiz", competitor="Ionic Sales Cloud", deal_stage="Negotiation")

res = learning.recommend(coldstream, **COLDSTREAM_PRICING)
rec, ctx = res["recommendation"], res["recommendation"]["context"]

# ---------------------------------------------------------------------------
section("2. Context-aware evidence")
# ---------------------------------------------------------------------------
check("basis is learned-contextual", rec["basis"] == "learned-contextual", rec["basis"])
check("contextual evidence was used to choose the tactic", ctx["used"] is True)
check("general and contextual evidence are distinguished",
      ctx["summary"]["relevant_total"] < rec["evidence"]["summary"]["relevant_total"],
      f"{ctx['summary']['relevant_total']} contextual of {rec['evidence']['summary']['relevant_total']} total")
for label in ("Same competitor", "Comparable role", "Same deal stage", "Same account"):
    check(f"matched dimension: {label}",
          any(d.startswith(label) for d in ctx["matched_dimensions"]))
check("every contextual row is at or above the threshold",
      all(i["context_score"] >= learning.CONTEXT_MATCH_MIN and i["context_matches"]
          for i in ctx["items"]))
check("TCO reframing is recommended", "total cost of ownership" in rec["recommendation"].lower(),
      rec["recommendation"])

# Similarity must never be claimed without data behind it.
bare = learning.recommend(meridian, "pricing", objection_text="Too expensive")["recommendation"]
check("no context supplied => contextual evidence not used", bare["context"]["used"] is False)
check("falls back to the general record", bare["basis"] == "learned", bare["basis"])
check("states why context was not used", bool(bare["context"]["note"]))

# A stage match alone is deliberately too weak to count as similarity.
score, _ = learning.match_context(
    {"deal_stage": "Negotiation"},
    learning.context_of({"stage": "Negotiation"}))
check("deal stage alone is below the context threshold", score < learning.CONTEXT_MATCH_MIN,
      f"score {score} < {learning.CONTEXT_MATCH_MIN}")

# A role is only borrowed from the deal's contact when the stakeholder IS that
# contact -- otherwise we must learn nothing about that person's role.
other = learning.context_of(
    {"company": "X", "contact_name": "Elena Ruiz", "contact_title": "Head of Procurement"},
    stakeholder="Some Other Person")
check("a role is not invented for a stakeholder who is not the deal contact",
      other["stakeholder_roles"] == [], str(other["stakeholder_roles"]))

# ---------------------------------------------------------------------------
section("3. Contradiction-aware learning")
# ---------------------------------------------------------------------------
conflict = rec["conflict"]
check("conflict block is present", conflict is not None)
check("labelled 'Mixed evidence'", conflict and conflict["label"] == "Mixed evidence")
check("counts both sides", conflict and (conflict["successful"], conflict["unsuccessful"]) == (2, 1),
      conflict and f"{conflict['successful']} up / {conflict['unsuccessful']} down")
check("lists the conditions where it worked", conflict and len(conflict["worked_when"]) == 2)
check("lists the conditions where it failed", conflict and len(conflict["failed_when"]) == 1)
check("names the condition to check first",
      conflict and "confirm none of these applies" in conflict["check_before_using"])
check("a clean contextual slice cannot hide a known failure (confidence is mixed)",
      rec["confidence"] == "mixed", rec["confidence"])

severities = {a["tactic"]: a["severity"] for a in rec["avoid_detail"]}
check("two failures and no successes reads as 'avoid'",
      severities.get("Offer an annual pre-pay discount") == "avoid", str(severities))
check("one failure and no successes reads only as 'use caution'",
      severities.get("Direct feature-by-feature comparison") == "use caution", str(severities))
check("a tactic with successes never appears in the avoid list",
      rec["recommendation"] not in severities)
check("a tactic with zero successes is never recommended",
      all(s["successful"] > 0 for s in rec["evidence"]["tactics"]
          if s["tactic"] == rec["recommendation"]))

# Ranking must not be a popularity contest: at a comparable success rate, a
# contradicted tactic loses to a clean one.
ranked = learning.tactic_stats([
    {"tactic_used": "clean", "outcome": "positive", "customer": "A", "date": "2026-01-01"},
    {"tactic_used": "clean", "outcome": "positive", "customer": "B", "date": "2026-01-02"},
    {"tactic_used": "contested", "outcome": "positive", "customer": "C", "date": "2026-01-03"},
    {"tactic_used": "contested", "outcome": "positive", "customer": "D", "date": "2026-01-04"},
    {"tactic_used": "contested", "outcome": "positive", "customer": "E", "date": "2026-01-05"},
    {"tactic_used": "contested", "outcome": "negative", "customer": "F", "date": "2026-01-06"},
])
check("a contradicted tactic does not win on raw count alone",
      ranked[0]["tactic"] == "clean",
      " > ".join(f"{s['tactic']}({s['status']},{s['score']})" for s in ranked))

# Each caveat should be attributable to exactly one panel.
kinds = {c["kind"] for c in rec["cautions"]}
check("cautions are tagged by kind", kinds and all(isinstance(c["kind"], str) for c in rec["cautions"]),
      str(sorted(kinds)))
check("legacy flat caution string still contains every caution",
      all(c["text"] in rec["caution"] for c in rec["cautions"]))

# ---------------------------------------------------------------------------
section("4. Outcome recording feeds the next recommendation")
# ---------------------------------------------------------------------------
before = rec["evidence"]["summary"]["relevant_total"]
before_ctx = ctx["summary"]["relevant_total"]
recorded = learning.record_outcome(coldstream, {
    "objection_type": "pricing", "outcome": "positive",
    "tactic_used": "Reframe as total cost of ownership, quantify hidden competitor cost",
    "stakeholder": "Elena Ruiz", "competitor": "Ionic Sales Cloud", "deal_stage": "Negotiation",
    "result": "Steering committee approved.", "reason": "the overhead number was verifiable",
    "recommendation_id": rec["id"]})
check("outcome lands in the experience ledger",
      store.get_experience(recorded["experience"]["id"]) is not None)
check("outcome is retained into both Hindsight banks",
      len(recorded["hindsight"]["retained_to"]) == 2, str(recorded["hindsight"]["retained_to"]))
check("ledger is written only after Hindsight is taught",
      RETAINED and RETAINED[-1]["id"] == recorded["experience"]["id"])

after_res = learning.recommend(coldstream, **COLDSTREAM_PRICING)
after = after_res["recommendation"]
check("evidence grows by exactly one", after["evidence"]["summary"]["relevant_total"] == before + 1,
      f"{before} -> {after['evidence']['summary']['relevant_total']}")
check("contextual evidence grows too",
      after["context"]["summary"]["relevant_total"] == before_ctx + 1,
      f"{before_ctx} -> {after['context']['summary']['relevant_total']}")
check("the learning delta reports the change", after_res["learning"]["changes"]["changed"] is True)
check("the new experience is named in the delta",
      any(e["id"] == recorded["experience"]["id"]
          for e in after_res["learning"]["changes"]["new_experiences"]))
check("replaying the same outcome id is a no-op",
      learning.record_outcome(coldstream, {
          "id": recorded["experience"]["id"], "objection_type": "pricing",
          "outcome": "positive", "tactic_used": "x"})["duplicate"] is True)

# ---------------------------------------------------------------------------
section("5. The learning curve comes from stored history")
# ---------------------------------------------------------------------------
rows = store.list_recommendations("pricing")
journey = after_res["learning"]["journey"]
check("interaction total equals the sum of stored times_asked",
      journey["interactions"] == sum(r.get("times_asked", 1) for r in rows),
      f"{journey['interactions']}")
check("every milestone interaction exists in the stored log",
      all(m["interaction"] in {r.get("first_interaction") for r in rows}
          for m in journey["milestones"]))
check("milestones ascend by interaction",
      [m["interaction"] for m in journey["milestones"]]
      == sorted(m["interaction"] for m in journey["milestones"]))

trust = learning.journey("trust")
check("the trust journey starts at interaction 1 with generic guidance",
      trust["milestones"][0]["interaction"] == 1 and trust["milestones"][0]["basis"] == "generic",
      f"interaction {trust['milestones'][0]['interaction']}, basis {trust['milestones'][0]['basis']}")
check("the trust journey reaches a learned state",
      any(m["basis"] != "generic" for m in trust["milestones"]))
check("a journey with no history is reported as empty, not invented",
      learning.build_journey([])["milestones"] == [])

# A tactic flip caused by a different deal's context is not 'learning', and
# must not be logged as a milestone.
same_counts = [
    {"objection_type": "x", "version": 1, "first_interaction": 1, "times_asked": 1,
     "basis": "learned", "tactic": "A", "confidence": "moderate",
     "counts": {"total": 5, "successful": 3, "unsuccessful": 2, "neutral": 0}},
    {"objection_type": "x", "version": 2, "first_interaction": 2, "times_asked": 1,
     "basis": "learned", "tactic": "B", "confidence": "moderate",
     "counts": {"total": 5, "successful": 3, "unsuccessful": 2, "neutral": 0}},
]
check("a tactic change with unchanged evidence is not a learning milestone",
      not any(m["stage"] == "revised" for m in learning.build_journey(same_counts)["milestones"]))
same_counts[1]["counts"] = {"total": 6, "successful": 4, "unsuccessful": 2, "neutral": 0}
check("a tactic change backed by new evidence IS a milestone",
      any(m["stage"] == "revised" for m in learning.build_journey(same_counts)["milestones"]))

check("forecast is deterministic arithmetic on the counts",
      "instead of" in after_res["learning"]["forecast"])

# ---------------------------------------------------------------------------
section("6. 'Why this recommendation?' matches the evidence")
# ---------------------------------------------------------------------------
why = after["why"]
for field in ("relevant_experiences", "successful", "failed", "neutral", "contextual_matches",
              "matched_on", "recommendation", "confidence", "conflicting_evidence",
              "cautions", "avoid", "next_action"):
    check(f"why.{field} is present", field in why)
check("why counts equal the evidence counts",
      (why["relevant_experiences"], why["successful"], why["failed"], why["neutral"])
      == tuple(after["evidence"]["summary"][k]
               for k in ("relevant_total", "successful", "unsuccessful", "neutral")))
check("why context count equals the context summary",
      why["contextual_matches"] == after["context"]["summary"]["relevant_total"])
check("why next_action equals the shown next action", why["next_action"] == after["next_action"])

# ---------------------------------------------------------------------------
section("7. Fallback and backwards compatibility")
# ---------------------------------------------------------------------------
check("reflect() failure still yields a complete recommendation",
      after["reasoning_source"] == "evidence-only" and bool(after["reasoning"]),
      after["reasoning_source"])
check("the reflect failure is surfaced, not swallowed", after_res["hindsight"]["reflect_ok"] is False)
check("counts survive the reflect failure untouched",
      after["evidence"]["summary"]["relevant_total"] == before + 1)

generic = learning.recommend(meridian, "timing", objection_text="Budget frozen")["recommendation"]
check("an objection type with no successes falls back to labelled generic guidance",
      generic["basis"] == "generic" and generic["confidence"] == "none",
      f"{generic['basis']}/{generic['confidence']}")
check("the generic fallback says so plainly", "generic guidance" in generic["caution"])

for key in ("objection_type", "basis", "confidence", "recommendation", "reasoning", "next_action",
            "caution", "avoid", "generic_baseline", "evidence", "id", "version"):
    check(f"recommendation.{key} still present", key in after)
for key in ("summary", "tactics", "items", "for_recommended"):
    check(f"evidence.{key} still present", key in after["evidence"])
check("avoid is still a flat list of strings",
      all(isinstance(a, str) for a in after["avoid"]))
check("evidence rows still carry same_competitor",
      "same_competitor" in after["evidence"]["items"][0])
check("recommend() still works without the new deal_stage argument",
      learning.recommend(brightline, "trust")["recommendation"]["objection_type"] == "trust")

# ---------------------------------------------------------------------------
print()
if FAILURES:
    print(f"FAILED ({len(FAILURES)}): " + "; ".join(FAILURES))
else:
    print("All checks passed.")
sys.exit(1 if FAILURES else 0)
