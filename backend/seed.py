"""
Populate a realistic demo: deals, call history, an experience ledger, and a
recorded generic -> learned recommendation arc.

What this creates:
  1. Three deals with several weeks of real call/email history, covering all
     five objection types (pricing, competitor, trust, timing, authority).
  2. A structured experience ledger (successes AND failures) behind those
     objections, retained into Hindsight as both raw call notes and
     structured outcome records -- see hindsight_service.py.
  3. A live "before vs. after" demo of the learning loop: it calls the real
     /recommend pipeline for a *trust* objection before any trust outcomes
     exist (generic answer), records several trust outcomes across deals,
     then calls /recommend again and prints the evidence-based answer that
     comes back. The same contrast is visible in the UI's Learning tab.
  4. A pricing scenario built specifically to exercise CONTEXT MATCHING and
     CONTRADICTION-AWARE learning: TCO reframing has two wins against Ionic
     Sales Cloud with procurement stakeholders in Negotiation (the context
     that matches Coldstream), and one loss in a completely different context
     -- so the agent recommends it while explicitly reporting mixed evidence
     and the condition to check first.

Re-running is safe. Every seeded deal is looked up by company name, every
seeded note by its text, and every seeded outcome carries a deterministic id
derived from a fixed seed key -- so a second run creates no duplicate deals,
notes, ledger rows or Hindsight memories.

Usage:
    cd backend
    python seed.py            # idempotent: fills in whatever is missing
    python seed.py --reset    # wipe data.json first (Hindsight banks are not
                              # touched; outcome document_ids dedupe there)
"""
import hashlib
import sys
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv

load_dotenv()

import store
import hindsight_service as hs
import learning

_created, _skipped = {"deals": 0, "notes": 0, "experiences": 0}, {"deals": 0, "notes": 0, "experiences": 0}


def seed_id(key: str) -> str:
    """Deterministic 8-char ledger id for a seeded experience. Derived from a
    fixed key (never from a date, which drifts between runs), so re-seeding
    resolves to the same row and both the ledger and Hindsight skip it."""
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:8]


def days_ago(n, hour=14):
    dt = datetime.now(timezone.utc) - timedelta(days=n)
    return dt.replace(hour=hour, minute=0, second=0, microsecond=0).isoformat()


def deal(company, **kwargs):
    """Get the seeded deal, creating it only if it isn't already there."""
    existing = store.find_deal_by_company(company)
    if existing:
        _skipped["deals"] += 1
        print(f"Deal already present: {company} ({existing['id']})")
        return existing
    d = store.create_deal(company=company, **kwargs)
    _created["deals"] += 1
    print(f"Created deal: {d['company']} ({d['id']})")
    return d


def log(key, d, note_type, raw_notes, ts, objection_type=None, outcome=None,
        tactic_used=None, stakeholder=None, competitor=None, reason=None,
        actual_action=None, result=None, objection_text=None):
    """Log one call/email: writes the raw note to the timeline + Hindsight's
    deal bank, and -- if it carries a tagged objection + outcome -- also
    writes a structured outcome record (ledger + Hindsight, both banks).
    Both halves are skipped if this exact note/outcome was already seeded."""
    deal_id = d["id"]
    if store.note_exists(deal_id, raw_notes):
        _skipped["notes"] += 1
    else:
        store.add_note(
            deal_id=deal_id, note_type=note_type, raw_notes=raw_notes,
            objection_type=objection_type, outcome=outcome, tactic_used=tactic_used,
            call_date=ts, stakeholder=stakeholder, competitor=competitor,
        )
        context = f"{note_type} with {d['contact_name']} ({d['contact_title']}) at {d['company']}"
        structured = {
            "customer": d["company"], "stakeholder": stakeholder or d["contact_name"],
            "objection": objection_type, "competitor": competitor, "tactic": tactic_used,
            "outcome": outcome, "date": ts[:10], "deal_stage": d.get("stage"),
        }
        hs.retain_call_note(deal_id=deal_id, content=raw_notes, context=context,
                            timestamp=ts, structured=structured)
        _created["notes"] += 1

    if objection_type and outcome:
        res = learning.record_outcome(d, {
            "id": seed_id(key), "objection_type": objection_type, "outcome": outcome,
            "tactic_used": tactic_used, "stakeholder": stakeholder, "competitor": competitor,
            "reason": reason, "actual_action": actual_action, "result": result,
            "objection_text": objection_text, "date": ts, "source": "seed",
        }, add_timeline_note=False)
        if res.get("duplicate"):
            _skipped["experiences"] += 1
        else:
            _created["experiences"] += 1
    print(f"  {note_type} on {ts[:10]} ({objection_type or 'no objection'})")


def historical(key, customer, stakeholder, objection_type, tactic_used, outcome, reason,
               competitor=None, actual_action=None, result=None, days=30,
               deal_stage="Negotiation", objection_text=None):
    """A recorded outcome from a deal that predates this tool (or happened off
    the timeline) -- exists only in the experience ledger + playbook bank, the
    same way a rep could backfill historical wins/losses. No deal_id, no
    per-deal Hindsight bank; deal=None is a supported, documented path."""
    res = learning.record_outcome(None, {
        "id": seed_id(key), "objection_type": objection_type, "outcome": outcome,
        "tactic_used": tactic_used, "stakeholder": stakeholder, "competitor": competitor,
        "reason": reason, "actual_action": actual_action, "result": result,
        "deal_stage": deal_stage, "objection_text": objection_text,
        "date": days_ago(days), "source": "seed",
    }, customer=customer)
    if res.get("duplicate"):
        _skipped["experiences"] += 1
        print(f"  historical (already present): {customer} / {objection_type} / {outcome}")
    else:
        _created["experiences"] += 1
        print(f"  historical: {customer} / {objection_type} / {outcome} ({tactic_used})")
    return res


def main():
    if "--reset" in sys.argv:
        store.reset_all()
        print("data.json reset.\n")
    print("Seeding Deal Intelligence Agent demo data...\n")

    # ---------------- Deal 1: Meridian Logistics (pricing, competitor) ----------------
    d1 = deal("Meridian Logistics", contact_name="Priya Kapoor",
              contact_title="VP of Operations", deal_value=84000, stage="Negotiation")
    log("meridian-discovery", d1, "call",
        "First discovery call with Priya. They're running route planning on spreadsheets "
        "across 40 depots. Pain point: dispatchers spend 2hrs/day reconciling manual schedules. "
        "Priya seemed cautiously interested but said budget approval goes through their CFO.",
        days_ago(28))
    log("meridian-pricing-roi", d1, "call",
        "Demo call. Priya loved the live route optimization view. She raised a pricing objection: "
        "'This is 3x what we pay for our current tool.' I walked through ROI on driver-hours saved "
        "and offered to run a 2-week pilot on their busiest region instead of full rollout.",
        days_ago(21), objection_type="pricing", outcome="positive",
        tactic_used="Reframe price as ROI on driver-hours saved",
        stakeholder="Priya Kapoor",
        objection_text="This is 3x what we pay for our current tool.",
        actual_action="Quantified driver-hours saved per week and offered a scoped 2-week regional pilot instead of full price commitment",
        result="Priya agreed to the pilot instead of walking away over price.",
        reason="the pilot removed the risk of committing to the full price up front")
    log("meridian-quote-email", d1, "email",
        "Priya emailed that the pilot region cut manual reconciliation time by 60%. She's ready to "
        "move forward but mentioned their CFO, Daniel Osei, is comparing us against Northwind Fleet "
        "Ops, a cheaper competitor. She asked for a formal quote with a multi-year discount.",
        days_ago(10))
    log("meridian-competitor-northwind", d1, "call",
        "Call with Priya and Daniel (CFO). Daniel's objection: Northwind Fleet Ops quoted 20% lower. "
        "I highlighted our real-time GPS integration Northwind doesn't have, and offered a 2-year "
        "contract at 12% discount instead of matching Northwind's price directly. Daniel wants it "
        "in writing before he takes it to their board next Tuesday.",
        days_ago(3), objection_type="competitor", outcome="positive",
        tactic_used="Differentiate on a capability the competitor lacks, discount instead of price-match",
        stakeholder="Daniel Osei", competitor="Northwind Fleet Ops",
        objection_text="Northwind Fleet Ops quoted 20% lower.",
        actual_action="Highlighted real-time GPS integration Northwind doesn't offer, then countered with a 2-year 12% discount instead of matching Northwind's price",
        result="Daniel asked for the offer in writing to bring to the board.",
        reason="he responded to a capability gap more than a price match")

    # ---------------- Deal 2: Brightline Health (timing; trust arrives in the demo arc) ----------------
    d2 = deal("Brightline Health", contact_name="Marcus Webb",
              contact_title="Director of Patient Services", deal_value=52000, stage="Discovery")
    log("brightline-intro", d2, "call",
        "Intro call with Marcus. Brightline runs 6 outpatient clinics and struggles with no-show "
        "rates (~18%). He's interested in automated reminders + rebooking. Mentioned trust as a "
        "concern -- patient data compliance (HIPAA) is a hard requirement, not optional.",
        days_ago(19))  # objection/outcome deliberately NOT tagged yet -- see the demo arc below
    log("brightline-timing", d2, "call",
        "Second call. Marcus is happy with the compliance docs. New objection: timing -- their "
        "fiscal year budget is locked until Q2, so he can't sign anything until then even if he "
        "wants to. He asked if we could start a free trial in the meantime.",
        days_ago(9), objection_type="timing", outcome="neutral",
        tactic_used="Offer a scoped free pilot to bridge a budget freeze",
        stakeholder="Marcus Webb",
        objection_text="Our fiscal year budget is locked until Q2, I can't sign anything until then.",
        actual_action="Offered a 30-day free pilot in one clinic to keep momentum through the budget freeze",
        result="Kept the relationship active but the deal did not move stage-wise until the pilot showed results.",
        reason="the budget freeze was a hard constraint the tactic could not remove, only work around")
    log("brightline-pilot-results", d2, "email",
        "Marcus emailed that the pilot clinic's no-show rate dropped from 18% to 11% in three weeks. "
        "He wants to present results to the CFO next month to unlock early budget. Asked for a "
        "one-pager with the pilot numbers.",
        days_ago(2))

    # ---------------- Deal 3: Coldstream Robotics (the context + contradiction demo) ----------------
    d3 = deal("Coldstream Robotics", contact_name="Elena Ruiz",
              contact_title="Head of Procurement", deal_value=145000, stage="Negotiation")
    log("coldstream-pricing-features", d3, "call",
        "First call with Elena. Coldstream needs inventory forecasting across 3 warehouses. "
        "She's evaluating us against Ionic Sales Cloud's supply chain module. Early pricing "
        "objection: 'Ionic bundles this into a suite we already pay for.'",
        days_ago(25), objection_type="pricing", outcome="negative",
        tactic_used="Direct feature-by-feature comparison",
        stakeholder="Elena Ruiz", competitor="Ionic Sales Cloud",
        objection_text="Ionic bundles this into a suite we already pay for, why would we pay twice.",
        actual_action="Walked through a feature-by-feature comparison against Ionic's module",
        result="Elena's tone stayed flat; she said 'we already pay for a bundle, why pay twice.'",
        reason="she was evaluating total cost, and a feature list doesn't answer a cost question")
    log("coldstream-pricing-tco", d3, "call",
        "Follow-up call. Reframed around total cost of ownership instead of features: showed that "
        "Ionic's 'bundled' module still requires 2 extra staff-hours/week of manual reconciliation "
        "their own team confirmed. Elena's tone shifted -- she asked for a formal TCO comparison doc.",
        days_ago(15), objection_type="pricing", outcome="positive",
        tactic_used="Reframe as total cost of ownership, quantify hidden competitor cost",
        stakeholder="Elena Ruiz", competitor="Ionic Sales Cloud",
        objection_text="Ionic bundles this into a suite we already pay for.",
        actual_action="Quantified Ionic's hidden 2 extra staff-hours/week of manual reconciliation as a real cost",
        result="Elena asked for a formal TCO comparison document to take to leadership.",
        reason="TCO reframing answered the cost question the feature comparison had missed")
    log("coldstream-authority-rfp", d3, "email",
        "Elena forwarded the TCO doc internally and got pushback from her VP about authority to sign "
        "without a formal RFP. She's now asking us to respond to a 12-question RFP by Friday.",
        days_ago(6), objection_type="authority", outcome="positive",
        tactic_used="Co-write the RFP response with the buyer instead of leaving them to draft it alone",
        stakeholder="Elena Ruiz",
        objection_text="My VP says I can't sign without a formal RFP process.",
        actual_action="Offered our solutions engineer to co-write RFP answers directly with Elena",
        result="RFP was submitted on time and leadership was reportedly impressed with the framing.",
        reason="it removed the burden from a buyer who had influence but not final sign-off")
    log("coldstream-final-review", d3, "call",
        "RFP submitted. Elena said leadership is impressed with the TCO framing and it's now down "
        "to us vs. Ionic in a final review. She hinted price will still be the deciding factor and "
        "asked if we have any flexibility on the first-year rate.",
        days_ago(1))

    # ---------------- Historical experiences (fill out every objection type) ----------------
    print("\nAdding historical experiences (pre-dating this tool)...")
    historical("hist-fenwick-prepay", "Fenwick Analytics", "VP Finance", "pricing",
               "Offer an annual pre-pay discount", outcome="negative",
               reason="the buyer wanted the price itself lower, not a payment-timing discount",
               actual_action="Offered 8% off for annual pre-pay",
               result="Buyer churned to a cheaper competitor.", days=55)
    historical("hist-solace-pricematch", "Solace Manufacturing", "COO", "competitor",
               "Match the competitor's price directly", outcome="negative",
               competitor="Ionic Sales Cloud",
               reason="matching price signals the product isn't worth the original price, undermining the pitch",
               actual_action="Matched Ionic's quoted price line-for-line",
               result="Buyer chose Ionic anyway, citing brand trust.", days=70)
    historical("hist-solace-differentiate", "Solace Manufacturing", "COO", "competitor",
               "Differentiate on a capability the competitor lacks, discount instead of price-match",
               outcome="positive", competitor="Northwind Fleet Ops",
               reason="a concrete capability gap was more persuasive than price parity",
               actual_action="Showed live GPS tracking Northwind's product doesn't have",
               result="Buyer signed a 1-year contract.", days=40)
    historical("hist-arden-verbal", "Arden Micro Insurance", "Compliance Officer", "trust",
               "Verbally reassure without documentation", outcome="negative",
               reason="a compliance buyer needed written certification, not a verbal assurance",
               actual_action="Told the buyer our data handling was compliant, no documents sent",
               result="Buyer asked for SOC 2 evidence and went quiet for three weeks after none was sent.",
               days=48)
    historical("hist-rivet-champion", "Rivet & Co", "Head of Security", "authority",
               "Ask the champion to personally sponsor internally", outcome="negative",
               reason="the champion had influence but the actual approver never saw the pitch directly",
               actual_action="Relied on the champion to relay the pitch to their VP",
               result="Deal went quiet after the champion left the company.", days=60)

    # ---- The context + contradiction set, aimed squarely at Coldstream's situation ----
    # Same competitor (Ionic Sales Cloud), same kind of stakeholder (procurement),
    # same stage (Negotiation) => a real context match, from fields the ledger stores.
    print("\nAdding the context-match / mixed-evidence pricing set...")
    historical("hist-halcyon-tco", "Halcyon Freight", "Director of Procurement", "pricing",
               "Reframe as total cost of ownership, quantify hidden competitor cost",
               outcome="positive", competitor="Ionic Sales Cloud", deal_stage="Negotiation",
               objection_text="Ionic is already in our suite, the incremental spend is hard to justify.",
               reason="procurement could defend a number, but not a feature argument",
               actual_action="Built a 3-year TCO model showing Ionic's reconciliation overhead in staff cost",
               result="Procurement recommended us to the steering committee.", days=36)
    # ...and the counter-example, in a deliberately DIFFERENT context, so the same
    # tactic reads as "mixed evidence" overall with a condition worth checking.
    historical("hist-vantage-tco-fail", "Vantage Foods", "VP Marketing", "pricing",
               "Reframe as total cost of ownership, quantify hidden competitor cost",
               outcome="negative", deal_stage="Discovery",
               objection_text="It just costs more than we planned to spend this year.",
               reason="they had no incumbent tool, so there was no hidden competitor cost to quantify and the TCO model looked like spin",
               actual_action="Presented a TCO model against a competitor they were not actually evaluating",
               result="Buyer disengaged and said the numbers felt manufactured.", days=22)
    # A second failure for the pre-pay discount, so it crosses from "use caution"
    # into "avoid" -- two failures, no successes anywhere on record.
    historical("hist-corvus-prepay", "Corvus Retail", "CFO", "pricing",
               "Offer an annual pre-pay discount", outcome="negative", deal_stage="Negotiation",
               objection_text="The annual number is simply above what we budgeted.",
               reason="the objection was to the absolute price, not to the payment schedule",
               actual_action="Offered 10% off for paying twelve months up front",
               result="Buyer said the cash-flow timing was never the issue and stopped responding.",
               days=18)

    # ---------------- Demo arc: generic recommendation -> recorded outcomes -> learned ----------------
    print("\n--- LEARNING DEMO ARC (trust objection) ---")
    already = [e for e in store.list_experiences(objection_type="trust") if e.get("outcome") == "positive"]
    if already:
        print("Trust outcomes are already recorded from a previous run, so the generic -> learned")
        print("flip has already happened. Run `python seed.py --reset` to replay it from scratch.")
    else:
        print("Step 1: asking for a recommendation BEFORE any trust outcomes are recorded...")
    before = learning.recommend(d2, "trust", objection_text="Patient data compliance is a hard requirement",
                                stakeholder="Marcus Webb")
    print(f"  basis={before['recommendation']['basis']}  tactic={before['recommendation']['recommendation']!r}")
    print(f"  evidence: {before['recommendation']['evidence']['summary']}")

    print("\nStep 2: recording trust outcomes (Brightline's own + two more from other deals)...")
    learning.record_outcome(d2, {
        "id": seed_id("brightline-trust-docs"),
        "objection_type": "trust", "outcome": "positive",
        "tactic_used": "Send compliance documentation proactively, before being asked twice",
        "stakeholder": "Marcus Webb",
        "objection_text": "Patient data compliance (HIPAA) is a hard requirement, not optional.",
        "recommended_tactic": before["recommendation"]["recommendation"],
        "actual_action": "Sent SOC 2 + HIPAA compliance documentation the same day he raised it",
        "result": "Marcus said the docs were 'exactly what our security team needed' and booked the next call.",
        "reason": "the buyer's concern was resolved before it could stall the deal",
        "recommendation_id": before["recommendation"]["id"],
        "date": days_ago(18), "source": "seed",
    })
    historical("hist-kestrel-docs", "Kestrel Diagnostics", "Head of IT", "trust",
               "Send compliance documentation proactively, before being asked twice",
               outcome="positive", deal_stage="Discovery",
               objection_text="Our security review needs evidence before we can proceed.",
               reason="removed the trust concern before it could stall momentum",
               actual_action="Sent SOC 2 report unprompted after the first mention of compliance",
               result="Buyer moved to contract review within a week.", days=33)
    historical("hist-arden-docs", "Arden Micro Insurance", "Compliance Officer", "trust",
               "Send compliance documentation proactively, before being asked twice",
               outcome="positive",
               objection_text="We need written certification of how patient and claims data is handled.",
               reason="written proof satisfied the compliance reviewer where a verbal claim had not",
               actual_action="Followed up the earlier failed conversation with a full SOC 2 + HIPAA packet",
               result="Deal restarted after three weeks of silence.", days=41)

    print("\nStep 3: asking again for a recommendation AFTER those outcomes are recorded...")
    after = learning.recommend(d2, "trust", objection_text="Patient data compliance is a hard requirement",
                               stakeholder="Marcus Webb")
    ar = after["recommendation"]
    print(f"  basis={ar['basis']}  tactic={ar['recommendation']!r}  confidence={ar['confidence']}")
    print(f"  evidence: {ar['evidence']['summary']}")
    print(f"  learning: {after['learning']['changes']['summary']}")
    print(f"  journey:  {after['learning']['journey']['summary']}")

    # ---------------- Demo arc 2: context match + mixed evidence (pricing) ----------------
    print("\n--- CONTEXT + CONTRADICTION ARC (pricing objection on Coldstream) ---")
    ctx_rec = learning.recommend(
        d3, "pricing",
        objection_text="Ionic bundles this into a suite we already pay for.",
        stakeholder="Elena Ruiz", competitor="Ionic Sales Cloud", deal_stage="Negotiation")
    r = ctx_rec["recommendation"]
    print(f"  basis={r['basis']}  tactic={r['recommendation']!r}  confidence={r['confidence']}")
    print(f"  general evidence:    {r['evidence']['summary']}")
    print(f"  contextual evidence: {r['context']['summary']}  used={r['context']['used']}")
    print(f"  matched on: {', '.join(r['context']['matched_dimensions']) or '(nothing)'}")
    if r["conflict"]:
        c = r["conflict"]
        print(f"  {c['label']}: '{c['tactic']}' {c['successful']} successful / {c['unsuccessful']} unsuccessful")
        for w in c["worked_when"]:
            print(f"    worked: {w}")
        for f in c["failed_when"]:
            print(f"    failed: {f}")
        print(f"    {c['check_before_using']}")
    for a in r["avoid_detail"]:
        print(f"  {a['severity'].upper()}: '{a['tactic']}' ({a['unsuccessful']} failure(s), no successes)")

    print(f"\nCreated: {_created}    Already present (skipped): {_skipped}")
    print("Re-running this script is safe -- it creates no duplicate deals, notes or ledger rows.")
    print("\nTry: GET /api/deals, POST /api/deals/<id>/recommend,")
    print("     GET /api/recommendations?objection_type=trust,")
    print("     GET /api/learning/journey?objection_type=trust")


if __name__ == "__main__":
    main()
