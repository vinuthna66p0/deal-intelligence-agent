import os
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

import store
import hindsight_service as hs
import learning

app = FastAPI(title="Deal Intelligence Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")


# ---------- Schemas ----------

class NewDeal(BaseModel):
    company: str
    contact_name: str
    contact_title: str
    deal_value: Optional[float] = None
    stage: Optional[str] = "Discovery"


class NewNote(BaseModel):
    note_type: str = "call"          # call | email | meeting
    raw_notes: str
    objection_type: Optional[str] = None   # pricing | competitor | timing | authority | trust | other
    outcome: Optional[str] = None          # advanced | stalled | lost
    tactic_used: Optional[str] = None
    call_date: Optional[str] = None        # ISO8601, defaults to now
    stakeholder: Optional[str] = None
    competitor: Optional[str] = None


class AskQuestion(BaseModel):
    question: str


class StageUpdate(BaseModel):
    stage: str


class RecommendRequest(BaseModel):
    """Step 1 of the learning loop: ask for a recommendation for an objection
    the rep is currently facing on this deal."""
    objection_type: str                    # pricing | competitor | trust | timing | authority
    objection_text: Optional[str] = None
    stakeholder: Optional[str] = None
    competitor: Optional[str] = None
    # Context for matching past experiences to THIS situation. Defaults to the
    # deal's own stage when omitted, so existing callers keep working.
    deal_stage: Optional[str] = None


class OutcomeRecord(BaseModel):
    """Step 2 of the learning loop: what actually happened after the
    recommendation (or after any tactic the rep chose to use)."""
    objection_type: str
    outcome: str                           # positive | negative | neutral (advanced/stalled/lost also accepted)
    tactic_used: str
    objection_text: Optional[str] = None
    stakeholder: Optional[str] = None
    competitor: Optional[str] = None
    recommended_tactic: Optional[str] = None
    actual_action: Optional[str] = None
    result: Optional[str] = None
    reason: Optional[str] = None
    deal_stage: Optional[str] = None
    recommendation_id: Optional[str] = None
    date: Optional[str] = None


# ---------- Deals ----------

@app.get("/api/deals")
def api_list_deals():
    return store.list_deals()


@app.post("/api/deals")
def api_create_deal(deal: NewDeal):
    return store.create_deal(
        company=deal.company,
        contact_name=deal.contact_name,
        contact_title=deal.contact_title,
        deal_value=deal.deal_value,
        stage=deal.stage,
    )


@app.get("/api/deals/{deal_id}")
def api_get_deal(deal_id: str):
    deal = store.get_deal(deal_id)
    if not deal:
        raise HTTPException(404, "Deal not found")
    return {"deal": deal, "notes": store.list_notes(deal_id)}


@app.patch("/api/deals/{deal_id}/stage")
def api_update_stage(deal_id: str, body: StageUpdate):
    if not store.get_deal(deal_id):
        raise HTTPException(404, "Deal not found")
    store.update_deal_stage(deal_id, body.stage)
    return {"ok": True}


# ---------- Notes (this is where memory gets written) ----------

@app.post("/api/deals/{deal_id}/notes")
def api_add_note(deal_id: str, note: NewNote):
    deal = store.get_deal(deal_id)
    if not deal:
        raise HTTPException(404, "Deal not found")

    saved = store.add_note(
        deal_id=deal_id,
        note_type=note.note_type,
        raw_notes=note.raw_notes,
        objection_type=note.objection_type,
        outcome=note.outcome,
        tactic_used=note.tactic_used,
        call_date=note.call_date,
        stakeholder=note.stakeholder,
        competitor=note.competitor,
    )

    context = f"{note.note_type} with {deal['contact_name']} ({deal['contact_title']}) at {deal['company']}"
    structured = {
        "customer": deal["company"],
        "stakeholder": note.stakeholder or deal["contact_name"],
        "objection": note.objection_type,
        "competitor": note.competitor,
        "tactic": note.tactic_used,
        "outcome": note.outcome,
        "date": (note.call_date or saved["timestamp"])[:10],
        "deal_stage": deal.get("stage"),
    }
    hs.retain_call_note(
        deal_id=deal_id,
        content=note.raw_notes,
        context=context,
        timestamp=saved["timestamp"],
        structured=structured,
    )

    # If this note carries a tagged objection + outcome, distill it into the
    # shared cross-deal playbook so every future deal benefits from it.
    if note.objection_type and note.outcome:
        lesson = (
            f"At {deal['company']} ({deal['contact_title']} as champion/contact), "
            f"a '{note.objection_type}' objection came up. "
            f"Tactic used: {note.tactic_used or 'not specified'}. "
            f"Outcome: {note.outcome}."
        )
        hs.retain_playbook_lesson(
            content=lesson,
            context=f"objection:{note.objection_type} outcome:{note.outcome}",
        )

    return saved


# ---------- Outcome-aware learning loop ----------
# MEMORY -> EVIDENCE -> RECOMMENDATION happens in /recommend.
# OUTCOME is written back via /outcomes, which feeds Hindsight AND the local
# experience ledger, so the next /recommend call visibly uses it (LEARNING).

@app.post("/api/deals/{deal_id}/recommend")
def api_recommend(deal_id: str, body: RecommendRequest):
    deal = store.get_deal(deal_id)
    if not deal:
        raise HTTPException(404, "Deal not found")
    if body.objection_type not in learning.OBJECTION_TYPES:
        raise HTTPException(400, f"objection_type must be one of {learning.OBJECTION_TYPES}")
    return learning.recommend(
        deal, body.objection_type, objection_text=body.objection_text,
        stakeholder=body.stakeholder, competitor=body.competitor,
        deal_stage=body.deal_stage,
    )


@app.post("/api/deals/{deal_id}/outcomes")
def api_record_outcome(deal_id: str, body: OutcomeRecord):
    deal = store.get_deal(deal_id)
    if not deal:
        raise HTTPException(404, "Deal not found")
    # learning.record_outcome() already retains a structured outcome record
    # into BOTH the deal bank and the shared playbook bank (see
    # hindsight_service.retain_outcome) -- no separate playbook write needed
    # here, unlike /notes' lighter-weight, unstructured lesson text.
    try:
        result = learning.record_outcome(deal, body.model_dump())
    except ValueError as e:
        raise HTTPException(400, str(e))
    return result


@app.get("/api/experiences")
def api_list_experiences(objection_type: Optional[str] = None, deal_id: Optional[str] = None):
    """The structured experience ledger -- the verifiable, non-hallucinated
    record backing every 'N successful / N unsuccessful' shown in the UI."""
    return store.list_experiences(objection_type=objection_type, deal_id=deal_id)


@app.get("/api/recommendations")
def api_list_recommendations(objection_type: Optional[str] = None):
    """History of past recommendations for a given objection type, so the UI
    can show the before/after learning contrast without recomputing it."""
    return store.list_recommendations(objection_type=objection_type)


@app.get("/api/learning/journey")
def api_learning_journey(objection_type: str):
    """The learning curve for one objection type: how many times it has been
    asked, and the interactions at which the agent moved from generic guidance
    to personalized, then context-matched, recommendations. Every number is
    read back out of the stored recommendation log -- nothing is illustrative."""
    if objection_type not in learning.OBJECTION_TYPES:
        raise HTTPException(400, f"objection_type must be one of {learning.OBJECTION_TYPES}")
    return learning.journey(objection_type)


# ---------- The "wow" endpoints ----------

@app.get("/api/deals/{deal_id}/brief")
def api_call_brief(deal_id: str):
    if not store.get_deal(deal_id):
        raise HTTPException(404, "Deal not found")
    return {
        "with_memory": hs.generate_call_brief(deal_id),
        "without_memory": hs.no_memory_baseline("brief me on this deal"),
    }


@app.post("/api/deals/{deal_id}/ask")
def api_ask(deal_id: str, body: AskQuestion):
    if not store.get_deal(deal_id):
        raise HTTPException(404, "Deal not found")
    return {
        "with_memory": hs.ask_about_deal(deal_id, body.question),
        "without_memory": hs.no_memory_baseline(body.question),
    }


@app.get("/api/playbook/insight")
def api_playbook_insight(objection_type: str):
    return {"insight": hs.playbook_insight(objection_type)}


# ---------- Frontend ----------

@app.get("/")
def index():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))


app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")
