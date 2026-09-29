# Deal Intelligence Agent

**A sales intelligence agent that remembers deal history, learns from
outcomes, and uses that evidence to improve future recommendations.**

Built for **HackWithHyderabad 3.0** around the theme **"AI Agents That
Learn Using Hindsight."**

> **Remember → Understand → Learn → Recommend → Observe → Improve**

------------------------------------------------------------------------

## 1. What is Deal Intelligence Agent?

Sales teams accumulate valuable information during calls, emails,
negotiations, and follow-ups:

-   What objections came up?
-   Which competitor was involved?
-   Who was the stakeholder?
-   What stage was the deal in?
-   Which tactic did the salesperson use?
-   Did that tactic actually work?
-   Did a similar situation succeed somewhere else?
-   What should the salesperson do differently next time?

That information is usually spread across CRM notes and past
conversations. Before an important call, a salesperson may spend time
searching through old records and still miss the most relevant
experience.

**Deal Intelligence Agent turns that history into usable decision
support.**

It uses **Hindsight** as its long-term memory layer and adds an
outcome-aware evidence layer on top. The agent does not simply retrieve
old text. It records what happened after a tactic was used, counts the
evidence, compares similar situations, detects conflicting outcomes, and
uses the resulting evidence to shape the next recommendation.

### The core idea

``` text
Past interactions
      ↓
Hindsight memory
      ↓
Relevant evidence
      ↓
Context matching
      ↓
Recommendation
      ↓
Salesperson takes action
      ↓
Outcome is recorded
      ↓
Hindsight + experience ledger
      ↓
Next recommendation improves
```

The project demonstrates **outcome-aware memory and reasoning, not
reinforcement learning**.

Nothing is trained or fine-tuned. The system improves because new
outcomes become new memories and new evidence, and the next
recommendation is recalculated from that updated history.

------------------------------------------------------------------------

# 2. The problem

A normal sales assistant can summarize a conversation.

A useful deal-intelligence system needs to answer a harder question:

> **"Given what has happened before, what should I do in this situation
> --- and why?"**

Consider a pricing objection.

A generic system might say:

> "Try demonstrating ROI."

Our agent can instead look at recorded experiences and say, in effect:

> "This tactic worked in two comparable procurement negotiations
> involving the same competitor, but failed in a different situation
> where the customer had no incumbent system. The evidence is mixed, so
> verify that the current customer has the same underlying condition
> before using it."

That distinction is the heart of the project.

------------------------------------------------------------------------

# 3. What makes this different?

The project is built around a complete learning loop rather than a
one-way "ask the AI" workflow.

### Traditional memory flow

``` text
Store conversation → Retrieve conversation → Generate answer
```

### Deal Intelligence Agent

``` text
MEMORY
   ↓
EVIDENCE
   ↓
CONTEXT MATCH
   ↓
RECOMMENDATION
   ↓
OUTCOME
   ↓
LEARNING
   ↓
BETTER FUTURE RECOMMENDATION
```

The UI makes this seven-step loop visible.

The important part is that **outcomes are first-class data**. A
successful tactic is useful evidence. A failed tactic is also useful
evidence. A mixed record is not silently averaged away.

------------------------------------------------------------------------

# 4. Key capabilities

## 4.1 Deal memory

Each deal has a dedicated Hindsight memory containing its calls, emails,
objections, competitor mentions, pricing discussions, and recorded
outcomes.

The agent can use that memory to produce a pre-call brief or answer
questions about the deal.

## 4.2 Cross-deal playbook

A shared `playbook` memory contains structured lessons across deals.

This allows the agent to answer questions such as:

-   What has worked against pricing objections?
-   Which competitor tactics have failed?
-   What approaches have helped with trust concerns?
-   Which tactics should we be cautious about?

## 4.3 Outcome-aware recommendations

The recommendation system considers:

-   objection type
-   objection text
-   stakeholder
-   competitor
-   deal stage
-   customer/deal context
-   previous outcomes
-   successful and unsuccessful tactics

Recommendations are therefore based on recorded experience instead of
generic advice alone.

## 4.4 Context-aware evidence

The system does not assume that an experience is relevant simply because
it has the same objection type.

It looks for explicit similarities such as:

  ------------------------------------------------------------------------
  Dimension             Matching rule                               Weight
  --------------------- --------------------- ----------------------------
  Same account          Same customer/company                            3

  Same competitor       Same competitor                                  3

  Same stakeholder      Same stakeholder                                 3

  Comparable role       Shared role word from                            2
                        a fixed vocabulary    

  Same deal stage       Same stage                                       2

  Overlapping concern   At least two                                     1
                        significant shared    
                        concern words         
  ------------------------------------------------------------------------

A contextual match is considered strong enough at a score of **3 or
more**.

Contextual evidence is used to influence the recommendation only when:

1.  At least two relevant contextual experiences exist, and
2.  At least one of them is successful.

The system does not invent similarity using an LLM.

Every match identifies the actual field and values that caused the
match.

## 4.5 Contradiction-aware learning

Real sales data is messy.

The same tactic can work in one situation and fail in another.

Instead of hiding that contradiction, the system exposes it.

A tactic can be classified as:

-   `consistent`
-   `mixed`
-   `failing`
-   `inconclusive`

For mixed evidence, the agent shows:

-   successful examples
-   unsuccessful examples
-   the conditions in which they occurred
-   a caution describing what should be checked before using the tactic

For tactics with failures but no successes:

-   one failure → `use caution`
-   two or more failures → `avoid`

This prevents the system from treating a tactic as universally effective
just because it has appeared frequently.

## 4.6 Evidence-based confidence

The system uses deterministic code to calculate:

-   total relevant experiences
-   successful outcomes
-   unsuccessful outcomes
-   neutral outcomes
-   tactic-level records
-   confidence
-   contextual matches
-   conflicts
-   cautions
-   avoid lists

Confidence can be:

``` text
none
weak
mixed
moderate
strong
```

The numbers shown in the UI are calculated by the application code, not
invented by the LLM.

## 4.7 Learning journey

The project records recommendation history and makes the learning curve
visible.

For example:

``` text
Interaction 1
Generic guidance

        ↓

Interaction 2
Personalized using recorded outcomes

        ↓

Later interaction
Context-matched recommendation

        ↓

Later interaction
Recommendation revised after evidence changed
```

The interaction numbers come from the stored recommendation log.

The system only reports milestones that actually occurred.

## 4.8 "Why this recommendation?"

Every recommendation includes an audit-style explanation containing:

-   relevant experience counts
-   successful / failed / neutral counts
-   contextual matches
-   matching dimensions
-   recommendation basis
-   confidence
-   tactic record
-   conflicting evidence
-   cautions
-   avoid list
-   next action

This makes the recommendation inspectable instead of presenting it as
unexplained AI output.

------------------------------------------------------------------------

# 5. Hindsight usage

Hindsight is a core part of the system, not a decorative integration.

The application uses two main memory banks:

### `deal:<id>`

Stores information belonging to one deal:

-   calls
-   emails
-   objections
-   stakeholder information
-   competitor mentions
-   pricing discussions
-   outcomes

This memory is used for deal-specific recall and pre-call briefs.

### `playbook`

Stores cross-deal lessons and outcome records.

This memory allows the agent to learn from experiences beyond the
current deal.

### How Hindsight is used

  -----------------------------------------------------------------------
  Component                           Responsibility
  ----------------------------------- -----------------------------------
  `Hindsight retain()`                Stores calls, notes, lessons, and
                                      outcomes

  `Hindsight recall()`                Retrieves relevant memories

  `Hindsight reflect()`               Produces natural-language reasoning
                                      from supplied evidence

  Local experience ledger             Stores structured outcome data used
                                      for deterministic evidence
                                      calculations

  Learning engine                     Matches context, counts evidence,
                                      detects contradictions, and
                                      determines confidence
  -----------------------------------------------------------------------

The application deliberately separates **facts/evidence** from **LLM
wording**.

The LLM is not allowed to invent the statistics shown by the
application.

If `reflect()` is unavailable or produces unusable output, the system
falls back to a deterministic explanation based on the same verified
evidence.

------------------------------------------------------------------------

# 6. Where the intelligence comes from

A key design decision is keeping important decisions outside the LLM.

### Application code decides

-   which experiences are relevant
-   which experiences match the current context
-   how many successes/failures exist
-   which tactic is selected
-   confidence
-   contradiction status
-   cautions
-   avoid recommendations
-   interaction counters
-   learning milestones
-   forecast values

### Hindsight provides

-   persistent memory
-   memory retrieval
-   cross-deal knowledge
-   natural-language reflection

### The LLM provides

-   the wording of the reasoning
-   the wording of the next action

The LLM receives verified evidence and is explicitly constrained from
inventing customers, numbers, or outcomes.

------------------------------------------------------------------------

# 7. Structured memory

Outcome records are retained with structured information so that
important entities are explicit.

Example:

``` text
Customer: Brightline Health
Stakeholder: Marcus Webb
Objection: trust — "Patient data compliance is a hard requirement"
Tactic: Send compliance documentation proactively, before being asked twice
Outcome: positive
Reason: The buyer's concern was resolved before it could stall the deal
Date: 2026-09-11
Deal stage: Discovery
```

Outcome records are also tagged with information such as:

``` text
objection:trust
outcome:positive
tactic:<tactic>
```

Raw call notes remain available as natural language, so users are not
forced to fill out a rigid form for every interaction.

Older unstructured memories remain usable.

------------------------------------------------------------------------

# 8. Experience ledger

The application maintains a local structured **experience ledger** in
`backend/data.json`.

This is a human-readable mirror of outcome records retained into
Hindsight.

It is used to calculate the evidence shown in the UI.

This separation gives the system two useful properties:

1.  **Hindsight provides long-term memory and semantic retrieval.**
2.  **The local ledger provides deterministic, inspectable evidence
    calculations.**

The ledger can be accessed through:

``` text
GET /api/experiences
```

------------------------------------------------------------------------

# 9. Demo scenarios

The seed data contains three demo deals:

### Meridian Logistics

Used to demonstrate:

-   deal timeline
-   pricing discussions
-   competitor history
-   call brief
-   cross-deal learning

### Brightline Health

Used to demonstrate:

-   trust/compliance objections
-   generic → learned recommendation improvement
-   outcome recording

### Coldstream Robotics

Used to demonstrate:

-   pricing objections
-   procurement context
-   competitor context
-   negotiation stage
-   contextual matching
-   mixed evidence
-   avoid/caution behavior

The seed data covers five objection types:

``` text
pricing
competitor
trust
timing
authority
```

It also includes historical experiences from other companies to simulate
backfilled CRM history.

------------------------------------------------------------------------

# 10. The main learning demonstration

The project contains a scripted demonstration of actual learning.

## Step 1 --- Before successful trust outcomes

The agent is asked for a trust recommendation.

It has no successful precedent yet, so it gives generic guidance.

``` text
basis = generic
tactic = Provide proof of credibility
```

## Step 2 --- Record outcomes

Successful trust outcomes are added, including compliance-documentation
examples.

## Step 3 --- Ask again

The same recommendation is requested again.

The system now has additional evidence and can move from generic
guidance to learned guidance.

The seed output demonstrates the change from:

``` text
Provide proof of credibility
```

to:

``` text
Send compliance documentation proactively,
before being asked twice
```

with the evidence and confidence updated from the newly recorded
outcomes.

This is the core "learning" moment of the demo.

------------------------------------------------------------------------

# 11. Context + contradiction demonstration

Coldstream Robotics provides the second major demo.

The pricing evidence contains both successful and unsuccessful examples.

The TCO tactic has worked in situations similar to Coldstream, including
a procurement stakeholder, negotiation stage, and the same competitor.

However, it failed in another situation where the customer had no
incumbent system.

The system therefore does not simply say:

> "TCO is the best tactic."

Instead, it can say that the evidence is **mixed**, show where it
worked, show where it failed, and identify the condition that separates
those situations.

This is an important part of the project's design:

> **A failed experience is not discarded. It becomes part of the context
> for the next decision.**

------------------------------------------------------------------------

# 12. Architecture

``` text
┌─────────────────────────────────────────────┐
│                 Frontend                    │
│              frontend/index.html            │
│                                             │
│ Timeline │ Call Brief │ Ask Agent │ Learn  │
└──────────────────────┬──────────────────────┘
                       │ HTTP
                       ▼
┌─────────────────────────────────────────────┐
│              FastAPI Backend                │
│                backend/main.py               │
│                                             │
│  Deals / Notes / Brief / Ask                │
│  Recommendations / Outcomes / Learning      │
└───────────────┬─────────────────┬───────────┘
                │                 │
                ▼                 ▼
┌──────────────────────┐  ┌──────────────────────┐
│ Local Experience     │  │ Hindsight             │
│ Ledger               │  │ Memory                │
│                      │  │                      │
│ data.json             │  │ deal:<id>            │
│ evidence counts       │  │ playbook             │
│ recommendation log   │  │ recall / reflect      │
└──────────────┬───────┘  └──────────────┬───────┘
               │                         │
               └────────────┬────────────┘
                            ▼
                   ┌───────────────────┐
                   │ Learning Engine   │
                   │ learning.py       │
                   │                   │
                   │ context matching  │
                   │ evidence          │
                   │ confidence        │
                   │ conflicts         │
                   │ learning journey  │
                   └───────────────────┘
```

------------------------------------------------------------------------

# 13. Project structure

``` text
deal-intel-agent/
│
├── backend/
│   ├── main.py
│   ├── hindsight_service.py
│   ├── learning.py
│   ├── store.py
│   ├── seed.py
│   ├── test_learning.py
│   └── data.json
│
├── frontend/
│   └── index.html
│
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

### Backend files

  -----------------------------------------------------------------------
  File                                Purpose
  ----------------------------------- -----------------------------------
  `main.py`                           FastAPI application and API
                                      endpoints

  `hindsight_service.py`              Hindsight client,
                                      retain/recall/reflect operations
                                      and structured memory helpers

  `learning.py`                       Evidence engine, context matching,
                                      recommendation logic, contradiction
                                      analysis, confidence, journey and
                                      forecast

  `store.py`                          Local JSON storage for deals,
                                      notes, experience ledger and
                                      recommendation history

  `seed.py`                           Demo data, historical experiences
                                      and learning demonstrations

  `test_learning.py`                  Standard-library test suite for the
                                      learning system

  `data.json`                         Local application data and
                                      structured experience ledger
  -----------------------------------------------------------------------

### Frontend

`frontend/index.html` contains the single-page interface, including:

-   deal selection
-   timeline
-   call brief
-   Ask the Agent
-   playbook insight
-   Learn from outcomes

------------------------------------------------------------------------

# 14. Technology stack

### Backend

-   Python
-   FastAPI
-   Uvicorn
-   Pydantic
-   python-dotenv

### Memory

-   Hindsight Cloud
-   `hindsight-client`

### Frontend

-   HTML
-   CSS
-   JavaScript

### Storage

-   Local JSON store for application metadata and structured experience
    evidence

### Testing

-   Python standard library
-   No pytest required for the learning test suite
-   No network required for automated learning tests

------------------------------------------------------------------------

# 15. Setup

## Prerequisites

You need:

-   Python 3
-   pip
-   a Hindsight Cloud account
-   a Hindsight API key

------------------------------------------------------------------------

## Step 1 --- Clone or download the project

``` bash
git clone <your-repository-url>
cd deal-intel-agent
```

If you downloaded the project as a ZIP, extract it and open the project
folder in VS Code.

------------------------------------------------------------------------

## Step 2 --- Create a virtual environment

Recommended:

``` bash
python3 -m venv venv
source venv/bin/activate
```

On Windows:

``` bash
venv\Scripts\activate
```

------------------------------------------------------------------------

## Step 3 --- Install dependencies

``` bash
python3 -m pip install -r requirements.txt
```

------------------------------------------------------------------------

## Step 4 --- Configure Hindsight

Create your environment file:

``` bash
cp .env.example .env
```

Then add your Hindsight credentials:

``` env
HINDSIGHT_BASE_URL=https://api.hindsight.vectorize.io
HINDSIGHT_API_KEY=your-hindsight-api-key
```

Do **not** commit `.env` to GitHub.

The repository should contain `.env.example`, not your real API key.

------------------------------------------------------------------------

# 16. Seed the demo

Move into the backend directory:

``` bash
cd backend
```

Run:

``` bash
python3 seed.py
```

The seed is **idempotent**.

That means running it again does not create duplicate demo deals, notes,
or experience rows.

A successful seed reports a summary similar to:

``` text
Created: {'deals': 3, 'notes': 11, 'experiences': 16}
Already present (skipped): {'deals': 0, 'notes': 0, 'experiences': 0}
```

The seed also runs the generic → learned demonstration and the
context/contradiction demonstration.

### Reset the demo

If the project supports the reset option and you want to replay the
learning demonstration from a clean local state:

``` bash
python3 seed.py --reset
```

Use this when preparing a fresh live demo, rather than repeatedly adding
data to an existing run.

------------------------------------------------------------------------

# 17. Run the application

From the `backend` directory:

``` bash
python3 -m uvicorn main:app --reload --port 8000
```

Open:

``` text
http://127.0.0.1:8000
```

Keep the terminal running while using the application.

------------------------------------------------------------------------

# 18. Demo walkthrough

The recommended demo takes about two minutes.

## 1. Start with memory

Open **Meridian Logistics**.

Show the timeline and the accumulated call/email history.

Then open the **Call Brief** and demonstrate the difference between:

-   a generic/no-memory brief
-   a Hindsight-powered brief using the deal's actual history

The point to communicate:

> The agent remembers the deal instead of starting from zero.

## 2. Show context + contradiction

Open **Coldstream Robotics**.

Go to **Learn from outcomes**.

Use:

-   Stakeholder: `Elena Ruiz`
-   Competitor: `Ionic Sales Cloud`
-   Stage: `Negotiation`
-   Objection: pricing concern

Then request a recommendation.

Walk through the seven-step pipeline:

``` text
1. Memory
2. Evidence
3. Context Match
4. Recommendation
5. Outcome
6. Learning
7. Forecast
```

The recommendation should show the relevant pricing evidence and explain
why the contextual subset matters.

Then show the mixed-evidence warning.

The key message:

> **The agent does not hide a failure just because a tactic worked
> elsewhere.**

## 3. Record an outcome

In the outcome section, record what happened after using the tactic.

The system stores the outcome in:

-   the local experience ledger
-   the deal's Hindsight memory
-   the shared playbook

Then request another recommendation.

The learning section explains what changed.

## 4. Show the learning curve

Open the learning journey.

Show how recommendation history changes as evidence accumulates.

The interaction numbers are taken from the stored recommendation
history.

## 5. Show generic → learned

Open **Brightline Health** and use the trust/compliance scenario.

Show the initial generic recommendation and the later learned
recommendation.

This gives judges a simple before/after demonstration of the central
idea.

## 6. Finish with the existing features

Briefly show:

-   Timeline
-   Call Brief
-   Ask the Agent
-   Playbook Insight

This demonstrates that the new learning system was added without
removing the original deal-intelligence workflow.

------------------------------------------------------------------------

# 19. API reference

## Get deals

``` http
GET /api/deals
```

## Create a deal

``` http
POST /api/deals
```

## Get a deal

``` http
GET /api/deals/{id}
```

## Update deal stage

``` http
PATCH /api/deals/{id}/stage
```

## Add a note

``` http
POST /api/deals/{id}/notes
```

The note endpoint remains backward compatible and can also accept
structured stakeholder/competitor information.

## Get a call brief

``` http
GET /api/deals/{id}/brief
```

## Ask the agent about a deal

``` http
POST /api/deals/{id}/ask
```

## Cross-deal playbook insight

``` http
GET /api/playbook/insight?objection_type=pricing
```

------------------------------------------------------------------------

## New learning endpoints

### Recommendation

``` http
POST /api/deals/{id}/recommend
```

Example body:

``` json
{
  "objection_type": "pricing",
  "objection_text": "Ionic bundles this into a suite we already pay for.",
  "stakeholder": "Elena Ruiz",
  "competitor": "Ionic Sales Cloud",
  "deal_stage": "Negotiation"
}
```

The response includes:

``` text
memory
recommendation
learning
hindsight
```

The recommendation contains evidence, context, conflict information,
cautions, the recommendation basis, confidence, and next action.

### Record an outcome

``` http
POST /api/deals/{id}/outcomes
```

Example fields:

``` json
{
  "objection_type": "pricing",
  "outcome": "positive",
  "tactic_used": "Reframe as total cost of ownership",
  "objection_text": "The customer thinks the solution is expensive.",
  "stakeholder": "Elena Ruiz",
  "competitor": "Ionic Sales Cloud",
  "deal_stage": "Negotiation",
  "result": "Customer requested a TCO proposal."
}
```

Allowed outcomes:

``` text
positive
negative
neutral
```

### Experience ledger

``` http
GET /api/experiences?objection_type=pricing&deal_id=<id>
```

Returns structured outcome evidence.

### Recommendation history

``` http
GET /api/recommendations?objection_type=trust
```

Returns recommendation history and interaction counters.

### Learning journey

``` http
GET /api/learning/journey?objection_type=trust
```

Returns:

-   interactions
-   distinct evidence states
-   milestones
-   summary

------------------------------------------------------------------------

# 20. Objection types

The current system supports five objection types:

``` text
pricing
competitor
trust
timing
authority
```

Outcome values are:

``` text
positive
negative
neutral
```

For backward compatibility, older note outcomes such as:

``` text
advanced
stalled
lost
```

are still accepted by the legacy notes workflow and mapped
appropriately.

------------------------------------------------------------------------

# 21. Testing

## Automated learning tests

From the backend directory:

``` bash
python3 test_learning.py
```

The suite contains approximately 80 standard-library checks.

It does not require:

-   pytest
-   a Hindsight API key
-   network access

The tests cover:

-   seeding idempotency
-   evidence counting
-   context matching
-   negative context cases
-   role matching
-   contradiction handling
-   outcome → recommendation feedback
-   learning journey
-   recommendation audit information
-   backward compatibility
-   deterministic fallback behavior

The tests use temporary local storage so the real `data.json` is not
modified.

Hindsight is stubbed at the service boundary for the learning tests.

This is intentional: the important evidence calculations must remain
testable without relying on an LLM.

------------------------------------------------------------------------

## Syntax and import checks

``` bash
python3 -m py_compile store.py learning.py hindsight_service.py main.py seed.py
```

Then:

``` bash
python3 -c "import main, learning, store, seed; print('ok')"
```

------------------------------------------------------------------------

## Manual recommendation test

``` bash
curl -s -X POST localhost:8000/api/deals/<coldstream-id>/recommend \
  -H 'Content-Type: application/json' \
  -d '{
    "objection_type":"pricing",
    "competitor":"Ionic Sales Cloud",
    "stakeholder":"Elena Ruiz",
    "deal_stage":"Negotiation",
    "objection_text":"Ionic bundles this into a suite we already pay for."
  }'
```

Then record an outcome and request the recommendation again.

The evidence count should increase because the new experience has been
added.

------------------------------------------------------------------------

# 22. Design principles

### 1. Memory should be useful, not decorative

Hindsight is used throughout the deal lifecycle rather than only for a
single "memory" screen.

### 2. Outcomes matter

A recommendation without feedback is only advice.

The system records what happened and feeds that result into future
decisions.

### 3. Failures are evidence too

The system deliberately keeps unsuccessful outcomes.

### 4. Context matters

A tactic that worked for one customer may fail for another.

The recommender therefore compares explicit deal attributes before
treating an old experience as relevant.

### 5. Numbers should be trustworthy

Counts, confidence, context matches and conflict detection are
calculated by code.

The LLM is not responsible for inventing statistics.

### 6. The system should be explainable

The user can inspect why a recommendation was made and which experiences
influenced it.

### 7. Keep the scope focused

The project focuses on one complete workflow rather than trying to
become an entire CRM.

------------------------------------------------------------------------

# 23. Limitations and scope

This prototype intentionally does **not** include:

-   model fine-tuning
-   reinforcement learning
-   user authentication
-   a complete CRM
-   a large analytics dashboard
-   autonomous sales outreach
-   multiple cooperating agents
-   embedding-based similarity
-   automatic role inference from arbitrary names
-   guaranteed business outcomes

The goal is to demonstrate a focused, inspectable learning loop powered
by persistent memory and recorded outcomes.

------------------------------------------------------------------------

# 24. Hackathon alignment

The project is designed around the hackathon's central idea:

> **AI agents that learn using Hindsight.**

### Hindsight memory

The system uses Hindsight for deal memory, cross-deal playbook
knowledge, structured outcome retention, recall and reflection.

### Learning

The agent improves through recorded outcomes rather than pretending that
every generated answer is automatically correct.

### Visible improvement

The learning journey makes changes across interactions visible.

### Technical depth

The project combines:

-   FastAPI
-   Hindsight
-   structured memory
-   deterministic evidence calculations
-   context matching
-   contradiction analysis
-   recommendation history
-   a browser-based UI

### Real-world relevance

The workflow is based on a practical sales problem: turning accumulated
CRM/deal history into usable decision support before and during customer
conversations.

### Focused scope

The project follows one clear story:

``` text
Remember
   ↓
Find evidence
   ↓
Understand context
   ↓
Recommend
   ↓
Record outcome
   ↓
Learn
   ↓
Improve
```

------------------------------------------------------------------------

# 25. Security

Never commit your real Hindsight API key.

Your local `.env` should be ignored by Git.

Use:

``` text
.env
```

locally and commit only:

``` text
.env.example
```

Before pushing the project to GitHub, verify that no API key, secret,
token, or private credential is present in the repository.

------------------------------------------------------------------------

# 26. Current project status

The core workflow is implemented and demonstrable:

-   [x] Deal memory
-   [x] Hindsight integration
-   [x] Cross-deal playbook
-   [x] Structured outcomes
-   [x] Experience ledger
-   [x] Evidence-based recommendations
-   [x] Context-aware matching
-   [x] Contradiction detection
-   [x] Confidence calculation
-   [x] Avoid/caution guidance
-   [x] Learning journey
-   [x] Recommendation history
-   [x] Forecast of the next evidence state
-   [x] Idempotent demo seeding
-   [x] Automated learning tests
-   [x] Browser-based demo UI

Before final submission, the remaining hackathon deliverables should be
prepared separately:

-   demo video
-   live demo
-   team member submission content
-   final GitHub repository review
-   final verification that secrets are not committed

------------------------------------------------------------------------

# 27. Final takeaway

**Deal Intelligence Agent is not trying to be another chatbot for
sales.**

It is a memory-and-learning layer for the deal process.

It remembers what happened, looks for evidence from previous outcomes,
checks whether those experiences actually resemble the current
situation, makes a recommendation, records what happened next, and uses
that new outcome to improve the next decision.

The most important part is simple:

``` text
The agent does not just remember what people said.

It remembers what happened after they acted.
```

And that is what turns deal history into an evolving playbook.
