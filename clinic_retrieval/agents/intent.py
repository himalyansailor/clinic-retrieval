"""Reading the question, then choosing the branch.

Two steps that used to be one pile of regular expressions. `read` is the small
model's only job on the way in: three labels and, if the question names a
medicine, the name as typed. `plan` is pure and takes the label together with
an already-resolved drug, so the branch decision is a function of two
arguments and touches no data.

The labels are not decoration. `inventory` goes to a structured lookup and
never to retrieval, because whether this clinic has a drug is a number in a
table and no document contains it. `policy` and `clinic` both go to the same hybrid
retrieval, so a confusion between those two costs nothing; a confusion with
`inventory` is the one that changes where the answer comes from.
"""
from __future__ import annotations

from dataclasses import dataclass

from clinic_retrieval.agents.drugs import Resolution
from clinic_retrieval.core import Reason
from clinic_retrieval.llm import Llm

LABELS = ("inventory", "policy", "clinic")

SYSTEM = """You label one question from a rural health clinic in India. The \
question may be in English, Hindi, or Hindi typed in English letters.

Labels:
- "inventory": asks whether a medicine is in stock here, how many are left, or \
whether the clinic has it.
- "policy": asks about a government health scheme, its amount, cover, \
eligibility, or who can claim.
- "clinic": asks what staff should do. A procedure, a protocol, a referral, \
record keeping, cold chain, or what to do when something is unavailable.

A question that asks what to do when a medicine is out of stock is "clinic", \
not "inventory": it asks for the procedure, not for a count.

Also give the medicine name exactly as the question writes it, or null if the \
question names none.

Reply with JSON only, in this shape:
{"intent": "policy", "drug": null}"""


@dataclass(frozen=True)
class Intent:
    label: str
    drug: str = ""


@dataclass(frozen=True)
class Route:
    agents: tuple[str, ...]
    drug: str | None                 # a formulary code, not a stock record
    why: str
    label: str = ""                  # the intent the model read, for the log
    owner: str = "docs"              # the agent whose refusal refuses the answer
    reason: Reason | None = None     # set when the route is escalation or clarify
    resolution: Resolution | None = None


def read(question: str, llm: Llm) -> Intent | None:
    """None when the model is unreachable or answers with something that is not
    one of the three labels. Guessing a label would pick a data source, so an
    unreadable question goes to a person instead."""
    reply = llm.json(SYSTEM, question, max_tokens=120)
    if reply is None:
        return None
    label = str(reply.get("intent") or "").strip().lower()
    if label not in LABELS:
        return None
    drug = reply.get("drug")
    return Intent(label, str(drug).strip() if drug else "")


def plan(intent: Intent, found: Resolution) -> Route:
    """Pure. The drug is already resolved by the time this runs, and whether
    this clinic holds any of it is the stock agent's business."""
    code = found.best.code if found.best else None
    named = f"{found.best.name} via {found.best.how}" if found.best else ""

    if intent.label == "inventory":
        if found.outcome == "resolved":
            return Route(("stock",), code, f"availability of {named}",
                         intent.label, owner="stock", resolution=found)
        if found.outcome == "ambiguous":
            return Route(("clarify",), None,
                         f"availability, {len(found.candidates)} drugs fit the name",
                         intent.label, owner="stock",
                         reason=Reason.AMBIGUOUS_DRUG, resolution=found)
        return Route(("escalation",), None,
                     "availability question, no drug in the formulary fits the name",
                     intent.label, owner="stock",
                     reason=Reason.NOT_IN_FORMULARY, resolution=found)

    if intent.label == "policy":
        # A scheme question that happens to name a medicine is still a scheme
        # question, and the stock table has nothing to add to it.
        return Route(("docs",), None, "scheme question", intent.label,
                     resolution=found)

    if found.outcome == "resolved":
        # A procedure question naming a drug the clinic may not hold: the count
        # changes which half of the procedure matters. The document still owns
        # the answer, so a procedure nobody can find is a refusal even though
        # the stock table had a number to offer.
        return Route(("stock", "docs"), code,
                     f"procedure question naming {named}", intent.label,
                     resolution=found)
    return Route(("docs",), None, "procedure question", intent.label,
                 resolution=found)
