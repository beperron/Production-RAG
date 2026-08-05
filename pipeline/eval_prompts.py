#!/usr/bin/env python3
"""The shared contract both generation arms write against.

Arm A (Claude) and Arm B (GLM 5.2) must receive the SAME instructions, or a
difference in the resulting leaderboard tells you about the prompt rather than
about the model. Only the model behind the prompt varies.
"""
from __future__ import annotations

TYPE_BRIEF = {
    "known_item":
        "A lawyer half-remembers this provision and wants to find it again. "
        "Ask for it the way someone would who knows the substance but not the "
        "number.",
    "procedural":
        "A practical question arising in a live matter -- a deadline, a "
        "required step, who must be served, what a court must do. Phrase it "
        "as someone with a file open in front of them, not as a student.",
    "cross_reference":
        "The question can only be answered by following this provision "
        "through to a rule it references, or by knowing this provision governs "
        "a procedure defined elsewhere.",
    "disambiguation":
        "The subject is addressed in more than one place in the Michigan Court "
        "Rules. Ask in a way that a careless reader would answer with the "
        "WRONG rule, but that this provision actually settles.",
    "unanswerable":
        "Write a question that sounds like Michigan civil procedure and that a "
        "practitioner might plausibly ask, but which the Michigan Court Rules "
        "DO NOT answer -- it is governed by statute, by local administrative "
        "order, or by nothing at all. Do NOT make it answerable by the "
        "provision shown.",
}

SYSTEM = """\
You are helping build a retrieval benchmark over the Michigan Court Rules. It \
will be shown to the Michigan court system, so every item must be defensible.

You write QUESTIONS ONLY. You never write answers, and you never decide \
whether a retrieval is correct -- the citation does that."""


OUT_OF_SCOPE = """\
Things the Michigan Court Rules do NOT govern, for reference:
  - substantive law and elements of claims (that is statute and case law)
  - damages amounts, interest rates, statutory fee schedules
  - the Michigan Rules of Evidence (a separate body of rules)
  - the Michigan Rules of Professional Conduct
  - local administrative orders and individual judges' practice guidelines
  - court staffing, budgets, facilities, and employment matters
  - federal procedure"""


def build_unanswerable_prompt(target):
    """Unanswerable items must NOT be seeded from a provision.

    Shown a provision and asked for something it does not answer, a model
    drifts to the provision's subject matter -- which the rules usually DO
    cover elsewhere. Seeding from out-of-scope domains instead keeps the
    negative genuinely negative.
    """
    return f"""{SYSTEM}

Write ONE question that a Michigan practitioner might plausibly ask, in or
near the subject area of Chapter {target['chapter']}, but which the MICHIGAN
COURT RULES DO NOT ANSWER.

{OUT_OF_SCOPE}

REQUIREMENTS
1. It must sound like a real question someone would ask a law librarian.
2. It must NOT be answerable by any Michigan Court Rule -- not this chapter,
   not any other. If a court rule sets the deadline, the procedure, the form
   or the service requirement you are asking about, the item is unusable.
3. Do not mention that it is unanswerable, and do not name a citation.
4. Prefer questions about substantive entitlement, amounts, evidence
   admissibility, or professional conduct -- these reliably sit outside the
   rules of procedure.

RETURN EXACTLY THIS JSON SHAPE:

{{
  "query": "<the question>",
  "reasoning": "<one sentence: which body of law actually governs this>",
  "also_answered_by": [],
  "confidence": "high" | "medium" | "low"
}}

Output the JSON object and nothing else -- no prose, no code fence."""


def build_prompt(target, siblings):
    """One provision -> one question. `siblings` are the other citations under
    the same rule, so the model can tell what makes this one distinct."""
    t = target
    if t["query_type"] == "unanswerable":
        return build_unanswerable_prompt(t)
    brief = TYPE_BRIEF[t["query_type"]]
    sib = "\n".join(f"  {s}" for s in siblings[:25]) or "  (none)"

    return f"""{SYSTEM}

THE PROVISION
  citation : {t['citation']}
  rule     : {t['rule_title']}
  chapter  : Chapter {t['chapter']}{(' > ' + t['subchapter']) if t.get('subchapter') else ''}

  text:
{t['text']}

OTHER PROVISIONS UNDER THE SAME RULE (so you can tell what is distinctive here)
{sib}

YOUR TASK -- write ONE question of type "{t['query_type']}"
{brief}

HARD REQUIREMENTS

1. DO NOT ECHO THE TEXT. The single most common way to ruin a retrieval
   benchmark is to write a question that reuses the provision's distinctive
   wording. That measures lexical overlap, not retrieval. Use the words a
   practitioner would use, not the words the rule uses. If a distinctive
   phrase appears in the provision, deliberately ask for it another way.

2. DO NOT NAME THE CITATION in the question (except for the citation_lookup
   type, which is generated separately and never reaches you). A question
   containing "MCR {t['rule']}" tests string matching.

3. BE ANSWERABLE FROM THIS PROVISION. A competent reader given only
   {t['citation']} must be able to answer it -- unless the type is
   "unanswerable", where nothing in the Michigan Court Rules should answer it.

4. BE SPECIFIC ENOUGH TO HAVE ONE BEST ANSWER. "What are the discovery rules?"
   is useless. If the subject genuinely appears in several rules, name enough
   circumstance to single this one out.

5. SOUND LIKE A PERSON. Judges, clerks and attorneys ask in plain language,
   sometimes tersely. Vary the register. Not every question needs to be a full
   grammatical sentence.

Also report `also_answered_by`: any OTHER citation you believe would answer
your question equally well. Be honest -- an unmarked co-valid answer scores a
correct retrieval as a failure. Use [] if there is genuinely only one.

RETURN EXACTLY THIS JSON SHAPE, using these key names and no others:

{{
  "query": "<the question>",
  "reasoning": "<one sentence: why {t['citation']} answers it>",
  "also_answered_by": ["<other citation>", "..."],
  "confidence": "high" | "medium" | "low"
}}

Output the JSON object and nothing else -- no prose, no code fence."""


SCHEMA = {
    "type": "object",
    "properties": {
        "query": {"type": "string",
                  "description": "the question, 4-40 words, no citation in it"},
        "reasoning": {"type": "string",
                      "description": "one sentence: why this provision answers it"},
        "also_answered_by": {
            "type": "array",
            "items": {"type": "string"},
            "description": "other citations that would answer equally well",
        },
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
    },
    "required": ["query", "reasoning", "also_answered_by", "confidence"],
}
