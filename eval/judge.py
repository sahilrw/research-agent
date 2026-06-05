"""LLM-as-judge for scoring agent reports against the benchmark.

These models are eval-specific (they grade pipeline output) so they live
here, not in src/report.py. JUDGE_SYSTEM also lives here rather than in
src/prompts.py for the same separation-of-concerns reason: nothing in the
research pipeline imports it.

Reuses the forced-tool-use pattern from plan()/extract_claims()/
analyze_gap()/synthesize(): one synthetic tool (submit_verdict) whose
input_schema is the Pydantic model's JSON schema, with tool_choice pinned
so the model is structurally forced to return a valid verdict.
"""

import os

from anthropic import Anthropic
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from src.report import Report, Source

load_dotenv()
client = Anthropic()

# Sonnet, not Haiku: judging is a harder reasoning task than the pipeline
# stages, and consistency matters more than latency at eval time.
JUDGE_MODEL = "claude-sonnet-4-6"
JUDGE_TEMPERATURE = 0.2


class DimensionScore(BaseModel):
    score: int = Field(..., ge=0, le=10)
    justification: str = Field(..., description="One-sentence reason")


class JudgeVerdict(BaseModel):
    faithfulness: DimensionScore = Field(
        ..., description="Are claims grounded in and supported by the cited sources?"
    )
    coverage: DimensionScore = Field(
        ..., description="How many of the benchmark key_claims does the summary address?"
    )
    citation_quality: DimensionScore = Field(
        ..., description="Are citations present, in-range, and supporting the right claims?"
    )
    conciseness: DimensionScore = Field(
        ..., description="Is the answer focused, or padded with off-topic prose?"
    )
    overall: int = Field(
        ...,
        ge=0,
        le=10,
        description="Holistic score as a PLAIN INTEGER 0-10 (e.g. 7), NOT an object. "
        "Not a mechanical average of the dimensions.",
    )
    key_claims_hit: list[str] = Field(
        ..., description="Benchmark key_claims the summary actually addressed"
    )
    key_claims_missed: list[str] = Field(
        ..., description="Benchmark key_claims missing or only superficially mentioned"
    )
    notable_failures: list[str] = Field(
        default_factory=list,
        description="Specific bugs: invented facts, broken citations, off-topic detours",
    )


JUDGE_SYSTEM = """
You are a strict, impartial research evaluator. You grade a research agent's answer against a curated benchmark entry. You are skeptical by default and you do not give the benefit of the doubt. Fluent, confident prose does NOT earn a high score on its own — only grounded, complete, well-cited content does.

You will receive:
- <question>: the original research question.
- <reference_answer>: a curator's short model answer (what "correct" looks like).
- <key_claims>: an enumerated list of atomic facts a correct answer MUST contain.
- <agent_summary>: the agent's answer, with inline [N] citation markers.
- <agent_sources>: an indexed directory of the sources the agent had, each with a credibility score. Citation markers [N] in the summary refer to these indices.

You must return a JudgeVerdict via the submit_verdict tool. Score four dimensions 0-10, plus an overall 0-10, plus the explicit key-claim partition.

## Dimension rubrics — calibrate strictly

Do NOT cluster scores around 7. Use the full range. A generic, partially-correct answer is a 5, not a 7.

### faithfulness — are the agent's statements actually supported by its cited sources?
- 9-10: Every substantive statement traces to a cited source that plausibly supports it. No invented facts. No claim asserted beyond what a source could say.
- 5-6: Mostly grounded, but at least one statement is unsupported by any cited source, or stretches a source beyond what it plausibly says.
- 1-2: Multiple fabricated or unsupported assertions; the answer leans on outside knowledge the sources don't back.
- A claim that is plausibly true in the real world but unsupported by any cited source is a FAITHFULNESS FAILURE, not a freebie. Penalize it.

### coverage — does the answer contain the benchmark key_claims?
- 9-10: Addresses all or nearly all key_claims substantively.
- 5-6: Addresses roughly half, or addresses most only superficially.
- 1-2: Addresses one or none.
- This score must be consistent with key_claims_hit / key_claims_missed. If 4 of 5 are hit, coverage is ~8; if 1 of 5, coverage is ~2.

### citation_quality — are citations present, valid, and correctly attributed?
- 9-10: Essentially every factual sentence has a [N] marker; all indices are in range [0, N-1]; cited sources plausibly support the statements.
- 5-6: Citations present but uneven — some factual sentences uncited, OR some citations point to sources that don't clearly support the claim.
- 1-2: Few citations, or citations reference out-of-range / invented indices, or markers are decorative and don't match content.
- An [N] that is out of range, or that points to a source that clearly does not support the sentence, is a BROKEN CITATION. List it in notable_failures.

### conciseness — is it focused on the question?
- 9-10: Tight and on-topic; every section earns its place.
- 5-6: Generally relevant but padded with tangential background or repetition.
- 1-2: Substantially off-topic, bloated, or evasive.

## Explicit key-claim check — do this literally

For EACH item in <key_claims>, decide explicitly whether the agent's summary addresses it. Put the claim text verbatim into EITHER key_claims_hit OR key_claims_missed. Every key_claim must appear in exactly one list. Do not be lenient: a key_claim that is only superficially gestured at, or mentioned without the specific fact it asserts, is a MISS. Partial coverage is a miss.

## notable_failures — flag specific bugs

Populate with concrete, specific problems, e.g.:
- "Invented fact: claims Redlock uses Raft consensus; no source says this."
- "Broken citation: [31] is out of range (only 30 sources)."
- "Citation mismatch: [4] is cited for the quorum formula but source 4 is about GC pauses."
- "Off-topic detour: three paragraphs on Redis persistence, irrelevant to the question."
If there are none, return an empty list.

## overall
A holistic 0-10 reflecting whether a careful engineer would trust and use this answer. It is NOT a mechanical average — a high faithfulness/coverage answer with a fabricated fact should be dragged down, because trust is the point. Briefly ensure your overall is consistent with the four dimensions; if it diverges sharply, your dimension scores or overall are wrong.
"""


def format_sources_xml(sources: list[Source]) -> str:
    """Render sources as an indexed XML directory for the judge.

    Mirrors synthesize()'s source formatting so the judge sees the same
    index<->source mapping the synthesizer cited against, which is what
    makes citation-quality grading meaningful.
    """
    blocks = []
    for idx, src in enumerate(sources):
        blocks.append(
            f'<source index="{idx}" credibility="{src.credibility or 0.0}">\n'
            f"<title>{src.title}</title>\n"
            f"<url>{src.url}</url>\n"
            f"<snippet>{src.snippet or ''}</snippet>\n"
            f"</source>"
        )
    return "\n".join(blocks)


def judge(question_entry: dict, report: Report) -> JudgeVerdict:
    """Score one agent Report against one benchmark question entry.

    question_entry needs at least: question, reference_answer, key_claims.
    """
    key_claims_block = "\n".join(f"- {kc}" for kc in question_entry["key_claims"])

    user_content = (
        f"<question>{question_entry['question']}</question>\n\n"
        f"<reference_answer>{question_entry['reference_answer']}</reference_answer>\n\n"
        f"<key_claims>\n{key_claims_block}\n</key_claims>\n\n"
        f"<agent_summary>\n{report.summary}\n</agent_summary>\n\n"
        f"<agent_sources>\n{format_sources_xml(report.sources)}\n</agent_sources>"
    )

    response = client.messages.create(
        model=JUDGE_MODEL,
        max_tokens=2000,
        temperature=JUDGE_TEMPERATURE,
        system=JUDGE_SYSTEM,
        messages=[{"role": "user", "content": user_content}],
        tools=[
            {
                "name": "submit_verdict",
                "description": "Submit the structured evaluation verdict for the agent's answer.",
                "input_schema": JudgeVerdict.model_json_schema(),
            }
        ],
        tool_choice={"type": "tool", "name": "submit_verdict"},
    )

    tool_block = next(b for b in response.content if b.type == "tool_use")
    return JudgeVerdict.model_validate(tool_block.input)
