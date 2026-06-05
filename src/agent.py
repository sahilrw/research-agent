"""Orchestrator: drives the iterative-deepening research loop.

Loop: plan -> search -> credibility-score -> read/extract claims ->
gap-check -> (if gaps and depth budget remains) enqueue follow-ups ->
synthesize final report.
"""

import math
import os
from datetime import datetime, timezone, date
from anthropic import Anthropic
from dotenv import load_dotenv
load_dotenv()

from src.report import Plan, Source, Claim, ReaderOutput, SubQuestion, GapAnalysis, Synthesis, Report
from src.prompts import PLANNER_SYSTEM, READER_SYSTEM, GAP_CHECK_SYSTEM, SYNTHESIZER_SYSTEM
from src.config import (
    DOMAIN_TIERS, TIER_SCORES, HALF_LIFE_DAYS, 
    WEIGHTS, DEFAULT_TIER, MISSING_DATE_SCORE,
    MAX_DEPTH, MAX_TOTAL_SUB_QUESTIONS, 
    CONFIDENCE_THRESHOLD,
)

from src.tools import tavily_search

client = Anthropic()

def plan(query: str) -> Plan:
    """
    Analyzes the user query using Claude Haiku 4.5 and forces a structured Plan output.
    """
    response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=1500,
        temperature=0.1,
        cache_control={"type": "ephemeral"},
        system=PLANNER_SYSTEM,
        messages=[
            {"role": "user", "content": f"Decompose this query: {query}"}
        ],
        tools=[
            {
                "name": "submit_research_plan",
                "description": "Submits the structured breakdown, classification, and rationale for a research query.",
                "input_schema": Plan.model_json_schema()
            }
        ],
        tool_choice={"type": "tool", "name": "submit_research_plan"}
    )
    
    # Extract the tool input payload
    tool_input = None
    for content_block in response.content:
        if content_block.type == "tool_use" and content_block.name == "submit_research_plan":
            tool_input = content_block.input
            break
            
    if not tool_input:
        raise ValueError("Model failed to output a structured plan.")
        
    return Plan.model_validate(tool_input)

def score_domain(domain: str) -> tuple[float, int]:
    tier = DOMAIN_TIERS.get(domain.lower(), DEFAULT_TIER)
    score = TIER_SCORES.get(tier, TIER_SCORES[DEFAULT_TIER])
    return score, tier

def score_recency(published: str | None, needs_recency: bool, current_date: date | None = None) -> float:
    if not published:
        return MISSING_DATE_SCORE
    
    try:
        date_str = published.split("T")[0]
        parsed_date = datetime.strptime(date_str, "%Y-%m-%d").date()

        if current_date is None:
            current_date = datetime.now(timezone.utc).date()

        delta_days = (current_date - parsed_date).days

        if delta_days <= 0:
            return 1.0
        
        half_life = HALF_LIFE_DAYS[needs_recency]

        return math.pow(0.5, delta_days/half_life)
    
    except Exception as e:
        print(f"score_recency error for published={published!r}: {e}")
        return MISSING_DATE_SCORE
    

def score_credibility(sources: list[Source], plan_context: Plan, current_date: date | None = None) -> list[Source]:
    w_domain = WEIGHTS["domain"]
    w_recency = WEIGHTS["recency"]

    for source in sources:
        domain_score, tier = score_domain(source.domain)
        recency_score = score_recency(source.published, plan_context.needs_recency, current_date=current_date)

        aggregate_score = (domain_score * w_domain) + (recency_score * w_recency)

        source.credibility = round(aggregate_score, 4)
        source.credibility_components = {
            "domain_score": round(domain_score, 4),
            "recency_score": round(recency_score, 4),
            "tier": float(tier)
        }
    
    return sources

def extract_claims(sub_question: SubQuestion, sources: list[Source]) -> list[Claim]:
    """
    Parses source extracts relative to a specific sub-question, isolates individual 
    atomic assertions via Sonnet, and maps relationships deterministically.
    """
    if not sources:
        return []

    xml_blocks = []
    for i, s in enumerate(sources):
        clean_content = (s.snippet or "").replace("<", "&lt;").replace(">", "&gt;")
        xml_blocks.append(
            f'<source index="{i}">\n'
            f'<title>{s.title}</title>\n'
            f'<url>{s.url}</url>\n'
            f'<content>{clean_content}</content>\n'
            f'</source>'
        )
    xml_sources = "\n".join(xml_blocks)

    user_content = (
        f"<sub_question>{sub_question.question}</sub_question>\n\n"
        f"<sources>\n{xml_sources}\n</sources>"
    )

    try:
        response = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=4000,
            cache_control={"type": "ephemeral"},
            temperature=0.2,
            system=READER_SYSTEM,
            messages=[{"role": "user", "content": user_content}],
            tools=[{
                "name": "submit_claims",
                "description": "Submits a structured array of isolated factual extractions.",
                "input_schema": ReaderOutput.model_json_schema()
            }],
            tool_choice={"type": "tool", "name": "submit_claims"}
        )

        tool_use_block = next(b for b in response.content if b.type == "tool_use")
        tool_input = tool_use_block.input

        output_container = ReaderOutput.model_validate(tool_input)
        extracted_claims = output_container.claims

        for claim in extracted_claims:
            claim.sub_question_id = sub_question.id

        return extracted_claims

    except Exception as e:
        print(f"Claim extraction block failed: {e}")
        return []
    
def analyze_gap(sub_question: SubQuestion, claims: list[Claim]) -> GapAnalysis:
    """
    Critically reviews extracted claims against a targeted sub-question.
    Computes semantic completeness and determines if follow-up research is required.
    """
    formatted_claims = []
    for idx, claim in enumerate(claims, 1):
        citations = ", ".join(str(i) for i in claim.source_indices)
        formatted_claims.append(f"{idx}. [Cites Sources: {citations}] {claim.text}")

    claims_payload = "\n".join(formatted_claims) if formatted_claims else "No claims extracted."

    user_content = (
        f"<sub_question>{sub_question.question}</sub_question>\n\n"
        f"<claims>\n{claims_payload}\n</claims>"
    )

    try:
        response = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=2000,
            temperature=0.2,
            cache_control={"type": "ephemeral"},
            system=GAP_CHECK_SYSTEM,
            messages=[{"role": "user", "content": user_content}],
            tools=[{
                "name": "submit_gap_analysis",
                "description": "Submits critical verification parameters and target follow-ups.",
                "input_schema": GapAnalysis.model_json_schema()
            }],
            tool_choice={"type": "tool", "name": "submit_gap_analysis"}
        )

        tool_use_block = next(b for b in response.content if b.type=="tool_use")
        return GapAnalysis.model_validate(tool_use_block.input)
    
    except Exception as e:
        print(f"Gap Analysis phase execution failure {e}")
        return GapAnalysis(
            confidence=0.0,
            gaps=["System processing error during evaluation loop."],
            needs_deepening=True,
            follow_up_sub_questions=[f"Detailed overview of {sub_question.question}"]
        )
    
def research_loop(engine_query: str, max_depth: int | None = None) -> tuple[Plan, list[Source], list[Claim]]:
    """
    Executes the master iterative-deepening research pipeline.
    Drives state over a dynamic question queue until budget exhaustion or closure.
    """
    if not os.environ.get("TAVILY_API_KEY"):
        print("\nCRITICAL CRASH: TAVILY_API_KEY is missing from the environment state.")
        return Plan(query_type="general", needs_recency=False, rationale="Failed run due to missing keys", sub_questions=[]), [], []

    print(f"Initializing master loop for query: '{engine_query}'")

    effective_max_depth = max_depth if max_depth is not None else MAX_DEPTH

    master_plan = plan(engine_query)
    all_sources: list[Source] = []
    all_claims: list[Claim] = []
    processed_question_set: set[str] = set()

    queue: list[SubQuestion] = []
    for sub_q in master_plan.sub_questions:
        sub_q.depth = 0
        sub_q.parent_id = None
        queue.append(sub_q)

    total_sub_questions_counter = len(queue)
    print(f"Initial plan created with {total_sub_questions_counter} root sub-questions.")

    while queue:
        current_sub_q = queue.pop(0)
        normalized_q = current_sub_q.question.lower().strip()

        if normalized_q in processed_question_set:
            continue
        processed_question_set.add(normalized_q)

        print(f"\n" + "="*80)
        print(f"[PROCESSING SUB-Q #{current_sub_q.id}] Depth: {current_sub_q.depth} | Queue Remaining: {len(queue)}")
        print(f"Question: '{current_sub_q.question}'")
        print("="*80)

        raw_retrieved = tavily_search(current_sub_q.question)
        if not raw_retrieved:
            print(f"Search yeilded zero results for {current_sub_q.question}")
            continue

        scored_sources = score_credibility(raw_retrieved, master_plan)

        local_to_global_map = {}
        for local_idx, src in enumerate(scored_sources):
            existing_matches = [i for i, existing_src in enumerate(all_sources) if existing_src.url == src.url]
            
            if existing_matches:
                local_to_global_map[local_idx] = existing_matches[0]
            else:
                all_sources.append(src)
                local_to_global_map[local_idx] = len(all_sources) - 1

        extracted_claims = extract_claims(current_sub_q, scored_sources)
        print(f"Extracted {len(extracted_claims)} atomic factual claims.")

        for claim in extracted_claims:
            remapped_indices= []
            for l_idx in claim.source_indices:
                if l_idx in local_to_global_map:
                    remapped_indices.append(local_to_global_map[l_idx])
            claim.source_indices = remapped_indices
            all_claims.append(claim)

        gap_report = analyze_gap(current_sub_q, extracted_claims)
        model_verdict = gap_report.needs_deepening                             
        gap_report.needs_deepening = gap_report.confidence < CONFIDENCE_THRESHOLD 
        current_sub_q.confidence = gap_report.confidence
        print(f"Critique Confidence: {gap_report.confidence:.2f} | Needs Deepening: {gap_report.needs_deepening} (model said: {model_verdict})")


        if gap_report.needs_deepening and gap_report.follow_up_sub_questions:
            next_depth_level = current_sub_q.depth + 1

            if next_depth_level > effective_max_depth:
                print(f"Hit effective mac depth cap ({MAX_DEPTH}) for this branch. Halting Expansion.")
                continue

            for follow_up_text in gap_report.follow_up_sub_questions:
                if total_sub_questions_counter >= MAX_TOTAL_SUB_QUESTIONS:
                    print(f"Global budget exhausted ({MAX_TOTAL_SUB_QUESTIONS} sub-question). Dropping further follow-ups.")
                    break

                if follow_up_text.lower().strip() in processed_question_set:
                    continue

                total_sub_questions_counter += 1

                child_sub_q = SubQuestion(
                    id=total_sub_questions_counter,
                    question=follow_up_text,
                    depth=next_depth_level,
                    parent_id=current_sub_q.id
                )

                master_plan.sub_questions.append(child_sub_q)
                queue.append(child_sub_q)
                print(f"Enqueued child follow-up #{child_sub_q.id}: '{child_sub_q.question}'")

    print(f"\nResearch Loop terminated cleanly.")
    print(f"  ├── Total Sub-Questions Explored: {total_sub_questions_counter}")
    print(f"  ├── Total Unique Sources Collected: {len(all_sources)}")
    print(f"  └── Total Atomic Claims Extracted: {len(all_claims)}")
    
    return master_plan, all_sources, all_claims

def synthesize(engine_query:str, sources: list[Source], claims: list[Claim], sub_questions: list[SubQuestion]) -> Synthesis:
    """
    Transforms the unstructured claim matrix and source directory into a 
    fully cited, structurally cohesive final markdown technical report.
    """
    print(f"\n Initializing final report synthesis...")

    sub_q_lines = []
    for sq in sub_questions:
        parent_info = f", parent={sq.parent_id}" if sq.parent_id else ""
        conf_info = f", confidence: {sq.confidence:.2f}" if sq.confidence is not None else ""
        sub_q_lines.append(f"{sq.id} (d={sq.depth}{parent_info}) '{sq.question}'{conf_info}")
    sub_questions_payload = "\n".join(sub_q_lines)

    source_blocks = []
    for idx, src in enumerate(sources):
        tier_val = src.credibility_components.get("tier", "N/A") if src.credibility_components else "N/A"
        source_blocks.append(
            f'<source index="{idx}" credibility="{src.credibility or 0.0}" tier="{tier_val}">\n'
            f'<title>{src.title}</title>\n'
            f'<url>{src.url}</url>\n'
            f'<snippet>{src.snippet or ""}</snippet>\n'
            f'</source>'
        )
    sources_payload = "\n".join(source_blocks)

    claim_lines = []
    for idx, clm in enumerate(claims):
        citations = ", ".join(str(i) for i in clm.source_indices)
        claim_lines.append(f"(Sub-Q {clm.sub_question_id}) (cites sources: {citations}) {clm.text}")
    claims_payload = "\n".join(claim_lines)

    user_content = (
        f"<query>{engine_query}</query>\n\n"
        f"<sub_questions>\n{sub_questions_payload}\n</sub_questions>\n\n"
        f"<sources>\n{sources_payload}\n</sources>\n\n"
        f"<claims>\n{claims_payload}\n</claims>"
    )

    try:
        response = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=4000,
            cache_control={"type": "ephemeral"},
            temperature=0.5,
            system=SYNTHESIZER_SYSTEM,
            messages=[{"role": "user", "content": user_content}],
            tools=[{
                "name": "submit_synthesis",
                "description": "Submits final cited technical summary narrative and validation metadata.",
                "input_schema": Synthesis.model_json_schema()
            }],
            tool_choice={"type": "tool", "name": "submit_synthesis"}
        )

        tool_block = next(b for b in response.content if b.type == "tool_use")
        return Synthesis.model_validate(tool_block.input)
    
    except Exception as e:
        print(f"Error during generation compilation phase: {e}")
        return Synthesis(
            summary=f"# Error\nFailed to compile synthesis narrative smoothly due to engine exception: {e}",
            used_source_indices=[],
            contradictions=["Pipeline exception encountered during extraction."],
            confidence_assessment="None. Processing engine failed."
        )
    

def run(query: str, max_depth: int | None = None) -> Report:
    """
    Main entry point for the entire autonomous research agent pipeline.
    Coordinates initialization, iterative deep-dive loops, and report compilation.
    """
    master_plan, all_sources, all_claims = research_loop(query, max_depth=max_depth)

    synthesis_output = synthesize(query, all_sources, all_claims, master_plan.sub_questions)
    
    return Report(
        query=query,
        summary=synthesis_output.summary,
        claims=all_claims,
        sources=all_sources,
        sub_questions=master_plan.sub_questions,
        contradictions=synthesis_output.contradictions,
        confidence_assessment=synthesis_output.confidence_assessment,
    )