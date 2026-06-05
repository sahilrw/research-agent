"""Run the agent against the benchmark and score with an LLM judge.

Compares (Phase 8b = full agent only; baselines/ablations land in 8c-d):
- full agent (deepening + credibility)
- baseline: no-search single-shot LLM          [8c]
- baseline: single-shot RAG (one search, one synth)  [8c]
- ablation: agent without credibility weighting [8d]
- ablation: agent without iterative deepening   [8d]

Outputs per-question scores (faithfulness, coverage, citation-quality)
and aggregate tables.

Run from project root:  python -m eval.run_eval
"""

import json
import os
import traceback

from src.agent import run, client, synthesize, tavily_search, Plan, research_loop
from src.report import Report, Synthesis, Source, Claim, SubQuestion
from src.prompts import SYNTHESIZER_SYSTEM
from eval.judge import judge, JudgeVerdict

BENCHMARK_PATH = "data/benchmark/questions.json"
RESULTS_OUTPUT_PATH = "eval/results.json"
EVAL_RUNS_DIR = "eval_runs"

B0_SYSTEM = (
    "You are a highly capable technical research assistant. Answer the provided research "
    "question comprehensively using only your internal training knowledge. You have no "
    "access to web search or external sources for this run. Do not invent or hallucinate "
    "fake citations, URLs, or source references. Be perfectly honest and precise about "
    "what you know and where your knowledge limits are reached."
)

def baseline_no_search(query: str) -> Report:
    """Evaluates raw parametric memory. One direct call forcing the exact 
    Synthesis tool schema payload structure with zero external source context.
    """
    print(f"BASELINE - Executing Zero-Search Parametric Memory...")


    response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=4000, 
        temperature=0.5,
        cache_control={"type": "ephemeral"},
        system=B0_SYSTEM,
        messages=[{"role": "user", "content": f"Please answer the following technical query:\n{query}"}]
    )
    summary_text = response.content[0].text

    return Report(
        query=query,
        summary=summary_text,
        claims=[],
        sources=[],
        sub_questions=[SubQuestion(id=1, question=query, depth=0, confidence=1.0)],
        contradictions=[],
        confidence_assessment="Answering purely from internal parametric weights without real-time validation or web search grounding."
    )

def baseline_simple_rag(query: str) -> Report:
    """Standard Single-Shot RAG execution. Gathers up to 10 context snippets 
    on the root query, bypasses granular claim parsing or credibility weights,
    and runs a single final synthesis call.
    """
    print(f"BASELINE - Executing Single-Shot RAG (Top 10 Sources)...")

    raw_results = tavily_search(query)
    sliced_results = raw_results[:10] if raw_results else []

    sources: list[Source] = []
    claims: list[Claim] = []

    for idx, src in enumerate(sliced_results):
        src.credibility = 1.0
        src.credibility_components = {"base_weight": 1.0}  # Must conform to dict[str, float]
        sources.append(src)

        claims.append(Claim(
            text=f"Raw Data Snippet: {src.snippet}",
            source_indices=[idx],
            sub_question_id=1
        ))

    sub_questions = [SubQuestion(id=1, question=query, depth=0, confidence=1.0)]

    synthesis = synthesize(query, sources, claims, sub_questions)

    return Report(
        query=query,
        summary=synthesis.summary,
        claims=claims,
        sources=sources,
        sub_questions=sub_questions,
        contradictions=synthesis.contradictions,
        confidence_assessment=synthesis.confidence_assessment
    )

def ablation_no_credibility(query: str) -> Report:
    """Runs the full agent pipeline but intercepts the output, forcing every single
    discovered source to have an identical flat credibility weight of 1.0.
    """
    print(f"ABLATION - Running Full Agent but Stripping Credibility Weights...")
    report = run(query)
    
    flat_sources = []
    for src in report.sources:
        src.credibility = 1.0
        src.credibility_components = {"base_weight": 1.0}
        flat_sources.append(src)
        
    report.sources = flat_sources
    return report

def ablation_no_deepening(query: str) -> Report:
    """Invokes research_loop directly with max_depth=0 to cut off branching growth."""
    print(f"ABLATION - Running Agent with Iterative Deepening Disabled (Max Depth = 0)...")
    master_plan, all_sources, all_claims = research_loop(query, max_depth=0)
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

SYSTEMS = {
    "full_agent": run,
    "baseline_no_search": baseline_no_search,
    "baseline_simple_rag": baseline_simple_rag,
    "ablation_no_credibility": ablation_no_credibility,
    "ablation_no_deepening": ablation_no_deepening,
}


def is_verifiable(q: dict) -> bool:
    """True if the entry's gold answer is trusted (no [VERIFY] tags).

    Judging against an unverified answer key produces garbage scores, so
    these entries are skipped until the curator confirms the facts.
    """

    return not any("[VERIFY]" in kc for kc in q.get('key_claims', []))


def run_one(q: dict, system_name: str, system_fn) -> dict:
    """Run one (question, system) pair through agent + judge -> one record.

    Returns a flat dict row so aggregation/serialization is easy later.
    system_fn is the callable from SYSTEMS (e.g. run); it takes the question
    string and returns a Report.
    """
    q_id = q["id"]
    variant_dir = os.path.join(EVAL_RUNS_DIR, system_name)
    os.makedirs(variant_dir, exist_ok=True)

    report_cache_path = os.path.join(variant_dir, f"{q_id}.report.json")
    verdict_cache_path = os.path.join(variant_dir, f"{q_id}.verdict.json")

    if os.path.exists(report_cache_path):
        print(f"Cache Hit: Loading Report artifact for {system_name}/{q_id} from disk.")
        with open(report_cache_path, "r", encoding="utf-8") as f:
            report = Report.model_validate_json(f.read())
    else:
        print(f"Cache Miss: Invoking system pipeline variant: '{system_name}'...")
        report = system_fn(q["question"])
        with open(report_cache_path, "w", encoding="utf-8") as f:
            f.write(report.model_dump_json(indent=2))

    if os.path.exists(verdict_cache_path):
        print(f"Cache Hit: Loading Judge Verdict for {system_name}/{q_id} from disk.")
        with open(verdict_cache_path, "r", encoding="utf-8") as f:
            verdict = JudgeVerdict.model_validate_json(f.read())
    else:
        print(f"Cache Miss: Dispatching Report to Judge...")
        verdict = judge(q, report)
        with open(verdict_cache_path, "w", encoding="utf-8") as f:
            f.write(verdict.model_dump_json(indent=2))

    total_claims = len(q.get("key_claims", []))
    claims_hit_count = len(verdict.key_claims_hit)

    return {
        "id": q["id"],
        "category": q["category"],
        "system": system_name,
        "overall": verdict.overall,
        "faithfulness": verdict.faithfulness.score,
        "coverage": verdict.coverage.score,
        "citation_quality": verdict.citation_quality.score,
        "conciseness": verdict.conciseness.score,
        "claims_hit": claims_hit_count,
        "claims_total": total_claims,
    }


def aggregate(records: list[dict]) -> None:
    """Print a per-question table of results, followed by a comparative 
    per-system summary matrix highlighting performance differences.
    """
    if not records:
        print("No records to report.")
        return

    print(f"\n{'id':<6}{'category':<16}{'system':<22}{'overall':<9}{'faith':<7}"
          f"{'cover':<7}{'cite':<7}{'concise':<9}{'claims':<8}")
    print("-" * 97)
    for r in records:
        print(f"{r['id']:<6}{r['category']:<16}{r['system']:<22}{r['overall']:<9}"
              f"{r['faithfulness']:<7}{r['coverage']:<7}{r['citation_quality']:<7}"
              f"{r['conciseness']:<9}{r['claims_hit']}/{r['claims_total']:<8}")
        
    print("\n" + "="*97)
    print("ARCHITECTURAL AGGREGATE PERFORMANCE COMPARISON")
    print("="*97)
    print(f"{'System Variant':<22}{'Avg Overall':<13}{'Avg Faith':<11}{'Avg Cover':<11}{'Avg Cite':<11}{'Avg Concise':<13}{'Claims Hit %':<12}")
    print("-" * 97)

    system_groups = {}
    for r in records:
        system_groups.setdefault(r["system"], []).append(r)

    for sys_name, rows in system_groups.items():
        n = len(rows)
        avg_overall = sum(row["overall"] for row in rows) / n
        avg_faith = sum(row["faithfulness"] for row in rows) / n
        avg_cover = sum(row["coverage"] for row in rows) / n
        avg_cite = sum(row["citation_quality"] for row in rows) / n
        avg_concise = sum(row["conciseness"] for row in rows) / n
        
        total_possible = sum(row["claims_total"] for row in rows)
        total_hit = sum(row["claims_hit"] for row in rows)
        hit_pct = (total_hit / total_possible * 100.0) if total_possible > 0 else 0.0

        print(f"{sys_name:<22}{avg_overall:<13.2f}{avg_faith:<11.2f}{avg_cover:<11.2f}"
              f"{avg_cite:<11.2f}{avg_concise:<13.2f}{hit_pct:<12.1f}%")
    print("="*97 + "\n")


def main(limit: int | None = None) -> None:
    """Load benchmark, run each verifiable question through each system, score.

    limit: cap the number of questions (use limit=1 to smoke-test the anchor).
    """
    if not os.path.exists(BENCHMARK_PATH):
        print(f"Error: Benchmark dataset missing at path: {BENCHMARK_PATH}")
        return
    
    print(f"Loading benchmark questions from: {BENCHMARK_PATH}")
    with open(BENCHMARK_PATH, encoding="utf-8") as f:
        benchmark = json.load(f)

    verifiable_questions = [q for q in benchmark if is_verifiable(q)]
    skipped_count = len(benchmark) - len(verifiable_questions)

    print(f"Filtering Status: Kept {len(verifiable_questions)} verified entries, "
          f"skipped {skipped_count} unverified entries containing '[VERIFY]' tokens.")

    if limit is not None:
        print(f"Slicing evaluation queue down to the first {limit} entry/entries.")
        verifiable_questions = verifiable_questions[:limit]

    records = []
    for q in verifiable_questions:
        print(f"\n" + "="*80)
        print(f"EVALUATING QUESTION ID: {q['id']} [{q['category']}]")
        print(f"Question: '{q['question']}'")
        print("="*80)

        for name, fn in SYSTEMS.items():
            try:
                record = run_one(q, name, fn)
                records.append(record)
            except Exception as e:
                print(f"Critical failure encountered executing Question ID {q['id']} on system '{name}': {e}")
                traceback.print_exc()
                continue
    #   5. aggregate(records)
    print("\n" + "="*80)
    print("EVALUATION RUN COMPLETE — SUMMARY MATRIX")
    print("="*80)
    aggregate(records)

    #   6. Save records to eval/results.json (json.dump, indent=2).
    output_dir = os.path.dirname(RESULTS_OUTPUT_PATH)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
    print(f"\nSerializing complete dataset log metrics directly to: {RESULTS_OUTPUT_PATH}")
    with open(RESULTS_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)


if __name__ == "__main__":
    # Smoke-test the anchor first: q004 should score ~7 like the earlier run.
    # Once that looks right, drop the limit to run all verifiable questions.
    main()
