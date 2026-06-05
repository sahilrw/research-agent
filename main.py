# main.py
import os
import re
import argparse
from datetime import datetime, timezone
from rich.console import Console
from rich.panel import Panel
from src.agent import run

console = Console()

def make_slug(query: str) -> str:
    """
    Transforms a raw user string query into a standardized, valid, timestamped filename.
    """
    # 1. Lowercase translation
    cleaned = query.lower()
    # 2. Strip out completely any character that is not alphanumeric, a space, or a hyphen
    cleaned = re.sub(r'[^a-z0-9\s\-]', '', cleaned)
    # 3. Collapse multiple whitespace blocks or continuous spaces/hyphens down to single hyphens
    cleaned = re.sub(r'[\s\-]+', '-', cleaned).strip('-')
    # 4. Truncate clean string cleanly down to ~50 characters avoiding broken words
    cleaned = cleaned[:50].rstrip('-')
    # 5. Extract fixed unique temporal string 
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    
    return f"{timestamp}-{cleaned}.json"

def main():
    parser = argparse.ArgumentParser(
        description="DeepResearch CLI Agent: Autonomous iterative-deepening technical research pipeline."
    )
    parser.add_argument(
        "query", 
        type=str, 
        help="The technical topic, engineering query, or open research question to run."
    )
    parser.add_argument(
        "--max-depth", 
        type=int, 
        default=None, 
        help="Optional upper bound tree depth constraint override (overrides config setting)."
    )
    parser.add_argument(
        "--out", 
        type=str, 
        default=None, 
        help="Optional explicit direct output file destination path. Overrides automatic slug generation."
    )
    
    args = parser.parse_args()

    # Determine absolute target output path configuration
    if args.out:
        out_path = args.out
    else:
        slug_name = make_slug(args.query)
        out_path = os.path.join("reports", slug_name)

    # Defensively verify directory existence for deep file structures
    target_dir = os.path.dirname(out_path)
    if target_dir:
        os.makedirs(target_dir, exist_ok=True)

    console.print(Panel(f"[bold cyan]{args.query}[/bold cyan]\n[dim]Destination: {out_path}[/dim]", title="Starting Deep Research Run"))

    # Execute main execution block
    report = run(args.query, max_depth=args.max_depth)

    # Serialize complete rich validated dataset directly onto disk
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report.model_dump_json(indent=2))

    # Output highly legible summary matrix directly onto stdout console
    console.print("\n")
    console.print(Panel.fit(f"[bold green]✓ Report generated successfully![/bold green]\n[yellow]File Location:[/yellow] {out_path}", title="Execution Complete"))
    
    # Render Preview Window
    preview_cutoff = 1200
    preview_text = report.summary[:preview_cutoff] + "..." if len(report.summary) > preview_cutoff else report.summary
    console.print(Panel(preview_text, title="Synthesis Report Preview (Truncated)"))
    
    # Render Structural Confidence
    console.print(Panel(f"[bold white]{report.confidence_assessment}[/bold white]", title="Research Quality Confidence Assessment"))

if __name__ == "__main__":
    main()