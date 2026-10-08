"""Command-line entry point: `ranklab <command>`."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

import httpx

from .config import env_list, load_dotenv
from .generate import generate_answers
from .judge import judge_pairwise, judge_rubric
from .prompts import default_prompts_path, load_prompts
from .providers import ModelSpec, ProviderError, list_models, missing_keys
from .report import markdown_report, preference_pairs, summarize
from .store import RunStore

DEMO_MODELS = ["mock:strong", "mock:medium", "mock:weak"]


def _specs(names: list[str]) -> list[ModelSpec]:
    specs = [ModelSpec.parse(name) for name in names]
    if len({s.id for s in specs}) != len(specs):
        raise SystemExit("error: a model is listed twice")
    missing = missing_keys(specs)
    if missing:
        raise SystemExit(f"error: set {', '.join(missing)} in .env (copy .env.example) or in your shell")
    return specs


def cmd_models(args: argparse.Namespace) -> int:
    try:
        names = list_models(args.provider)
    except (ProviderError, httpx.HTTPError) as exc:
        raise SystemExit(f"error: {exc}") from None
    print("\n".join(f"{args.provider}:{name}" for name in names))
    return 0


def cmd_generate(args: argparse.Namespace) -> int:
    names = args.models.split(",") if args.models else env_list("RANKLAB_MODELS")
    if len(names) < 2:
        raise SystemExit("error: compare at least two models: --models a,b or RANKLAB_MODELS in .env")
    specs = _specs(names)
    store = RunStore(args.runs_dir, args.run)
    if store.exists() and args.prompts:
        print(f"note: run {args.run!r} keeps the prompt set it started with; --prompts is ignored")
    prompts = load_prompts(Path(args.prompts) if args.prompts else default_prompts_path())
    store.create_or_update([s.id for s in specs], prompts)
    print(f"Run {args.run!r}: {len(store.prompts())} prompts x {len(specs)} models")
    stats = asyncio.run(generate_answers(store, specs, args.concurrency))
    print(f"Answers: {stats['done']} new, {stats['skipped']} already collected, {stats['failed']} failed")
    if stats["failed"]:
        print("Run the same command again to retry the failures.")
    return 1 if stats["failed"] else 0


def cmd_judge(args: argparse.Namespace) -> int:
    name = args.judge or os.environ.get("RANKLAB_JUDGE_MODEL")
    if not name:
        raise SystemExit("error: choose a judge: --judge MODEL or RANKLAB_JUDGE_MODEL in .env")
    [judge] = _specs([name])
    store = RunStore(args.runs_dir, args.run)
    if judge.id in store.models():
        print(f"warning: {judge.id} is also a contestant, and judges tend to prefer their own answers")
    store.set_judge(judge.id)
    failed = 0
    if args.mode in ("rubric", "both"):
        print(f"Rubric scores from {judge.id}")
        failed += asyncio.run(judge_rubric(store, judge, args.concurrency))["failed"]
    if args.mode in ("pairwise", "both"):
        print(f"Pairwise verdicts from {judge.id} (every pair judged in both orders)")
        failed += asyncio.run(judge_pairwise(store, judge, args.concurrency))["failed"]
    print("Done." if not failed else f"Done with {failed} failures; run the same command again to retry them.")
    return 1 if failed else 0


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from .server import create_app

    app = create_app(RunStore(args.runs_dir, args.run))
    print(f"RankLab, run {args.run!r}: http://{args.host}:{args.port}  (Ctrl+C to stop)")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    store = RunStore(args.runs_dir, args.run)
    text = markdown_report(summarize(store))
    out = Path(args.out) if args.out else store.dir / "report.md"
    out.write_text(text + "\n", encoding="utf-8")
    print(text)
    print(f"\nSaved to {out}")
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    store = RunStore(args.runs_dir, args.run)
    rows = preference_pairs(store, args.source)
    out = Path(args.out) if args.out else store.dir / f"preferences-{args.source}.jsonl"
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    print(f"Wrote {len(rows)} preference pairs to {out}")
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    store = RunStore(args.runs_dir, args.run)
    specs = _specs(DEMO_MODELS)
    store.create_or_update([s.id for s in specs], load_prompts(default_prompts_path()))
    print(f"Demo run {args.run!r}: mock models and a mock judge (synthetic data, no API key needed)")
    asyncio.run(generate_answers(store, specs))
    [judge] = _specs(["mock:judge"])
    store.set_judge(judge.id)
    asyncio.run(judge_rubric(store, judge))
    asyncio.run(judge_pairwise(store, judge))
    print(f"Ready. Vote in the browser with: ranklab serve --run {args.run}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ranklab", description="Rank LLM answers with an AI judge and blind human preference votes."
    )
    parser.add_argument("--runs-dir", default=os.environ.get("RANKLAB_RUNS_DIR", "runs"), help="where runs are stored (default: ./runs)")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("models", help="list the models a provider offers")
    p.add_argument("--provider", default="openai", choices=["openai", "openrouter", "ollama"])
    p.set_defaults(func=cmd_models)

    p = sub.add_parser("generate", help="collect answers from each model for every prompt")
    p.add_argument("--run", required=True, help="run name, e.g. oct-2026")
    p.add_argument("--models", help="comma-separated provider:model list (default: RANKLAB_MODELS)")
    p.add_argument("--prompts", help="JSONL prompt set (default: the built-in 30-prompt set)")
    p.add_argument("--concurrency", type=int, default=4)
    p.set_defaults(func=cmd_generate)

    p = sub.add_parser("judge", help="score answers with an AI judge (rubric + pairwise)")
    p.add_argument("--run", required=True)
    p.add_argument("--judge", help="provider:model (default: RANKLAB_JUDGE_MODEL)")
    p.add_argument("--mode", choices=["rubric", "pairwise", "both"], default="both")
    p.add_argument("--concurrency", type=int, default=4)
    p.set_defaults(func=cmd_judge)

    p = sub.add_parser("serve", help="open the voting app and leaderboard in the browser")
    p.add_argument("--run", required=True)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8800)
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("report", help="write a Markdown report of the rankings and checks")
    p.add_argument("--run", required=True)
    p.add_argument("--out")
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("export", help="export preference pairs (chosen/rejected JSONL)")
    p.add_argument("--run", required=True)
    p.add_argument("--source", choices=["human", "judge", "all"], default="human")
    p.add_argument("--out")
    p.set_defaults(func=cmd_export)

    p = sub.add_parser("demo", help="build a synthetic demo run with mock models")
    p.add_argument("--run", default="demo")
    p.set_defaults(func=cmd_demo)
    return parser


def main(argv: list[str] | None = None) -> int:
    load_dotenv(Path.cwd() / ".env")
    args = build_parser().parse_args(argv)
    try:
        return args.func(args) or 0
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
