"""
helix/cli/__main__.py

Helix CLI entry point.

Commands:
  helix doctor              Environment + provider health check
  helix run <file>          Execute an agent file
  helix trace <run_id>      View a trace
  helix eval run            Run eval suite
  helix cache stats         Cache statistics
  helix cost --all          Cost report
  helix models              List models with pricing
  helix replay <run_id>     Interactive failure replay
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(prog="helix", description="Helix Agent OS")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("doctor", help="Check environment and provider health")

    run_p = sub.add_parser("run", help="Execute an agent file")
    run_p.add_argument("file", help="Python file to execute")
    run_p.add_argument("--dry-run", action="store_true", help="Estimate cost only")

    trace_p = sub.add_parser("trace", help="View a run trace")
    trace_p.add_argument("run_id", help="Run ID to inspect")
    trace_p.add_argument("--diff", metavar="RUN_ID_B", help="Compare with another run")

    eval_p = sub.add_parser("eval", help="Evaluation commands")
    eval_sub = eval_p.add_subparsers(dest="eval_cmd", required=True)
    eval_sub.add_parser("run", help="Run eval suite")
    eval_cmp = eval_sub.add_parser("compare", help="Compare two eval runs")
    eval_cmp.add_argument("run_a")
    eval_cmp.add_argument("run_b")

    cache_p = sub.add_parser("cache", help="Cache commands")
    cache_sub = cache_p.add_subparsers(dest="cache_cmd", required=True)
    cache_sub.add_parser("stats", help="Show cache statistics")
    cache_sub.add_parser("clear", help="Clear semantic cache")

    cost_p = sub.add_parser("cost", help="Cost report")
    cost_p.add_argument("--all", action="store_true", help="All runs")

    cfg_p = sub.add_parser("config", help="Manage API keys and settings")
    cfg_sub = cfg_p.add_subparsers(dest="config_cmd", required=True)
    cfg_set = cfg_sub.add_parser("set",    help="Save an API key  (e.g. helix config set GOOGLE_API_KEY sk-...)")
    cfg_set.add_argument("key",   help="Environment variable name, e.g. GOOGLE_API_KEY")
    cfg_set.add_argument("value", help="The API key value")
    cfg_sub.add_parser("list",   help="List saved keys (masked)")
    cfg_del = cfg_sub.add_parser("delete", help="Delete a saved key")
    cfg_del.add_argument("key",  help="Key name to delete")
    cfg_sub.add_parser("path",   help="Show config file location")

    sub.add_parser("models", help="List available models with pricing")

    replay_p = sub.add_parser("replay", help="Interactive failure replay")
    replay_p.add_argument("run_id", help="Run ID to replay")

    args = parser.parse_args()
    asyncio.run(_dispatch(args))


async def _dispatch(args: argparse.Namespace) -> None:
    cmd = args.command
    if cmd == "doctor":
        await _cmd_doctor()
    elif cmd == "run":
        await _cmd_run(args)
    elif cmd == "trace":
        await _cmd_trace(args)
    elif cmd == "eval":
        await _cmd_eval(args)
    elif cmd == "cache":
        await _cmd_cache(args)
    elif cmd == "cost":
        await _cmd_cost(args)
    elif cmd == "config":
        await _cmd_config(args)

    elif cmd == "models":
        await _cmd_models()
    elif cmd == "replay":
        await _cmd_replay(args)


async def _cmd_doctor() -> None:
    print("Helix Doctor -- checking environment...\n")
    from helix.runtime.health import environment_doctor
    results = await environment_doctor()

    overall = results["overall"].upper()
    print("Overall: " + overall + "\n")

    # Remind Windows/PowerShell users of correct syntax
    import sys, os
    if sys.platform == "win32" and results["providers_available"] == 0:
        print("  NOTE (Windows/PowerShell): use $env: to set environment variables.")
        print("  Example:  $env:GOOGLE_API_KEY = \"your-key-here\"")
        print("  NOT:      set GOOGLE_API_KEY=\"...\"  (that sets a PS variable, not env)")
        print()

    available = results["providers_available"]
    print(f"LLM Providers ({available} available):")
    for name, info in results["providers"].items():
        status = info["status"]
        desc = info["description"]
        if status == "ready":
            mark = "+"
            extra = "  (via " + info.get("via_key", "") + ")"
        elif "local" in status or "running" in status:
            mark = "~"
            extra = ""
        elif "key set" in status:
            mark = "~"
            extra = "  => " + info.get("fix", "")
        else:
            mark = "-"
            keys = info.get("missing_keys", [])
            fix = info.get("fix", "")
            parts = []
            if keys:
                parts.append("set " + " or ".join(keys))
            if fix and "key set" not in status:
                parts.append(fix)
            extra = ("  => " + " | ".join(parts)) if parts else ""
        print(f"  [{mark}] {name:<14} {status:<26} {desc}{extra}")

    print("\nPackages:")
    for pkg, info in results["packages"].items():
        mark = "+" if info["status"] == "ok" else "-"
        detail = info.get("version", info.get("fix", ""))
        print(f"  [{mark}] {pkg:<28} {detail}")

    if results.get("errors"):
        print("\nErrors:")
        for e in results["errors"]:
            print(f"  [!] {e}")
    if results.get("warnings"):
        print("\nWarnings:")
        for w in results["warnings"]:
            print(f"  [w] {w}")


async def _cmd_run(args: argparse.Namespace) -> None:
    path = Path(args.file)
    if not path.exists():
        print(f"Error: File not found: {args.file}", file=sys.stderr)
        sys.exit(1)
    if args.dry_run:
        print(f"[DRY RUN] Would execute {args.file}")
        return
    print(f"Running {args.file}...")
    import runpy
    runpy.run_path(str(path), run_name="__main__")


async def _cmd_trace(args: argparse.Namespace) -> None:
    trace_dir = Path(".helix/traces")

    if args.diff:
        from helix.observability.ghost_debug import GhostDebugResolver
        resolver = GhostDebugResolver()
        report = await resolver.compare(args.run_id, args.diff)
        if report.identical:
            print("Traces are identical -- no divergence detected.")
        else:
            print(f"Diverged at step {report.diverged_at_step}: {report.diverged_at_span}")
            print(f"  Cause: {report.likely_cause}")
            print(f"  Fix:   {report.recommendation}")
        return

    trace_path = trace_dir / f"{args.run_id}.json"
    if not trace_path.exists():
        print(f"Trace not found: {args.run_id}", file=sys.stderr)
        sys.exit(1)

    trace = json.loads(trace_path.read_text())
    print(f"Run:    {trace.get('run_id')}")
    print(f"Agent:  {trace.get('agent_name')}")
    print(f"Time:   {trace.get('duration_s', 0):.2f}s")
    print(f"Spans:  {trace.get('span_count', 0)}")
    print()
    for i, span in enumerate(trace.get("spans", [])):
        err = " [ERROR]" if span.get("error") else ""
        print(f"  {i:2d}. {span['name']:<40} {span.get('duration_ms', 0):6.0f}ms{err}")


async def _cmd_eval(args: argparse.Namespace) -> None:
    if args.eval_cmd == "run":
        print("Import EvalSuite in your code and call suite.run(agent).")
    elif args.eval_cmd == "compare":
        results_dir = Path(".helix/eval_results")
        for run_id in (args.run_a, args.run_b):
            path = results_dir / f"{run_id}.json"
            if not path.exists():
                print(f"Eval result not found: {run_id}", file=sys.stderr)
                sys.exit(1)
        from helix.config import EvalRunResult
        a = EvalRunResult(**json.loads((results_dir / f"{args.run_a}.json").read_text()))
        b = EvalRunResult(**json.loads((results_dir / f"{args.run_b}.json").read_text()))
        delta = b.pass_rate - a.pass_rate
        print(f"Pass rate: {a.pass_rate:.1%} -> {b.pass_rate:.1%} (delta {delta:+.1%})")
        for case_name in sorted(set(a.scores_by_case) | set(b.scores_by_case)):
            sa = a.scores_by_case.get(case_name, 0)
            sb = b.scores_by_case.get(case_name, 0)
            marker = " v" if sb < sa - 0.05 else (" ^" if sb > sa + 0.05 else "")
            print(f"  {case_name:<40} {sa:.3f} -> {sb:.3f}{marker}")


async def _cmd_cache(args: argparse.Namespace) -> None:
    if args.cache_cmd == "stats":
        print("Cache stats: call agent.cache_controller.stats() in code.")
    elif args.cache_cmd == "clear":
        print("Cache clear: call agent.cache_controller.semantic.clear() in code.")


async def _cmd_cost(args: argparse.Namespace) -> None:
    trace_dir = Path(".helix/traces")
    if not trace_dir.exists():
        print("No traces found. Run an agent first.")
        return

    total_cost = 0.0
    runs = []
    for path in sorted(trace_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text())
            cost = sum(s.get("meta", {}).get("cost_usd", 0) for s in data.get("spans", []))
            runs.append({
                "run_id": data.get("run_id", path.stem),
                "agent": data.get("agent_name", "unknown"),
                "cost": cost,
                "duration_s": data.get("duration_s", 0),
            })
            total_cost += cost
        except Exception:
            continue

    if not runs:
        print("No cost data found.")
        return

    print(f"{'Run ID':<20} {'Agent':<20} {'Cost':>10} {'Duration':>10}")
    print("-" * 65)
    for r in runs[-20:]:
        print(f"{r['run_id'][:20]:<20} {r['agent'][:20]:<20} ${r['cost']:>9.4f} {r['duration_s']:>9.1f}s")
    print("-" * 65)
    print(f"{'TOTAL':<41} ${total_cost:>9.4f}")


async def _cmd_config(args) -> None:
    from helix.config_store import set_key, list_keys, delete_key, config_path

    if args.config_cmd == "set":
        set_key(args.key, args.value)
        # Mask API keys but show model names in full
        sensitive = any(k in args.key for k in ["KEY", "SECRET", "TOKEN", "PASSWORD"])
        display_val = (args.value[:6] + "...") if (sensitive and len(args.value) > 9) else args.value
        print(f"Saved {args.key} = {display_val}")
        print(f"Config file: {config_path()}")
        print()
        if args.key == "HELIX_DEFAULT_MODEL":
            print(f"Default model set to: {args.value}")
            print("Run 'helix models' to see all available models.")
        else:
            print("Run 'helix doctor' to verify the provider is now available.")

    elif args.config_cmd == "list":
        keys = list_keys()
        cfg_file = config_path()
        print(f"Config file: {cfg_file}")
        print()
        if not keys:
            print("No API keys saved or set in environment.")
            print()
            print("Set a key with:")
            print("  helix config set GOOGLE_API_KEY your-key-here")
        else:
            print(f"{'Key':<35} {'Value'}")
            print("-" * 60)
            for k, v in sorted(keys.items()):
                print(f"  {k:<33} {v}")

    elif args.config_cmd == "delete":
        removed = delete_key(args.key)
        if removed:
            print(f"Deleted {args.key}")
        else:
            print(f"{args.key} not found in saved config.")

    elif args.config_cmd == "path":
        print(config_path())


async def _cmd_models() -> None:
    from helix.models.router import MODEL_PRICING, _detect_provider
    from helix.config_store import best_available_model

    best = best_available_model()
    print(f"{'Model':<48} {'Provider':<12} {'Prompt/1K':>10} {'Compl/1K':>10}  Available")
    print("-" * 95)
    for model, pricing in sorted(MODEL_PRICING.items()):
        provider = _detect_provider(model)
        prompt = pricing.get("prompt", 0)
        compl = pricing.get("completion", 0)
        free = " (free)" if prompt == 0 else ""
        current = " <- default" if model == best else ""
        # Check if provider key is set
        import os
        key_map = {
            "openai": ["OPENAI_API_KEY"],
            "anthropic": ["ANTHROPIC_API_KEY"],
            "gemini": ["GOOGLE_API_KEY", "GEMINI_API_KEY"],
            "groq": ["GROQ_API_KEY"],
            "mistral": ["MISTRAL_API_KEY"],
            "cohere": ["COHERE_API_KEY"],
            "together": ["TOGETHER_API_KEY"],
            "openrouter": ["OPENROUTER_API_KEY"],
            "deepseek": ["DEEPSEEK_API_KEY"],
            "xai": ["XAI_API_KEY"],
            "perplexity": ["PERPLEXITY_API_KEY"],
            "fireworks": ["FIREWORKS_API_KEY"],
        }
        keys = key_map.get(provider, [])
        ready = "[+]" if (not keys or any(os.environ.get(k) for k in keys)) else "   "
        print(f"{ready} {model:<46} {provider:<12} ${prompt:>9.5f} ${compl:>9.5f}{free}{current}")
    print()
    print(f"Default model (from available keys): {best}")
    print("Run 'helix doctor' to see which providers are configured.")


async def _cmd_replay(args: argparse.Namespace) -> None:
    from helix.observability.replay import FailureReplay
    try:
        replay = FailureReplay.from_run_id(args.run_id)
    except FileNotFoundError:
        print(f"Trace not found: {args.run_id}", file=sys.stderr)
        sys.exit(1)
    print(replay.summary())


if __name__ == "__main__":
    main()
