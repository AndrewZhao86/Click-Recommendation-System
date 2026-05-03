import argparse
import asyncio
import os
import sys


def _configure_logging() -> None:
    """Configure structlog JSON logging + OTEL tracing for every CLI subcommand.

    Phase 8b: lazy imports keep `python -m click_rec.cli --help` fast —
    we don't want every shell completion to drag in the OTEL SDK.
    Library modules (consumer, replay_dlq, etc.) must NOT call
    `logging.basicConfig` themselves; structlog's `ProcessorFormatter`
    bridge converts every existing `logger.info("event_x", extra=...)`
    call site into a JSON record with trace correlation.
    """
    from click_rec.telemetry.logging import configure_logging
    from click_rec.telemetry.tracing import init_tracing

    configure_logging(os.environ.get("LOG_LEVEL", "INFO"))
    init_tracing()


def main() -> int:
    _configure_logging()
    parser = argparse.ArgumentParser(prog="click_rec")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("seed", help="Seed catalogue + users")

    consumer = sub.add_parser("consumer", help="Run Kafka enrichment consumer")
    consumer.add_argument("--workers", type=int, default=1)

    replay = sub.add_parser("replay", help="Publish synthetic events to Kafka")
    replay.add_argument("--events", type=int, default=10_000)
    replay.add_argument(
        "--capture-eval-log",
        type=str,
        default=None,
        help="Tee click events to this JSONL path for `make eval-offline`",
    )

    sub.add_parser("replay-dlq", help="Replay DLQ back to main topic")

    eval_llm = sub.add_parser(
        "eval-llm", help="Run hybrid-vs-hybrid+LLM rerank eval + LLM-as-judge"
    )
    eval_llm.add_argument("--num-users", type=int, default=200)
    eval_llm.add_argument("--k", type=int, default=10)
    eval_llm.add_argument(
        "--judge-sample",
        type=int,
        default=10,
        help="Number of queries to send to the LLM-as-judge (default: 10).",
    )
    eval_llm.add_argument(
        "--eval-log",
        type=str,
        default=None,
        help="Path to replay-captured JSONL (default: artifacts/eval_clicks.jsonl)",
    )
    eval_llm.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output artifact path (default: artifacts/eval_llm_<ts>.json)",
    )
    eval_offline = sub.add_parser(
        "eval-offline", help="Run NDCG@10 / MRR@10 offline eval"
    )
    eval_offline.add_argument("--num-users", type=int, default=200)
    eval_offline.add_argument("--k", type=int, default=10)
    eval_offline.add_argument(
        "--eval-log",
        type=str,
        default=None,
        help="Path to replay-captured JSONL (default: artifacts/eval_clicks.jsonl)",
    )
    eval_offline.add_argument(
        "--golden",
        type=str,
        default=None,
        help="Path to hand-curated golden queries (default: tests/data/golden_queries.json)",
    )
    eval_offline.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output artifact path (default: artifacts/eval_offline_<ts>.json)",
    )

    args = parser.parse_args()

    if args.cmd == "seed":
        from scripts import seed_catalog

        return asyncio.run(seed_catalog.run())
    elif args.cmd == "consumer":
        from click_rec.kafka.consumer import run_consumer_pool

        return asyncio.run(run_consumer_pool(workers=args.workers))
    elif args.cmd == "replay":
        from scripts import replay_clicks

        return asyncio.run(
            replay_clicks.run(args.events, capture_eval_log=args.capture_eval_log)
        )
    elif args.cmd == "replay-dlq":
        from scripts import replay_dlq

        return asyncio.run(replay_dlq.run())
    elif args.cmd == "eval-llm":
        from click_rec.eval import llm as eval_llm_mod

        return asyncio.run(eval_llm_mod.run_eval_llm_cli(args))
    elif args.cmd == "eval-offline":
        from click_rec.eval import offline

        return asyncio.run(offline.run_offline_eval_cli(args))
    return 0


if __name__ == "__main__":
    sys.exit(main())
