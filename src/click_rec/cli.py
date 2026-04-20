import argparse
import asyncio
import sys


def main() -> int:
    parser = argparse.ArgumentParser(prog="click_rec")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("seed", help="Seed catalogue + users")

    consumer = sub.add_parser("consumer", help="Run Kafka enrichment consumer")
    consumer.add_argument("--workers", type=int, default=1)

    replay = sub.add_parser("replay", help="Publish synthetic events to Kafka")
    replay.add_argument("--events", type=int, default=10_000)

    sub.add_parser("replay-dlq", help="Replay DLQ back to main topic")
    sub.add_parser("eval-llm", help="Run LLM-as-judge eval on golden set")
    sub.add_parser("eval-offline", help="Run NDCG@10 / MRR@10 offline eval")

    args = parser.parse_args()

    if args.cmd == "seed":
        from scripts import seed_catalog

        return asyncio.run(seed_catalog.run())
    elif args.cmd == "consumer":
        print(f"TODO: Phase 4a — consumer (workers={args.workers})")
    elif args.cmd == "replay":
        from scripts import replay_clicks

        return asyncio.run(replay_clicks.run(args.events))
    elif args.cmd == "replay-dlq":
        print("TODO: Phase 4b — replay DLQ")
    elif args.cmd == "eval-llm":
        print("TODO: Phase 7 — LLM eval")
    elif args.cmd == "eval-offline":
        print("TODO: Phase 6 — offline eval")
    return 0


if __name__ == "__main__":
    sys.exit(main())
