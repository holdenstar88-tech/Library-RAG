from __future__ import annotations

import argparse
import json

from app.services.library_rag import get_service


def main() -> None:
    parser = argparse.ArgumentParser(description="Library RAG utilities")
    parser.add_argument("command", choices=["health", "answer", "sync"], help="Command to run")
    parser.add_argument("question", nargs="?", default="", help="Question for answer command")
    args = parser.parse_args()

    service = get_service()
    if args.command == "health":
        print(json.dumps(service.health().model_dump(mode="json"), ensure_ascii=False, indent=2))
    elif args.command == "answer":
        if not args.question:
            raise SystemExit("question is required for answer command")
        print(json.dumps(service.answer(args.question).model_dump(mode="json"), ensure_ascii=False, indent=2))
    elif args.command == "sync":
        result = service.sync()
        print(json.dumps(result.__dict__, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
