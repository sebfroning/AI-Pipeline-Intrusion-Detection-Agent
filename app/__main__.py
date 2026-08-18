import argparse

from langchain.messages import ToolMessage

from app.agents.supervisor import create_supervisor


SPECIALIST_NAMES = {
    "ask_fruit_specialist": "fruit",
    "ask_weather_specialist": "weather",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the local multi-agent supervisor.")
    parser.add_argument(
        "--parallel-specialists",
        action="store_true",
        help="Run multiple selected specialists concurrently.",
    )
    parser.add_argument("prompt", help="Prompt to send to the supervisor")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    supervisor = create_supervisor(
        parallel_specialists=args.parallel_specialists,
    )
    result = supervisor.invoke(
        {"messages": [{"role": "user", "content": args.prompt}]},
        config={"max_concurrency": 2 if args.parallel_specialists else 1},
    )

    used_specialists = []
    for message in result["messages"]:
        if isinstance(message, ToolMessage) and message.name in SPECIALIST_NAMES:
            specialist = SPECIALIST_NAMES[message.name]
            if specialist not in used_specialists:
                used_specialists.append(specialist)

    summary = ", ".join(used_specialists) or "none"
    print(f"Used specialists: {summary}")
    print(result["messages"][-1].text)


if __name__ == "__main__":
    main()
