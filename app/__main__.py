import argparse
from typing import cast

from langchain.messages import ToolMessage

from app.agents.supervisor import create_supervisor
from app.state import AccessMode, OversightMetadata, TargetKind


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
    parser.add_argument(
        "--target-name",
        default="local model",
        help="Name of the model, agent, or pipeline being overseen.",
    )
    parser.add_argument(
        "--target-kind",
        choices=("model", "agent", "pipeline"),
        default="model",
        help="Kind of system being overseen (default: model).",
    )
    parser.add_argument(
        "--access-mode",
        choices=("black_box", "gray_box", "white_box"),
        default="black_box",
        help="Level of access specialists have to the target (default: black_box).",
    )
    parser.add_argument(
        "--target-description",
        help="Short description of the target and the oversight objective.",
    )
    parser.add_argument(
        "--available-interface",
        action="append",
        default=[],
        help="An interface or artifact specialists may use; repeat as needed.",
    )
    parser.add_argument("prompt", help="Prompt to send to the supervisor")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    supervisor = create_supervisor(
        parallel_specialists=args.parallel_specialists,
    )
    oversight: OversightMetadata = {
        "target_name": args.target_name,
        "target_kind": cast(TargetKind, args.target_kind),
        "access_mode": cast(AccessMode, args.access_mode),
        "available_interfaces": args.available_interface,
    }
    if args.target_description:
        oversight["description"] = args.target_description
    result = supervisor.invoke(
        {
            "messages": [{"role": "user", "content": args.prompt}],
            "oversight": oversight,
        },
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
