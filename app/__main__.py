import argparse

from langchain.messages import ToolMessage

from app.agents.supervisor import supervisor


SPECIALIST_NAMES = {
    "ask_fruit_specialist": "fruit",
    "ask_weather_specialist": "weather",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local multi-agent supervisor.")
    parser.add_argument("prompt", help="Prompt to send to the supervisor")
    args = parser.parse_args()

    result = supervisor.invoke(
        {"messages": [{"role": "user", "content": args.prompt}]}
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
