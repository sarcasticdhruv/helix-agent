"""
examples/workflow.py

Workflow example — sequential and parallel step execution.
Run with:  python examples/workflow.py
"""

from helix.core.workflow import Workflow, step

# --- Define steps -----------------------------------------------------------


@step(name="gather", retry=1)
async def gather_data(topic: str) -> dict:
    """Simulate gathering data on a topic."""
    return {
        "topic": topic,
        "sources": [f"Source 1 on {topic}", f"Source 2 on {topic}"],
        "raw_data": f"Raw information about {topic} gathered from multiple sources.",
    }


@step(name="analyse")
async def analyse_data(data: dict) -> dict:
    """Simulate analysis."""
    return {
        **data,
        "key_points": [
            f"Key insight 1 about {data['topic']}",
            f"Key insight 2 about {data['topic']}",
        ],
        "sentiment": "positive",
    }


@step(name="format_report")
async def format_report(data: dict) -> str:
    """Format the final report."""
    lines = [
        f"# Report: {data['topic']}",
        "",
        "## Key Points",
    ]
    for point in data.get("key_points", []):
        lines.append(f"- {point}")
    lines += ["", f"Sentiment: {data.get('sentiment', 'neutral')}"]
    return "\n".join(lines)


def main():
    # Build a sequential pipeline
    pipeline = (
        Workflow("research-pipeline")
        .then(gather_data)
        .then(analyse_data)
        .then(format_report)
        .with_budget(1.00)
    )

    result = pipeline.run_sync("Renewable Energy")

    print(result.final_output)
    print(f"\nSteps completed: {len(result.steps)}")
    print(f"Duration:        {result.duration_s:.2f}s")
    if result.error:
        print(f"Error:           {result.error}")


if __name__ == "__main__":
    main()
