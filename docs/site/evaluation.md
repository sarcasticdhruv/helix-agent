# Evaluation

```python
import asyncio
import helix
from helix.eval.suite import EvalSuite
from helix.config import EvalCase

suite = EvalSuite("qa-suite")
suite.add_cases([
    EvalCase(
        name="capital_cities",
        input="What is the capital of France?",
        expected_facts=["Paris"],
        max_cost_usd=0.05,
    ),
    EvalCase(
        name="math",
        input="What is 15% of 240?",
        expected_facts=["36"],
        max_cost_usd=0.05,
    ),
])

async def main():
    agent = helix.Agent(name="Bot", role="Assistant", goal="Answer questions accurately.")
    results = await suite.run(agent, verbose=True)
    print(f"Pass rate:  {results.pass_rate:.0%}")
    print(f"Total cost: ${results.total_cost_usd:.4f}")
    suite.assert_pass_rate(0.90)   # raises AssertionError if below 90%

asyncio.run(main())
```

The eval suite runs 6 scorers per case: factual accuracy, tool selection, trajectory adherence, cost efficiency, step efficiency, and output quality.

**`@suite.case` decorator:**

```python
from helix.eval.suite import EvalSuite
from helix.config import EvalCase

suite = EvalSuite("my-suite")

@suite.case
def capitals():
    return EvalCase(
        input="What is the capital of Germany?",
        expected_facts=["Berlin"],
        max_cost_usd=0.05,
    )

@suite.case
def arithmetic():
    return EvalCase(
        input="What is 25% of 400?",
        expected_facts=["100"],
    )

# suite now has both cases registered; the function name becomes the case name
```

**EvalCase options:**

| Parameter | Description |
|---|---|
| `input` | Task string sent to the agent |
| `expected_facts` | Strings that must appear in the output |
| `expected_tools` | Tool names the agent is expected to call |
| `expected_trajectory` | `ExpectedTrajectory` for sequence/step constraints |
| `max_steps` | Maximum reasoning steps (default 10) |
| `max_cost_usd` | Cost cap per case (default 1.00) |
| `pass_threshold` | Minimum overall score to pass (default 0.70) |
| `tags` | Labels for filtering subsets |
