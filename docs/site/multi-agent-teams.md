# Multi-Agent Teams

Teams coordinate multiple agents with three execution strategies.

```python
import helix

searcher = helix.Agent(name="Searcher", role="Web researcher",   goal="Find sources.")
analyst  = helix.Agent(name="Analyst",  role="Data analyst",     goal="Analyze data.")
writer   = helix.Agent(name="Writer",   role="Technical writer", goal="Write reports.")

# sequential: searcher output feeds into analyst, then into writer
team = helix.Team(
    name="research-team",
    agents=[searcher, analyst, writer],
    strategy="sequential",
    budget_usd=5.00,
)

result = team.run_sync("Write a report on renewable energy trends.")
print(result.final_output)
print(f"Total cost: ${result.total_cost_usd:.4f}")
```

**Strategies:**

- `sequential` - each agent receives the previous agent's output as its input
- `parallel` - all agents run on the same input concurrently, outputs returned as a list
- `hierarchical` - a lead agent decomposes the task and delegates subtasks to specialists

```python
lead = helix.Agent(name="Lead", role="Project lead", goal="Decompose and delegate tasks.")

team = helix.Team(
    name="product-team",
    agents=[searcher, analyst, writer],
    strategy="hierarchical",
    lead=lead,
)
```
