# 08 - LangGraph agents and self-correction

## Learning promise

Learners can build a stateful tool-using agent with retries, memory, routing, and
explicit stop conditions.

## Deck narrative

Workflow versus agent → state → nodes → conditional edges → tool loop → errors
as observations → retries → termination.

## Notebook lab

Implement an agent that recovers from invalid tickers and unsupported metrics.

## Checkpoint

1. When should an error become a model observation?
2. How do retries become an infinite loop?
3. What state must survive between nodes?
