# Model strategy

## Local-first baseline

Ollama is the default teaching environment because it makes inference visible,
supports offline exercises, and avoids mandatory API spend.

Suggested roles:

| Role | Local baseline |
|---|---|
| Lightweight demonstrations | Gemma 3 4B |
| Tool use and agents | Qwen3 8B |
| Embeddings | Qwen3 Embedding 0.6B |

Exact model versions are reviewed before each delivery.

## Cloud adapters

OpenAI and Gemini adapters provide:

- a quality benchmark;
- stronger structured outputs and tool calling;
- larger context windows;
- a reliable fallback for the final product demonstration.

## Design rule

Course code depends on an internal interface, not directly on one provider:

```python
model = create_chat_model(provider=settings.provider, model=settings.model)
```

Each exercise states which behaviors are portable and which are provider-specific.

## Classroom policy

- The baseline exercise must run locally.
- The instructor demo must have a tested cloud fallback.
- Model comparisons use the same prompt, context, and evaluation criteria.
- Cost, latency, privacy, and reliability are measured rather than assumed.
