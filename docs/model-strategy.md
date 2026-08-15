# Model strategy

## Local-first baseline

Ollama is the default teaching environment because it makes inference visible,
supports offline exercises, and avoids mandatory API spend.

Default local roles:

| Role | Local baseline |
|---|---|
| Chat, structured outputs, and agents | Qwen3 8B |
| Embeddings | Qwen3 Embedding 0.6B |

Exact model versions are reviewed before each delivery.

## OpenAI baseline

The hosted path uses `gpt-5-mini` for the guided notebooks and
`text-embedding-3-small` for embeddings. The instructor can set
`FINAI_CHAT_MODEL=gpt-5.1` for a higher-quality capstone demonstration without
changing notebook code.

OpenAI provides:

- a quality benchmark;
- stronger structured outputs and tool calling;
- larger context windows;
- a reliable fallback for the final product demonstration.

## Design rule

Course code depends on an internal interface, not directly on one provider:

```python
settings = Settings.from_environment()
model = create_chat_model(settings)
```

Each exercise states which behaviors are portable and which are provider-specific.

## Classroom policy

- The baseline exercise must run locally.
- The instructor demo must have a tested cloud fallback.
- Model comparisons use the same prompt, context, and evaluation criteria.
- Cost, latency, privacy, and reliability are measured rather than assumed.

## Environment contract

```text
FINAI_MODEL_PROVIDER=ollama|openai
FINAI_CHAT_MODEL=<optional provider-specific override>
FINAI_EMBEDDING_PROVIDER=ollama|openai
FINAI_EMBEDDING_MODEL=<optional provider-specific override>
FINAI_OLLAMA_BASE_URL=http://localhost:11434
OPENAI_API_KEY=<required only for a live OpenAI run>
```

All LLM-dependent notebooks read this same contract. They do not contain an
Ollama-only or OpenAI-only implementation branch.
