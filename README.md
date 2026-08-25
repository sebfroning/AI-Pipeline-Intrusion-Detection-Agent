# Local supervisor and specialists

A small, local multi-agent example built with LangChain and Ollama. A supervisor
answers general questions itself and delegates fruit or harvest-weather questions
to focused specialist agents when needed. A finalizer then produces one report
from the supervisor synthesis and every structured specialist result.

## Setup

Create and activate a virtual environment, then install the pinned dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Make sure Ollama is running and the default model is available:

```bash
ollama serve
ollama pull qwen3:4b
```

## Run

Pass one prompt to the module:

```bash
python -m app "How could dry weather affect an apple harvest?"
```

The command prints which specialists were used, followed by the finalizer's
report. Each run is stateless. The supervisor may use neither specialist, one
specialist, or both.

## Workflow

The outer LangGraph has a fixed execution path:

```text
START -> supervisor agent/tool loop -> finalizer -> END
```

Specialist tools append typed entries containing the specialist name, focused
question, and finding to `specialist_results` in graph state. The list uses a
reducer so results from parallel tool calls are retained. The finalizer always
runs after a successful supervisor pass, reconciles all collected findings, and
stores its answer in `final_report` as well as the final message.

## Oversight metadata

Every run includes metadata about the model, agent, or pipeline being overseen.
It is stored under the typed `oversight` state field and inherited by each
specialist. The metadata records:

- `target_name` and `target_kind` (`model`, `agent`, or `pipeline`)
- `access_mode` (`black_box`, `gray_box`, or `white_box`)
- an optional `description`
- optional `available_interfaces`, such as an inference API, request logs,
  activations, weights, or training data

The metadata is also added to every agent's system prompt so agents can select
tools that match the access they actually have. Configure it from the CLI:

```bash
python -m app \
  --target-name "harvest classifier" \
  --target-kind model \
  --access-mode white_box \
  --target-description "Classifies fruit harvest risk" \
  --available-interface weights \
  --available-interface activations \
  "Tell me about apples"
```

Code that invokes the graph directly must provide the same state shape:

```python
result = supervisor.invoke({
    "messages": [{"role": "user", "content": "Inspect this model"}],
    "oversight": {
        "target_name": "harvest classifier",
        "target_kind": "model",
        "access_mode": "black_box",
        "available_interfaces": ["inference API"],
    },
})
```

To use another installed Ollama model:

```bash
OLLAMA_MODEL=another-model python -m app "Tell me about pears"
```

To allow both specialists to run concurrently when a prompt needs both:

```bash
python -m app --parallel-specialists \
  "Tell me about pears and give me a harvest weather pattern"
```

The flag allows the supervisor to request both specialists in one response and
sets the LangChain execution concurrency to two. Ollama must also be configured
to process two requests concurrently; when starting the server manually, use:

```bash
OLLAMA_NUM_PARALLEL=2 ollama serve
```

Parallel inference uses more memory and may not improve latency on CPU-only
systems. Without the flag, specialist execution remains sequential.

## Try it

```bash
python -m app "What is 2 + 2?"
python -m app "Tell me about apples"
python -m app "Give me a harvest weather pattern"
python -m app "Tell me about pears and give me a harvest weather pattern"
```

Specialist selection is model-driven. By default, the supervisor calls at most
one specialist at a time; `--parallel-specialists` allows it to call both in one
response when both are relevant. Each specialist is called no more than once.
The weather specialist returns one of `sunny`, `rainy`, or `dry`; because its
random-choice tool is optional, the model can also choose a label directly.
