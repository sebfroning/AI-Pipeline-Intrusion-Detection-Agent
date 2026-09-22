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
ollama pull nomic-embed-text
```

### Libra GPU nodes

Ollama is not provided as a cluster module. On a GPU node, use the project scripts
(user-local install, no sudo):

```bash
srun --partition=gpu --gres=gpu:1 --pty bash
module load python/3.13.0-gcc-13.1.0-7ypl2
source .venv/bin/activate

# one-time install (already done if ~/.local/ollama/bin/ollama exists)
bash scripts/ollama/install.sh

# each GPU session
source scripts/ollama/env.sh
bash scripts/ollama/pull-model.sh   # start server + pull chat and embedding models
python -m app "How could dry weather affect an apple harvest?"
```

Or run everything in one step:

```bash
bash scripts/run-gpu.sh "How could dry weather affect an apple harvest?"
```

Stop the server when finished: `bash scripts/ollama/stop.sh`

## Run

Pass one prompt to the module:

```bash
python -m app "How could dry weather affect an apple harvest?"
```

The command prints which specialists were used, followed by the finalizer's
report. The supervisor may use neither specialist, one specialist, or both.

## Workflow

The outer LangGraph has a fixed execution path:

```text
START -> recall episodes -> supervisor agent/tool loop -> finalizer
      -> extract and save episode -> END
```

Specialist tools append typed entries containing the specialist name, focused
question, and finding to `specialist_results` in graph state. The list uses a
reducer so results from parallel tool calls are retained. The finalizer always
runs after a successful supervisor pass, reconciles all collected findings, and
stores its answer in `final_report` as well as the final message.

## Episodic memory

Each run retrieves up to three similar past episodes before the supervisor is
called. The recalled episodes are appended to the supervisor and specialist
system prompts and included in the finalizer input. They are explicitly marked
as historical experience: an old verdict can inform the approach to a new run,
but is not evidence about the new target.

After the final report, LangMem extracts noteworthy experience into a structured
episode containing the target, situation, approach, outcome, and reusable lesson.
Trivial runs may be skipped by the memory manager. Episodes are held in a
LangGraph `InMemoryStore` and written to `data/memory.json` after every successful
extraction. The file is loaded at the start of the next run.

Similarity search uses Ollama's `nomic-embed-text` model by default. Configure
memory with:

- `MEMORY_STORE_PATH`: JSON persistence path
- `MEMORY_EMBED_MODEL`: Ollama embedding model; set it to an empty string to
  disable vector ranking
- `MEMORY_EMBED_DIMS`: embedding dimensions, default `768`

Memory is best-effort. Retrieval or extraction errors are logged and do not
prevent the final report from being returned.

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
