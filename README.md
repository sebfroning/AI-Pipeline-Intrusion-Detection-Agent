# Local supervisor and specialists

A small, local multi-agent example built with LangChain and Ollama. A supervisor
answers general questions itself and delegates fruit or harvest-weather questions
to focused specialist agents when needed.

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
bash scripts/ollama/pull-model.sh   # start server + pull qwen3:4b
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

The command prints which specialists were used, followed by the answer. Each run
is stateless. The supervisor may use neither specialist, one specialist, or both.

To use another installed Ollama model:

```bash
OLLAMA_MODEL=another-model python -m app "Tell me about pears"
```

## Try it

```bash
python -m app "What is 2 + 2?"
python -m app "Tell me about apples"
python -m app "Give me a harvest weather pattern"
python -m app "Tell me about pears and give me a harvest weather pattern"
```

Specialist selection is model-driven. The prompts tell the supervisor to call at
most one specialist at a time and each specialist no more than once. The weather
specialist returns one of `sunny`, `rainy`, or `dry`; because its random-choice
tool is optional, the model can also choose a label directly.
