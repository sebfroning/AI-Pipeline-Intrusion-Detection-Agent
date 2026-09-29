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

### PostgreSQL memory setup

Install Docker Engine with the Compose plugin (or Docker Desktop) using the
[Docker installation guide](https://docs.docker.com/engine/install/). Python and
Ollama run on the host; Compose runs a PostgreSQL 17/pgvector container for shared
memory and a separate container for each specialist.

From the repository root, create `.env` if it does not already exist:

```bash
cp -n .env.example .env
```

Set independent local passwords for `POSTGRES_PASSWORD`, `FRUIT_POSTGRES_PASSWORD`,
`WEATHER_POSTGRES_PASSWORD`, and `MITHRIDAT_POSTGRES_PASSWORD` in `.env`.
If upgrading an existing `.env`, add the specialist settings from `.env.example`.
Both Compose and
Python read this file; exported environment variables take precedence. `.env`
is ignored by Git. The password initializes a **new** database volume; editing
it later does not change the password of an existing database user.

Start the databases and initialize each schema:

```bash
docker compose up -d --wait
python -m app.memory init --all
python -m app.memory search "apple harvest"
python -m app.memory search "apple harvest" --owner fruit
python -m app "How could dry weather affect an apple harvest?"
```

| Memory owner | Compose service | Host port | Named volume |
| --- | --- | --- | --- |
| shared | `db` | 5432 | `memory_data` |
| fruit | `db-fruit` | 5433 | `fruit_memory_data` |
| weather | `db-weather` | 5434 | `weather_memory_data` |
| mithridat | `db-mithridat` | 5435 | `mithridat_memory_data` |

Ports bind to `127.0.0.1` and can be overridden in `.env`. A containerized app
on the Compose network should use each service name and container port 5432.

`init --all` creates LangGraph's tables and enables the `vector` extension in
every database. Use `init --owner fruit` to initialize just one. Initialization can be
rerun for schema migrations and does not require Ollama inference. Import and
semantic search require the embedding model to be available in Ollama.

New installations start with empty memory; no import is required. To import
legacy shared episodes, use `python -m app.memory import-json data/memory.json`.
The importer preserves record keys and values, maps the legacy shared namespace
to the current project's shared collection, and verifies saved content. Reruns
skip identical records and reject conflicting content instead of overwriting it.
The original JSON file remains untouched. Stop agent writers during import,
export, and reindex maintenance.
All maintenance commands accept `--owner shared|fruit|weather|mithridat` and
default to `shared`. Imports reject records from other owners before connecting;
exports and reindexing operate only on the selected owner's namespace. If an old
database contains specialist episodes, partition its export by namespace before
importing each group with the corresponding `--owner`. Existing records and
volumes are not automatically deleted or migrated.

Useful database commands:

```bash
docker compose ps
docker compose logs db
docker compose logs db-fruit
docker compose stop
docker compose start
```

Each named volume survives container recreation and ordinary
`docker compose down`. **`docker compose down -v` deletes all memory volumes.**
A volume is persistent storage, not a backup. To create a full database backup:

```bash
docker compose exec -T db sh -c 'pg_dump -U "$POSTGRES_USER" -d "$POSTGRES_DB" -Fc' > data/memory-backup.dump
```

Back up each specialist database the same way, replacing `db` and the output
filename. Keep backup copies outside the database host. For portable JSON exports:

```bash
python -m app.memory export-json /tmp/episodes.json
python -m app.memory export-json /tmp/fruit-episodes.json --owner fruit
python -m app.memory export-json /tmp/weather-episodes.json --owner weather
python -m app.memory export-json /tmp/mithridat-episodes.json --owner mithridat
```

Exports require a new output filename, include all pages of results, and contain
episode content rather than embeddings. Importing an export regenerates vectors.

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
      -> extract shared and specialist episodes into PostgreSQL -> END
```

Specialist tools append typed entries containing the specialist name, focused
question, and finding to `specialist_results` in graph state. The list uses a
reducer so results from parallel tool calls are retained. The finalizer always
runs after a successful supervisor pass, reconciles all collected findings, and
stores its answer in `final_report` as well as the final message.

## Episodic memory

Each run retrieves up to three similar shared episodes before the supervisor is
called. Each specialist searches again using its focused question and oversight
metadata, retrieving up to two episodes from its own collection and filling the
remaining slots (up to three total) with shared episodes. A specialist does not
search other specialists' collections. The supervisor's shared context is also
included in the finalizer input. Historical verdicts guide approach but are not
evidence about the current target.

After the final report, LangMem extracts noteworthy experience into a structured
episode containing the target, situation, approach, outcome, and reusable lesson.
Trivial runs may be skipped by the memory manager. LangMem writes directly to a
LangGraph `PostgresStore`; normal runs no longer load or rewrite the JSON file.
The database persists both episode content and embeddings. Existing memory
manager behavior, including updating relevant memories, is retained; this is a
collection of extracted experience, not an immutable archive of every run.

After shared extraction, the workflow separately extracts experience for each
specialist that ran, using only its question, finding, tool observations, and
oversight metadata. Failures in one extraction do not prevent the others. This
adds an extraction step per invoked specialist before the CLI returns.

Namespaces are `("memories", MEMORY_PROJECT, "episodes", "shared")` and
`("memories", MEMORY_PROJECT, "episodes", "specialists", specialist_name)`.
Each owner now has a separate database container, credentials, and persistent
volume. Namespaces remain as project/owner labels inside each database. Private
reads and writes select the specialist's connection; shared recall uses a second
connection to the shared database. Missing specialist credentials never redirect
writes to shared memory. Unknown specialist names are rejected.

Fruit and weather are integrated now. Mithridat's database and memory routing
are provisioned, but its standalone agent is not yet routed by this supervisor;
it can use `recall_context(..., specialist="mithridat")` and
`extract_episode(..., specialist="mithridat")` when that integration is added.
All agents still run in one Python process with access to the configured
credentials; separate databases do not isolate specialist code from that process.

Similarity search uses Ollama's `nomic-embed-text` model and exact cosine search
in pgvector. Exact search preserves filtered retrieval quality for these small
collections; an approximate index can be introduced after measuring scale.
Configure memory in `.env` or the environment:

- `POSTGRES_HOST`, `POSTGRES_PORT`, `POSTGRES_DB`, `POSTGRES_USER`,
  `POSTGRES_PASSWORD`: local database connection settings
- `MEMORY_DATABASE_URL`: optional complete PostgreSQL connection string,
  overriding the shared connection settings (also supports remote databases)
- `FRUIT_POSTGRES_*`, `WEATHER_POSTGRES_*`, `MITHRIDAT_POSTGRES_*`: the same
  connection settings for each specialist, with independent passwords
- `FRUIT_MEMORY_DATABASE_URL`, `WEATHER_MEMORY_DATABASE_URL`,
  `MITHRIDAT_MEMORY_DATABASE_URL`: optional full URLs overriding only that
  specialist's settings; the shared URL is never a specialist fallback
- `MEMORY_PROJECT`: project namespace, default `intrusion-agent`
- `MEMORY_EMBED_MODEL`: Ollama embedding model; set it to an empty string to
  disable vector ranking
- `MEMORY_EMBED_DIMS`: embedding dimensions, default `768`

The configured embedding model and dimension count are recorded independently
in every database at initialization. All owners initially use the same settings.
Changing either against an existing vector database is rejected: export the
episodes, initialize a new database with the new settings, and import them there.
To export with a mismatched local model configuration, use
`MEMORY_EMBED_MODEL="" python -m app.memory export-json /tmp/episodes.json`.
If vector ranking is temporarily disabled, new or updated episodes are not
re-embedded. After restoring the original model settings, run
`python -m app.memory reindex --owner <owner>` for each affected database before
relying on semantic retrieval again.

You can inspect specialist retrieval without running a chat agent:

```bash
python -m app.memory search "apple inspection" --specialist fruit
```

Memory is best-effort. Retrieval or extraction errors are logged and do not
prevent the final report from being returned.
Private and shared retrieval fail independently: a shared outage preserves private
results, while a private outage allows shared episodes to fill the requested limit.
Failed extractions are logged, not automatically retried. Database connections
are opened per memory operation and closed afterward, including on errors.

For cluster jobs, run PostgreSQL on a persistent host reachable from the compute
nodes and set `MEMORY_DATABASE_URL` plus each specialist's `*_MEMORY_DATABASE_URL`
there. The default Compose ports are bound to
localhost for development; it is not directly reachable from another machine.
Use the cluster's supported database/networking arrangement rather than starting
a temporary database separately inside every GPU job.

### Tests

The default tests use isolated in-memory stores and deterministic embeddings:

```bash
python -m unittest discover -s tests -v
```

To test real PostgreSQL persistence, parallel writes, and vector ranking, create
a separate database and explicitly opt in. The integration test creates its
schema, uses a unique namespace, and deletes its test records. It uses synthetic
vectors, so Ollama is not required:

```bash
TEST_MEMORY_DATABASE_URL='postgresql://user:password@127.0.0.1:5432/test_memory' \
  python -m unittest discover -s tests -p test_memory_postgres.py -v
```

To also test physical separation and shared recall, provide
`TEST_FRUIT_MEMORY_DATABASE_URL`, `TEST_WEATHER_MEMORY_DATABASE_URL`, and
`TEST_MITHRIDAT_MEMORY_DATABASE_URL`, each pointing to a dedicated test database
in its respective container, alongside `TEST_MEMORY_DATABASE_URL`. The test
writes identical keys in all four stores and verifies they remain independent.
Use test databases, not the application's databases, for these synthetic vectors.

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
