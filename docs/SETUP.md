# Setup guide

Start with the [quick start](../README.md#install). This guide covers benchmark
dependencies, model credentials, and the optional Codex executor. For individual
benchmarks, see the [benchmark guides](BENCHMARKS.md).

## Requirements

Use Linux or macOS and Python 3.12 for benchmark runs; the Python package also
supports 3.13. On Debian/Ubuntu:

```bash
sudo apt-get update
sudo apt-get install -y git curl unzip ca-certificates pandoc poppler-utils
```

On macOS, git, curl and unzip come with the system, and LAB's Pandoc and Poppler
install with `brew install pandoc poppler`. The runner scripts work with macOS's
own bash 3.2 and need no GNU tools.

Pandoc and Poppler are needed by LAB. Also install:

- [Node.js](https://nodejs.org/en/download) 20 or newer.
- [uv](https://docs.astral.sh/uv/getting-started/installation/).
- [Google Cloud CLI](https://cloud.google.com/sdk/docs/install).
- [Docker Engine](https://docs.docker.com/engine/install/) for WorkBuddy-Bench
  (Docker Desktop on macOS), with permission to run `docker info`.

Benchmark setup installs its own pinned Bun runtime under `dependencies/`.
Keep the main Harness² environment active when running the commands below.

## Models and credentials

Copy `.env.example` to `.env` if you have not already done so. Every setting in it
is commented out: remove the `#` only from the settings you fill in. A setting left
empty is still passed to the tools harness2 starts, and some treat an empty value
as an error; for example, an empty `GOOGLE_APPLICATION_CREDENTIALS` makes `gcloud`
fail. Set your Vertex AI project and model IDs, then authenticate:

```bash
gcloud auth application-default login
```

Shell environment variables take precedence over `.env`.
`GOOGLE_CLOUD_PROJECT` identifies your project; `HARNESS2_VERTEX_PROJECT` is read
when it is not set. `GOOGLE_CLOUD_LOCATION` defaults to `global`. Your account must
have access to each configured model.

| Setting | Meaning |
|---|---|
| `HARNESS2_SOLVER_MODEL` | Executor model in OpenCode's `provider/model` format |
| `HARNESS2_IMPROVER_CHANNEL` | How the improvement agents run: `opencode` (default) or `claude_cli`, the `claude` command on its own login. See [Improvement agents on a Claude login](#improvement-agents-on-a-claude-login) |
| `HARNESS2_IMPROVER_MODEL` | Improver model. With `opencode`, in OpenCode's `provider/model` format; defaults to the executor. With `claude_cli`, a Claude model name such as `sonnet`, `opus` or a full `claude-…` ID; required |
| `HARNESS2_JUDGE_MODEL` | Bare judge model ID; the adapter supplies the Vertex prefix for Gemini IDs, and on LAB also for Claude IDs |
| `HARNESS2_SOLVER_BASE_URL`, `HARNESS2_SOLVER_API_KEY` | WorkBuddy-Bench only. Endpoint and API key for the executor model: any service that speaks the OpenAI chat API, such as OpenRouter, OpenAI or Ollama. Leave unset to use Vertex AI. See [WorkBuddy-Bench](BENCHMARKS.md#workbuddy-bench) |
| `HARNESS2_JUDGE_BASE_URL`, `HARNESS2_JUDGE_API_KEY` | JobBench and the WorkBuddy-Bench web subset. Endpoint and API key for the judge model, on the same kind of service. Leave unset to use Vertex AI. See [JobBench](BENCHMARKS.md#jobbench) |

The executor and the judge are usually different models, so each has its own
endpoint setting. Both can point at the same service.
| `HARNESS2_RUNS_DIR` | Output root; defaults to `runs/`. See [results and resume](EXPERIMENTS.md#results-and-resume) |

The provided configuration uses Vertex Gemini executors and judges, and supports
either Vertex Claude or Gemini as the improver. LAB also supports the upstream
judge's direct providers when their credentials are configured. For WorkBuddy,
setup writes the solver and judge model configs from these settings, for an
OpenAI-compatible endpoint or for Vertex Gemini; see
[WorkBuddy-Bench](BENCHMARKS.md#workbuddy-bench). Rerun setup after changing
either model or endpoint.

### Authentication alternatives

On a machine with a configured attached service account, Application Default
Credentials can use that account. Other options include:

- `gcloud auth application-default login --no-browser`, which prints a command
  to run on another machine that has a browser.
- `GOOGLE_APPLICATION_CREDENTIALS` pointing at a service account key file.
- `gcloud auth application-default login --impersonate-service-account=<account>`.

Keep `gcloud` on `PATH` for these Vertex configurations, including the key-file
route: the installation check runs
`gcloud auth application-default print-access-token`.

### Improvement agents on a Claude login

The improvement agents (editor, composer, selector and reflector) can run
through the `claude` command (Claude Code) instead of OpenCode, on the Claude
account it is logged in to. Install Claude Code, log in with `claude auth login`,
and set:

```bash
HARNESS2_IMPROVER_CHANNEL=claude_cli
HARNESS2_IMPROVER_MODEL=sonnet      # or opus, or a full claude-... ID
```

`harness2-check` and each run check the login with `claude auth status` before
starting. harness2 runs each session with `claude -p`:

- without an Anthropic API key and without Vertex AI, Bedrock, Foundry or Google
  Cloud settings, so it uses the login;
- without your own Claude Code settings, `CLAUDE.md`, plugins, hooks and MCP
  servers;
- able to read and change files only in its session folder: Claude Code refuses
  anything that would need approval, such as running a script;
- without saving it to your Claude Code history. Its prompt and reply are kept
  under `runs/<benchmark>/adapt/_audit/`.

The sessions count against the plan's usage limits. The cost harness2 records
for them is the API price that `claude` reports, not an amount billed to the
plan. Do not log OpenCode in with a Claude account: using a Claude login in
other tools is not allowed.

## Benchmark installation

Install only the benchmark you intend to run:

```bash
harness2-setup lab
harness2-check --bench lab
```

Use `jb` for JobBench or `wb` for WorkBuddy-Bench. See the
[benchmark guides](BENCHMARKS.md) for subset selection and licensing.

The runner overlays and compatibility patches are included in
[`third_party/`](../third_party/README.md). Setup clones supported upstream
revisions under `dependencies/`, applies the patches, creates the benchmark
environments, and downloads task data.

Each benchmark environment lives inside its upstream checkout:

- `dependencies/harvey-labs/.venv`
- `dependencies/job-bench-eval/.venv`
- `dependencies/workbuddy-bench/.venv`

Harness² selects these interpreters automatically; do not activate them for the
main commands.

### Installation checks

`harness2-setup <bench> --dry-run` previews setup without making changes.
`harness2-check` checks installed dependencies, credentials, and configuration
without calling a model. Refinement runs perform the same checks, including
judge configuration, before starting.

These checks do not test model availability. Access to an executor, improver, or
judge model is tested when that model is first called. Refinement and judging
return a nonzero exit status when incomplete.

## Codex executor

The Codex executor reaches its model in one of two ways: through your Codex
login (a ChatGPT plan), or through a Responses API endpoint and key.

### Codex login (ChatGPT plan)

Set these in `.env`, then install and log harness2's Codex in once:

```bash
HARNESS2_CODEX_AUTH=chatgpt
HARNESS2_CODEX_MODEL=<a model the pinned Codex can use>
```

```bash
harness2-setup jb --substrate cx
CODEX_HOME=~/.harness2/codex dependencies/codex/dist/codex login
CODEX_HOME=~/.harness2/codex dependencies/codex/dist/codex debug models
harness2-check --bench jb --substrate cx
```

`debug models` lists the models this Codex version can use. The pinned Codex
(0.144.6) refuses newer models such as `gpt-6.1-sol` with "not supported when
using Codex with a ChatGPT account".

About the login:

- It is harness2's own, kept in `HARNESS2_CODEX_HOME` (default
  `~/.harness2/codex`). That folder must be outside this repository, because the
  login file holds access tokens.
- It is separate from your `~/.codex`. Codex renews a login by rewriting its file,
  and renewing one copy can end the session another copy holds.
- Every Codex run on JobBench and LAB uses this one file, and Codex renews it in
  place about every 8 days. OpenAI recommends one login file per concurrent job:
  if two runs renew it at the same moment, one of them can fail. Running the same
  command again resumes it.
- WorkBuddy-Bench runs Codex inside Docker and cannot use the login. Use an
  endpoint and key for it.

### Endpoint and key

For Gemini on Vertex AI, the included `harness2-bridge` provides a Responses API
endpoint through a local LiteLLM server, using the same Application Default
Credentials and `GOOGLE_CLOUD_PROJECT`.

Install the optional dependencies and start the bridge in a separate session:

```bash
uv pip install -e '.[render,bridge]'
harness2-bridge                 # for WorkBuddy: harness2-bridge --host 0.0.0.0
```

The bridge prints the `HARNESS2_CODEX_*` settings to add to `.env`. Its port and
generated access key are stored under `dependencies/codex/` and reused on
restart. Pass `--substrate cx` to setup, checks, refinement, judging, and
reporting:

```bash
harness2-setup lab --substrate cx
harness2-check --bench lab --substrate cx
harness2 --bench lab --substrate cx --k 1 --mode parallel \
  --only-domains antitrust-competition --max-tasks 1
```

WorkBuddy executes Codex inside Docker containers, so the bridge must listen on
`--host 0.0.0.0`; it also prints `HARNESS2_CODEX_CONTAINER_URL`. This binds the
bridge to all network interfaces; restrict access to the intended clients.

To use another Responses API endpoint, set `HARNESS2_CODEX_BASE_URL`,
`HARNESS2_CODEX_API_KEY`, and `HARNESS2_CODEX_MODEL` directly. Setup records the
endpoint and model in `dependencies/codex/home/config.toml`; rerun setup after
changing either. `harness2-check` reports stale configuration until then.

Codex runs with its sandbox disabled. Review the
[executor network restrictions](EXPERIMENTS.md#web-access) before evaluation.

## Network access during setup

Setup downloads dependencies from the following hosts:

| Host | Used for |
|---|---|
| `github.com` | Pinned upstream checkouts |
| `registry.npmjs.org` | Pinned Bun runtime and Codex CLI, plus dependencies resolved by `bun install` |
| `huggingface.co` and its CDN | JobBench data and WorkBuddy subset archives |
| A Docker registry | WorkBuddy container image |
| `astral.sh` | CPython 3.12 when the host has no system Python 3.12 |

Use `harness2-setup <bench> --dry-run` to review setup commands when configuring
network access. Experiments also access the model endpoints configured in
`.env`. Setup downloads and executor web access have different requirements;
see [web access during evaluation](EXPERIMENTS.md#web-access).

## Wheel installations

A wheel ships the `harness2` package only. The task partitioner,
`python scripts/split_tasks.py`, requires this source tree. For a wheel
installation, work from a dedicated directory: `.env`, `dependencies/`, and
`runs/` are resolved there.
