# Benchmark guides

Complete the [installation and credentials](SETUP.md) first. Each
benchmark installs separately; install the ones you intend to run. Every command
below takes `--substrate cx` to use the
[Codex executor](SETUP.md#codex-executor) instead of OpenCode. Choose the
recursion mode with `--mode parallel` or `--mode sequential` and the task-level
recursion budget with `--k`. Keep both identical across the run, judge, and
report commands.

## LAB

Upstream: [harveyai/harvey-labs](https://github.com/harveyai/harvey-labs).
Legal work tasks with source documents and rubric-based grading. The adapter
groups tasks by practice area and removes criteria from the material passed to
the improver. Needs Pandoc and Poppler.

```bash
harness2-setup lab
harness2-check --bench lab
harness2 --bench lab --k 1 --mode parallel
harness2-judge --bench lab --k 1 --mode parallel
harness2-report --bench lab --k 1 --mode parallel
```

Sequential recursion runs the same commands with `--mode sequential`.

Tasks arrive with the pinned checkout at
`dependencies/harvey-labs/tasks/<practice-area>/<task>/task.json`, and the adapter
discovers all installed tasks; no separate task list is required. Setup installs
the LAB dependency environment, the runner overlays, and the included Vertex
judge patch.

The judge reads outputs only after selection. Reports use Partial (fraction of
criteria passed) and All-pass (all criteria passed). A failed judge remains
unscored and writes its output to `judge.log` in the run directory; a Vertex
`429 RESOURCE_EXHAUSTED` there means quota, so rerun `harness2-judge` later.
Changing the judge model changes the measurement. Outputs are under
`runs/lab/`.

## JobBench

Upstream: [Job-Bench/job-bench-eval](https://github.com/Job-Bench/job-bench-eval).
Data: [JobBench/job-bench](https://huggingface.co/datasets/JobBench/job-bench).
Professional tasks, with the library grouped by the dataset's occupation
directories.

```bash
harness2-setup jb
harness2-check --bench jb
harness2 --bench jb --k 1 --mode parallel
harness2-judge --bench jb --k 1 --mode parallel
harness2-report --bench jb --k 1 --mode parallel
```

Sequential recursion runs the same commands with `--mode sequential`.

Setup downloads only the main dataset from Hugging Face — public and ungated, so
no account, token, or terms acceptance is required — records its revision, and
places tasks at `dependencies/job-bench-eval/dataset/main/`. It also installs the
document extraction dependencies and a judge patch that handles reasoning effort
and empty completions. Rubrics stay outside the improver's evidence.

With a Gemini `HARNESS2_JUDGE_MODEL`, the judge uses the project's Vertex
OpenAI-compatible endpoint and Application Default Credentials. For a separate
judge provider, set `HARNESS2_JUDGE_BASE_URL`, `HARNESS2_JUDGE_API_KEY`, and the
model ID that endpoint accepts; the ID is sent as written. Without
`HARNESS2_JUDGE_API_KEY`, harness2 uses a `gcloud` access token, and only for a
Google endpoint (`*.googleapis.com`); any other endpoint needs the key, and
`harness2-check` reports it missing. The reported score uses the upstream rubric
weights; unscored tasks are reported separately. Results are under `runs/jb/`.

Note: JobBench data, including `RUBRICS.json`, is public on Hugging Face, so an
executor with web access may retrieve it. Block those hosts during runs as
described under [Web access](EXPERIMENTS.md#web-access).

### Judge with a Claude Code login

`harness2-judge-proxy` lets the JobBench judge use your Claude Code login instead
of a cloud endpoint, with no Google Cloud settings. It is a local server that
accepts OpenAI chat-completions requests and answers each one by running
`claude -p`. Install Claude Code and log in with `claude` first, then start the
server in a separate terminal and leave it running:

```bash
harness2-judge-proxy
```

It prints the settings to add to `.env`. On the first start it also prints a new
key; while that key is in `.env`, the server keeps using it. Set
`HARNESS2_JUDGE_MODEL` to a model name that `claude --model` accepts. Use a full
model ID rather than an alias such as `opus`, so that the run records the exact
judge model.

```bash
HARNESS2_JUDGE_BASE_URL=http://127.0.0.1:8642/v1
HARNESS2_JUDGE_API_KEY=<the key the server printed>
HARNESS2_JUDGE_MODEL=<full Claude model ID>
```

Then run `harness2-check` and `harness2-judge` as usual. For each request, the
server:

- runs `claude -p` in an empty folder, with no tools, no MCP servers, and none of
  your settings, `CLAUDE.md` files, skills, hooks or plugins;
- sends the system messages as the system prompt, and the user messages,
  including base64 images, as one user turn;
- passes the model to `--model` and the reasoning effort to `--effort` when
  `claude` accepts that value, and ignores other fields such as temperature and
  token limits;
- removes the settings that would make `claude` use an API key or a cloud
  provider instead of your login, such as `ANTHROPIC_API_KEY`.

The server listens on 127.0.0.1 only and refuses requests without the key. Each
judge call counts against your Claude plan's usage limits. The JobBench judge
makes one call per rubric, for example 8 for `biostatisticians/task1`. The
server runs at most 8 calls at once (`--max-calls`); further calls wait, and the
judge retries a call that gets no answer within 240 seconds.

Scores from this judge are not comparable to the paper's, because the judge
model is different.

The WorkBuddy-Bench web subset's grader reads the same two endpoint settings
(rerun `harness2-setup wb` after changing them), but it has not been tried with
this server. LAB cannot use it: the LAB judge has no client for OpenAI
chat-completions, and harness2 sends `claude-*` judge models to Vertex AI.

## WorkBuddy-Bench

Upstream: [Tencent/workbuddy-bench](https://github.com/Tencent/workbuddy-bench).
Data: [tencent/workbuddy-bench](https://huggingface.co/datasets/tencent/workbuddy-bench).
Office, web, and code subsets. Needs Docker.

The upstream custom Tencent license includes a territorial limitation; read
[the license](https://github.com/Tencent/workbuddy-bench/blob/main/LICENSE)
before downloading or using the benchmark. Derived runner files and patches
retain that license.

```bash
docker info
harness2-setup wb --subset office
harness2-check --bench wb --subset office
harness2 --bench wb --subset office --k 1 --mode parallel
harness2-judge --bench wb --subset office --k 1 --mode parallel
harness2-report --bench wb --subset office --k 1 --mode parallel
```

Sequential recursion runs the same commands with `--mode sequential`.

`harness2-setup wb` without `--subset` installs all three subsets in one run;
pass `--subset` to install only the one you intend to run. Every other command
takes one subset at a time.

Setup installs the pinned Harbor-based runtime, applies the included adapter,
mount, and credential/transport patches, copies the harness runners, writes the
solver and judge model configs from `.env`, downloads the requested subset with
the upstream fetch script, and builds the OpenCode harness image. No model YAML or
proxy configuration needs to be copied from another checkout.

Every WorkBuddy run loads a model config for its executor, and the web subset's
grader also loads the judge config. OpenCode reaches its model through the solver
config. Codex reaches its model directly, through an endpoint or the Codex login,
but its runs still load a model config: the solver config with an endpoint, and
with the login a `codex` config that setup writes from `HARNESS2_CODEX_MODEL`.
The solver and judge configs point at:

- the endpoint in `HARNESS2_SOLVER_BASE_URL` (executor) or
  `HARNESS2_JUDGE_BASE_URL` (judge), with its `..._API_KEY`, when that is set.
  Any service that speaks the OpenAI chat API works, such as OpenRouter, OpenAI
  or Ollama, and the executor model must support tool calls. The executor model
  name is sent without the part before the first `/`, the judge model name as
  written. Requests carry no output-token or reasoning settings, so the service's
  defaults apply. Ollama's default context is small (4,096 tokens on Ollama 0.40)
  and it silently drops the rest of a longer prompt, so give the model a larger
  context first, for example `PARAMETER num_ctx 65536` in its Modelfile;
- otherwise Vertex AI, for a `google-vertex/gemini...` executor or a bare
  `gemini...` judge with `GOOGLE_CLOUD_PROJECT` set. The job-local proxy refreshes
  ADC tokens and preserves Gemini tool-call thought signatures.

The configs name the URL and key variables and never hold a key. A config with
neither is skipped: setup completes and says so, and `harness2-check` reports it.

Setup patches the upstream fetch script to download with `curl` (then `wget`,
then `hf`) and to require the dataset repository's `SHA256SUMS`: an archive that
cannot be verified is not unpacked. A verified download prints
`wb-bench-<subset>-v1.0.tar.gz: OK`.

Tasks are discovered from `task.toml` and `instruction.md` under
`dependencies/workbuddy-bench/datasets/wb-bench-<subset>-v1.0/tasks/`. Libraries use
`metadata.role` on code and `metadata.category` on office and web. Each rollout
uses a fresh benchmark container. The adapter reads the verifier's `score.json`;
the improver never receives verifier output. Missing verifier results remain
unscored. Runs live under `runs/wb/<subset>/`.

With `--substrate cx` on a machine that is not Linux, setup also installs the
Linux build of Codex for the containers, in `dependencies/codex/linux-dist`. For
Codex on an endpoint, start the bridge with `harness2-bridge --host 0.0.0.0` so
that the containers can reach it. For Codex on the Codex login, see
[Codex login in WorkBuddy-Bench](SETUP.md#codex-login-in-workbuddy-bench).
