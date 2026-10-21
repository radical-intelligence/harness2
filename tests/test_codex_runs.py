# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Codex runs: the installed-version check, HTTP 429 retries, the usage-limit stop, and the
JobBench runner's temp folders.

`codex` is replaced by a fake that plays back a list of outcomes, one per `codex exec`.
"""

import contextlib
import io
import json
import os
import subprocess
import sys
import tarfile
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from harness2 import config, install
from harness2.benchmarks import jobbench, lab, workbuddy
from harness2.cli import run_sweep
from harness2.core import schedule
from harness2.demo import DemoAdapter, DemoProposer
from harness2.substrate import codex as codex_substrate
from harness2.substrate.base import Install, RunRequest
from harness2.substrate.codex import CodexRateLimited, CodexUsageLimit

RATE_LIMITED = "exceeded retry limit, last status: 429 Too Many Requests"
USAGE_LIMIT = (
    "You’ve hit your usage limit. Upgrade to Pro (https://chatgpt.com/explore/pro), "
    "visit https://chatgpt.com/settings/usage to purchase more credits or try again at "
    "Oct 22, 2026 3:00 PM."
)
NO_WAITS = (0.01, 0.01, 0.01)

FAKE_CODEX = """#!/usr/bin/env python3
# Each `codex exec` takes the next outcome from $FAKE_CODEX_PLAN (comma-separated; the
# last one repeats) and records its -C folder in $FAKE_CODEX_STATE/calls.
import json, os, sys, time
from pathlib import Path

if sys.argv[1:] == ["--version"]:
    print("codex-cli 0.163.0")
    sys.exit(0)
state = Path(os.environ["FAKE_CODEX_STATE"])
ws = sys.argv[sys.argv.index("-C") + 1]
with open(state / "calls", "a") as calls:
    calls.write(ws + "\\n")
n = len((state / "calls").read_text().splitlines())
plan = os.environ.get("FAKE_CODEX_PLAN", "ok").split(",")
outcome = plan[min(n, len(plan)) - 1]


def emit(**event):
    print(json.dumps(event), flush=True)


emit(type="thread.started", thread_id=f"thread-{n}")
emit(type="turn.started")
if outcome in ("ok", "slow"):
    if outcome == "slow":
        time.sleep(2)
    (Path(ws) / "output").mkdir(exist_ok=True)
    (Path(ws) / "output" / "answer.txt").write_text("done\\n")
    emit(type="item.completed", item={"id": "item_0", "type": "agent_message", "text": "done"})
    emit(type="turn.completed", usage={"input_tokens": 1, "output_tokens": 1})
    sys.exit(0)
message = {
    "429": %r,
    "usage": %r,
    "fail": "stream disconnected before completion",
}[outcome]
emit(type="error", message=message)
emit(type="turn.failed", error={"message": message})
sys.exit(1)
""" % (RATE_LIMITED, USAGE_LIMIT)


def fake_dist(path: Path, version: str = config.CODEX_VERSION) -> Path:
    """A Codex install folder whose `codex` is the fake."""
    (path / "bin").mkdir(parents=True)
    (path / "codex-path").mkdir()
    (path / "bin" / "codex").write_text(FAKE_CODEX, encoding="utf-8")
    (path / "bin" / "codex").chmod(0o755)
    (path / "codex").symlink_to("bin/codex")
    (path / "codex-package.json").write_text(json.dumps({"version": version}))
    return path


def stream(*events: dict) -> str:
    return "".join(json.dumps(e) + "\n" for e in events)


def failed_stream(message: str) -> str:
    return stream(
        {"type": "thread.started"},
        {"type": "error", "message": message},
        {"type": "turn.failed", "error": {"message": message}},
    )


OK_STREAM = stream({"type": "thread.started"}, {"type": "turn.completed", "usage": {}})


@contextlib.contextmanager
def captured(path: Path):
    """Send this process's stdout and stderr, children included, to `path`."""
    sys.stdout.flush()
    sys.stderr.flush()
    saved = os.dup(1), os.dup(2)
    with open(path, "w") as sink:
        os.dup2(sink.fileno(), 1)
        os.dup2(sink.fileno(), 2)
        try:
            yield
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(saved[0], 1)
            os.dup2(saved[1], 2)
            for fd in saved:
                os.close(fd)


class TempDirTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.addCleanup(patch.stopall)
        self.root = Path(temporary.name).resolve()


class FailureDetectionTests(TempDirTest):
    def write(self, name: str, text: str) -> Path:
        path = self.root / name
        path.write_text(text, encoding="utf-8")
        return path

    def test_rate_limit_and_usage_limit_are_read_from_codex_messages(self):
        found = codex_substrate.run_failure(self.write("a.jsonl", failed_stream(RATE_LIMITED)))
        self.assertEqual(found, ("rate_limited", RATE_LIMITED))
        for wording in (USAGE_LIMIT, USAGE_LIMIT.replace("’", "'")):
            path = self.write("b.jsonl", failed_stream(wording))
            self.assertEqual(codex_substrate.run_failure(path), ("usage_limit", wording))
        stderr = self.write("stderr.log", f"ERROR: {RATE_LIMITED}\n")
        self.assertEqual(
            codex_substrate.run_failure(self.root / "missing.jsonl", stderr)[0], "rate_limited"
        )

    def test_finished_runs_and_command_output_never_count(self):
        finished = stream(
            {"type": "error", "message": RATE_LIMITED},
            {"type": "turn.completed", "usage": {}},
        )
        self.assertIsNone(codex_substrate.run_failure(self.write("a.jsonl", finished)))
        output = stream(
            {
                "type": "item.completed",
                "item": {"type": "command_execution", "aggregated_output": RATE_LIMITED},
            },
            {"type": "turn.failed", "error": {"message": "stream disconnected"}},
        )
        self.assertIsNone(codex_substrate.run_failure(self.write("b.jsonl", output)))

    def test_the_usage_limit_names_the_reset_time(self):
        self.assertEqual(CodexUsageLimit(USAGE_LIMIT).reset, "Oct 22, 2026 3:00 PM")
        self.assertIn(USAGE_LIMIT, str(CodexUsageLimit(USAGE_LIMIT)))
        self.assertEqual(CodexUsageLimit("You've hit your usage limit.").reset, "")
        self.assertFalse(issubclass(CodexUsageLimit, Exception))


class VersionCheckTests(TempDirTest):
    def setUp(self):
        super().setUp()
        self.dist = self.root / "dist"
        self.linux = self.root / "linux-dist"
        patch.multiple(
            config,
            CODEX_DIST=str(self.dist),
            CODEX_CONTAINER_DIST=str(self.linux),
            CODEX_AUTH="",
            CODEX_HOME_DIR=self.root / "home",
            SOLVER_MODEL_CX="gpt-test",
        ).start()

    def test_a_missing_or_other_version_names_both_versions_and_the_setup_command(self):
        error = codex_substrate.install_error("jb")
        self.assertIn("Codex is not installed", error)
        self.assertIn("harness2-setup jb --substrate cx", error)
        fake_dist(self.dist, "0.144.6")
        error = codex_substrate.install_error("lab")
        self.assertIn("is version 0.144.6", error)
        self.assertIn(f"harness2 uses {config.CODEX_VERSION}", error)
        self.assertIn("harness2-setup lab --substrate cx", error)
        (self.dist / "codex-package.json").unlink()
        self.assertIn("is version unknown", codex_substrate.install_error("lab"))

    def test_workbuddy_also_checks_the_linux_build(self):
        fake_dist(self.dist)
        self.assertIsNone(codex_substrate.install_error("wb"))
        error = codex_substrate.install_error("wb", containers=True)
        self.assertIn("Linux containers is not installed", error)
        fake_dist(self.linux, "0.144.6")
        error = codex_substrate.install_error("wb", containers=True)
        self.assertIn("Linux containers", error)
        self.assertIn("is version 0.144.6", error)
        self.assertIn("harness2-setup wb --substrate cx", error)

    def test_each_benchmark_check_refuses_an_old_codex(self):
        fake_dist(self.dist, "0.144.6")
        adapters = {
            "jb": jobbench.JobBenchAdapter("cx"),
            "lab": lab.LabAdapter(substrate="cx"),
            "wb --subset code": workbuddy.WbAdapter("code", substrate="cx"),
        }
        for bench, adapter in adapters.items():
            with self.subTest(bench=bench):
                self.assertEqual(adapter.substrate.bench, bench)
                with self.assertRaisesRegex(RuntimeError, "is version 0.144.6") as caught:
                    adapter.substrate.preflight()
                self.assertIn(f"harness2-setup {bench} --substrate cx", str(caught.exception))
        with self.assertRaisesRegex(RuntimeError, "harness2-setup jb --substrate cx"):
            adapters["jb"].preflight()

    def test_setup_replaces_the_whole_install(self):
        fake_dist(self.dist, "0.144.6")
        (self.dist / "file-of-the-old-version").write_text("")
        root = self.root / "deps" / "codex"
        patch.multiple(config, DEPENDENCIES=self.root / "deps", CODEX_AUTH="chatgpt").start()

        def npm_install(cmd, **_kwargs):
            self.assertEqual(cmd[-1], f"@openai/codex@{config.CODEX_VERSION}")
            vendor = root / "node_modules/@openai/codex-darwin-arm64/vendor/aarch64-apple-darwin"
            fake_dist(vendor)
            (vendor / "codex").unlink()

        with (
            patch.object(install, "run", side_effect=npm_install),
            patch("sys.stdout", new_callable=io.StringIO),
        ):
            install.codex()
        self.assertFalse((self.dist / "file-of-the-old-version").exists())
        self.assertEqual(codex_substrate.installed_version(self.dist), config.CODEX_VERSION)
        self.assertEqual(os.readlink(self.dist / "codex"), "bin/codex")
        self.assertIsNone(codex_substrate.install_error("jb"))

    def test_setup_replaces_the_whole_linux_build(self):
        fake_dist(self.dist)
        fake_dist(self.linux, "0.144.6")
        (self.linux / "file-of-the-old-version").write_text("")

        def npm_pack(cmd, cwd=None, **_kwargs):
            self.assertEqual(cmd[2], f"@openai/codex@{config.CODEX_VERSION}-linux-arm64")
            build = Path(cwd) / "package" / "vendor" / "aarch64-unknown-linux-musl"
            fake_dist(build)
            (build / "codex").unlink()
            with tarfile.open(Path(cwd) / "codex.tgz", "w:gz") as tar:
                tar.add(Path(cwd) / "package", arcname="package")

        with (
            patch.object(install, "run", side_effect=npm_pack),
            patch.object(install.os, "uname", Mock(return_value=Mock(machine="arm64"))),
        ):
            install.codex_for_containers()
        self.assertFalse((self.linux / "file-of-the-old-version").exists())
        self.assertIsNone(codex_substrate.install_error("wb", containers=True))


class RetryTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(patch.stopall)
        patch.object(config, "CODEX_429_WAITS", NO_WAITS).start()
        self.discarded = 0

    def run_plan(self, plan):
        outcomes = iter(plan)

        def discard():
            self.discarded += 1

        with patch("sys.stdout", new_callable=io.StringIO) as out:
            try:
                codex_substrate.retry_rate_limited(
                    lambda: next(outcomes), discard, label="test rollout"
                )
            finally:
                self.log = out.getvalue()

    def test_three_retries_then_a_connection_failure(self):
        self.run_plan([("rate_limited", RATE_LIMITED), None])
        self.assertEqual(self.discarded, 1)
        self.assertIn("waiting 0.01 s, then retry 1 of 3", self.log)
        with self.assertRaisesRegex(CodexRateLimited, "after 3 retries.*connection failure"):
            self.run_plan([("rate_limited", RATE_LIMITED)] * 4)
        self.assertEqual(self.log.count("waiting"), 3)

    def test_the_default_waits_are_1_2_and_4_minutes(self):
        self.assertEqual(
            config.env_seconds("HARNESS2_UNSET_FOR_TEST", "60,120,240"), (60, 120, 240)
        )
        with patch.dict(os.environ, {"HARNESS2_CODEX_429_WAITS": "1, 2"}):
            self.assertEqual(config.env_seconds("HARNESS2_CODEX_429_WAITS", "60"), (1.0, 2.0))

    def test_the_usage_limit_stops_without_waiting(self):
        with self.assertRaises(CodexUsageLimit):
            self.run_plan([("usage_limit", USAGE_LIMIT)])
        self.assertEqual((self.discarded, self.log), (1, ""))


class JobBenchCodexRunTests(TempDirTest):
    """The JobBench Codex runner script, run for real with the fake `codex`."""

    TASK = "prof/task1"

    def setUp(self):
        super().setUp()
        self.dist = fake_dist(self.root / "dist")
        self.state = self.root / "state"
        self.state.mkdir()
        self.workspaces = self.root / "workspaces"
        self.workspaces.mkdir()
        self.store = self.root / "store"
        task = self.store / self.TASK / "task_folder"
        task.mkdir(parents=True)
        (task / "TASK_INSTRUCTIONS.txt").write_text("Write answer.txt.\n")
        self.harness = self.root / "harness"
        self.harness.mkdir()
        (self.harness / "AGENTS.md").write_text("Be brief.\n")
        patch.multiple(
            config,
            CODEX_DIST=str(self.dist),
            CODEX_HOME_DIR=self.root / "home",
            SOLVER_MODEL_CX="gpt-test",
            CODEX_429_WAITS=NO_WAITS,
        ).start()
        patch.dict(
            os.environ,
            {"FAKE_CODEX_STATE": str(self.state), "TEMP_WORKSPACE_BASE": str(self.workspaces)},
        ).start()
        patch.object(jobbench, "codex_config_error", return_value=None).start()
        self.adapter = jobbench.JobBenchAdapter("cx", store_root=self.store)
        patch.object(self.adapter.substrate, "preflight").start()

    def calls(self) -> list[str]:
        path = self.state / "calls"
        return path.read_text().splitlines() if path.is_file() else []

    def launch(self, plan: str, label: str = "base1"):
        os.environ["FAKE_CODEX_PLAN"] = plan
        req = RunRequest(
            task_ref=self.TASK, run_ids=(f"jb-cx-agg1-t/{self.TASK}/{label}",), timeout=60
        )
        self.log_path = self.root / "log.txt"
        with captured(self.log_path):
            return self.adapter._launch_codex(req, Install(self.harness, "base", {}))

    def test_a_429_is_retried_from_a_fresh_start_and_kept(self):
        out = self.launch("429,ok")
        self.assertEqual(len(self.calls()), 2)
        self.assertEqual((out[0] / "answer.txt").read_text(), "done\n")
        streams = list(self.adapter._traj_dir(self.TASK, out[0].name).glob("*.jsonl"))
        self.assertEqual(len(streams), 1)
        self.assertIsNone(codex_substrate.run_failure(streams[0]))
        log = self.log_path.read_text()
        self.assertIn("Codex stopped on HTTP 429", log)
        self.assertIn("retry 1 of 3 from a fresh start", log)
        self.assertEqual(list(self.workspaces.iterdir()), [])

    def test_a_429_that_persists_is_a_connection_failure(self):
        with self.assertRaisesRegex(CodexRateLimited, "connection failure"):
            self.launch("429")
        self.assertEqual(len(self.calls()), 4)
        label = f"jb-cx-agg1-t__{self.TASK.replace('/', '__')}__base1"
        self.assertFalse(self.adapter._out_dir(self.TASK, label).exists())
        self.assertFalse(self.adapter._traj_dir(self.TASK, label).exists())

    def test_the_usage_limit_stops_with_the_reset_time(self):
        with self.assertRaises(CodexUsageLimit) as caught:
            self.launch("usage")
        self.assertEqual(caught.exception.reset, "Oct 22, 2026 3:00 PM")
        self.assertEqual(len(self.calls()), 1)

    def test_other_failures_are_not_retried(self):
        with self.assertRaisesRegex(RuntimeError, "Benchmark runner exited") as caught:
            self.launch("fail")
        self.assertNotIsInstance(caught.exception, CodexRateLimited)
        self.assertEqual(len(self.calls()), 1)

    def test_parallel_runs_of_one_task_get_their_own_folders(self):
        env = {
            **os.environ,
            "FAKE_CODEX_PLAN": "slow",
            "HARNESS2_CODEX_BIN": str(self.dist / "codex"),
            "TASKS_BASE_DIR": str(self.store),
            "TASK_IDS": self.TASK,
            "BENCHMARK_MODELS": "gpt-test",
            "TIMEOUT_PER_TASK": "60",
        }
        runner = config.ROOT / "third_party/jb_runner/run_benchmark_codex_cli_adapt.sh"
        procs = [
            subprocess.Popen(
                ["/bin/bash", str(runner)],
                env={**env, "LABEL": label},
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            for label in ("run-a", "run-b")
        ]
        self.assertEqual([p.wait(timeout=120) for p in procs], [0, 0])
        folders = self.calls()
        self.assertEqual(len(set(folders)), 2, folders)
        for folder in folders:
            self.assertEqual(Path(folder).parent, self.workspaces)
            self.assertTrue(Path(folder).name.startswith("prof_task1_gpt-test_"))
        self.assertEqual(list(self.workspaces.iterdir()), [])
        for label in ("run-a", "run-b"):
            answer = self.store / self.TASK / "model_output" / label / "answer.txt"
            self.assertEqual(answer.read_text(), "done\n")


class LabCodexRunTests(TempDirTest):
    """LAB's launcher, with the runner replaced by one that writes what the real one would."""

    def setUp(self):
        super().setUp()
        patch.multiple(config, RUNS=self.root / "runs", CODEX_429_WAITS=NO_WAITS).start()
        patch.object(config, "lab_repo", return_value=self.root / "lab").start()
        patch.object(lab, "_require_runner").start()
        self.adapter = lab.LabAdapter(substrate="cx")
        self.plan: list[str] = []
        self.calls = 0

        def runner(cmd, **_kwargs):
            run_dir = self.adapter._run_dir(cmd[cmd.index("--run-id") + 1])
            self.assertFalse((run_dir / "metrics.json").exists(), "not a fresh start")
            outcome = self.plan[min(self.calls, len(self.plan) - 1)]
            self.calls += 1
            (run_dir / "output").mkdir(parents=True, exist_ok=True)
            texts = {
                "ok": OK_STREAM,
                "429": failed_stream(RATE_LIMITED),
                "usage": failed_stream(USAGE_LIMIT),
            }
            (run_dir / "codex_session.jsonl").write_text(texts[outcome])
            (run_dir / "codex_stderr.log").write_text("")
            (run_dir / "metrics.json").write_text(json.dumps({"exit_code": int(outcome != "ok")}))
            if outcome == "ok":
                (run_dir / "output" / "memo.md").write_text("done\n")
            else:
                raise RuntimeError("Benchmark runner exited with status 1")

        patch.object(lab, "run_detached", side_effect=runner).start()

    def launch(self, plan):
        self.plan = plan
        req = RunRequest(
            task_ref="area/task", run_ids=("lab-cx-agg1-t/area/task/base1",), timeout=60
        )
        with patch("sys.stdout", new_callable=io.StringIO) as out:
            try:
                return self.adapter._launch_cx(req, Install(self.root, "base", {}))
            finally:
                self.log = out.getvalue()

    def test_a_429_run_is_discarded_and_rolled_again(self):
        (run_dir,) = self.launch(["429", "ok"])
        self.assertEqual(self.calls, 2)
        self.assertEqual((run_dir / "output" / "memo.md").read_text(), "done\n")
        self.assertIn("retry 1 of 3", self.log)

    def test_the_usage_limit_leaves_no_run_to_score(self):
        with self.assertRaises(CodexUsageLimit):
            self.launch(["usage"])
        self.assertEqual(self.calls, 1)
        self.assertFalse(self.adapter._run_dir("lab-cx-agg1-t/area/task/base1").exists())


class WorkBuddyCodexRunTests(TempDirTest):
    """WorkBuddy-Bench's trials, with Harbor replaced by a batch that writes trial folders."""

    def setUp(self):
        super().setUp()
        patch.multiple(config, RUNS=self.root / "runs", CODEX_429_WAITS=NO_WAITS).start()
        patch.object(workbuddy, "resolve_task", side_effect=lambda name, subset: name).start()
        self.adapter = workbuddy.WbAdapter("code", substrate="cx")
        patch.object(self.adapter.substrate, "preflight").start()
        self.outcomes: list[str] = []
        self.batches: list[int] = []

        def run_batch(subset, substrate, label_dir, label, tasks, ws, k, *, done, need, timeout_s):
            wanted = need(tasks[0])
            self.batches.append(wanted)
            run = label_dir / f"run{len(self.batches)}"
            for i in range(wanted):
                outcome = self.outcomes.pop(0)
                trial = run / f"{tasks[0]}__t{len(self.batches)}{i}"
                (trial / "agent").mkdir(parents=True)
                (trial / "verifier").mkdir()
                (trial / "verifier" / "score.json").write_text('{"score": 0}')
                texts = {
                    "ok": OK_STREAM,
                    "429": failed_stream(RATE_LIMITED),
                    "usage": failed_stream(USAGE_LIMIT),
                }
                (trial / "agent" / "codex-output.jsonl").write_text(texts[outcome])
            return [run]

        patch.object(workbuddy, "run_batch", side_effect=run_batch).start()

    def launch(self, outcomes, k=1):
        self.outcomes = list(outcomes)
        run_ids = tuple(f"wb-code-cx-agg{k}-t/task-a/base{i}" for i in range(1, k + 1))
        with patch("sys.stdout", new_callable=io.StringIO):
            return self.adapter._launch(
                RunRequest("task-a", run_ids, 60), Install(self.root, "base", {})
            )

    def test_only_the_trials_that_stopped_on_429_are_rolled_again(self):
        trials = self.launch(["429", "ok", "ok"], k=2)
        self.assertEqual(self.batches, [2, 1])
        self.assertEqual(len(trials), 2)
        rollouts = self.adapter.spec.runs_root / "rollouts"
        left = sorted(t.name for t in rollouts.rglob("task-a__*"))
        self.assertEqual(left, ["task-a__t11", "task-a__t20"])
        for trial in trials:
            self.assertIsNone(codex_substrate.run_failure(trial / "agent" / "codex-output.jsonl"))

    def test_a_429_that_persists_is_a_connection_failure(self):
        with self.assertRaises(CodexRateLimited):
            self.launch(["429"] * 4)
        self.assertEqual(self.batches, [1, 1, 1, 1])

    def test_the_usage_limit_stops_the_batch(self):
        with self.assertRaises(CodexUsageLimit):
            self.launch(["usage"])
        self.assertEqual(self.batches, [1])


class UsageLimitStopTests(TempDirTest):
    def test_the_run_stops_and_resumes_with_the_tasks_it_did_not_finish(self):
        adapter = DemoAdapter(self.root)
        proposer = DemoProposer(adapter.spec.channels["proposer"])
        execute = adapter.execute

        def out_of_usage(task_id, harness_dir, run_id, **kwargs):
            if task_id == "ledger-2":
                raise CodexUsageLimit(USAGE_LIMIT)
            return execute(task_id, harness_dir, run_id, **kwargs)

        def run_domain(stop=None):
            return schedule.run_domain(
                "accounting",
                adapter.tasks(),
                1,
                adapter=adapter,
                prop=proposer,
                runs_root=adapter.spec.runs_root,
                library_root=self.root / "library",
                stop=stop,
            )

        stop = threading.Event()
        with (
            patch.object(adapter, "execute", side_effect=out_of_usage),
            patch("sys.stdout", new_callable=io.StringIO),
        ):
            with self.assertRaises(CodexUsageLimit):
                run_domain(stop)
        self.assertTrue(stop.is_set())
        summaries = adapter.spec.runs_root / "demo-oc-agg1"
        self.assertEqual(
            json.loads((summaries / "ledger-1/summary.json").read_text())["status"], "ok"
        )
        self.assertFalse((summaries / "ledger-2/summary.json").exists())
        with patch("sys.stdout", new_callable=io.StringIO):
            self.assertEqual(run_domain(stop)["tasks"], [])
            resumed = run_domain()
        self.assertEqual([row["status"] for row in resumed["tasks"]], ["skipped", "ok"])

    def test_the_command_reports_the_reset_time_and_exits_3(self):
        with (
            patch.object(run_sweep, "_main", side_effect=CodexUsageLimit(USAGE_LIMIT)),
            patch("sys.stderr", new_callable=io.StringIO) as err,
        ):
            code = run_sweep.main(["--bench", "jb", "--substrate", "cx", "--k", "1"])
        self.assertEqual(code, run_sweep.USAGE_LIMIT_EXIT)
        self.assertIn(USAGE_LIMIT, err.getvalue())
        self.assertIn("not scored", err.getvalue())
        self.assertIn("again after Oct 22, 2026 3:00 PM", err.getvalue())


if __name__ == "__main__":
    unittest.main()
