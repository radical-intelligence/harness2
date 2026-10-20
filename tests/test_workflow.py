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

"""Focused regressions for the paper's execution, evidence, and adoption contracts."""

import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from harness2 import config, install, preflight
from harness2.benchmarks import jobbench
from harness2.core import context, judge, recursion, schedule, tree
from harness2.core.library import Library
from harness2.core.proposer import Proposer, ProposerError, Reply
from harness2.core.spec import resolve_channel_model
from harness2.demo import BASE_SCRIPT, DemoAdapter, DemoProposer
from harness2.substrate import codex as codex_substrate


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.adapter = DemoAdapter(self.root)
        self.proposer = DemoProposer(self.adapter.spec.channels["proposer"])

    def run_stream(self, k=2, proposer=None):
        return schedule.run_domain(
            "accounting",
            self.adapter.tasks(),
            k,
            adapter=self.adapter,
            prop=proposer or self.proposer,
            runs_root=self.adapter.spec.runs_root,
            library_root=self.root / "library" / str(k),
        )

    def test_recursion_budget_and_resume(self):
        with patch.dict(config.PROPOSER_MODEL_BY_CHANNEL, {"opencode": "", "claude_cli": ""}):
            unresolved = resolve_channel_model(None, None)
            self.assertEqual(unresolved.model, "")
            with self.assertRaises(ValueError):
                Proposer(unresolved)
            with self.assertRaises(ValueError):
                resolve_channel_model("opencode", "")
        for k in (1, 2):
            with self.subTest(k=k):
                result = self.run_stream(k)
                self.assertEqual((result["done"], result["failed"]), (2, 0))
                for task_id in self.adapter.tasks():
                    summary = json.loads(
                        (
                            self.adapter.spec.runs_root
                            / f"demo-oc-agg{k}"
                            / task_id
                            / "summary.json"
                        ).read_text()
                    )
                    for mode in ("agg", "seq"):
                        ledger = summary["modes"][mode]["ledger"]
                        self.assertEqual((ledger["R"], ledger["P"]), (4 * k, 3 * k))
                        selection = summary["selections"][mode]
                        expected = "Total: 85.15\n" if task_id == "ledger-1" else "Total: 20.25\n"
                        self.assertEqual(
                            self.adapter.deliverable_text(Path(selection["chosen_run_dir"])),
                            expected,
                        )
                executions, calls = self.adapter.execution_count, len(self.proposer._calls)
                resumed = self.run_stream(k)
                self.assertEqual(
                    (self.adapter.execution_count, len(self.proposer._calls)), (executions, calls)
                )
                self.assertEqual(
                    [row["status"] for row in resumed["tasks"]], ["skipped", "skipped"]
                )
        self.assertFalse(
            any(
                reply.label.endswith(".agg1") or reply.label.endswith(".seq1")
                for reply in self.proposer._calls
                if reply.label.startswith("select.")
            )
        )
        work = self.adapter.spec.runs_root / "demo-oc-seq2" / "ledger-1"
        summary_path = self.adapter.spec.runs_root / "demo-oc-agg2/ledger-1/summary.json"
        chosen = json.loads(summary_path.read_text())["selections"]["seq"]
        (work / "harnesses" / chosen["chosen_label"] / "scripts/summarize.py").write_text(
            "changed after selection\n"
        )
        repaired = self.run_stream()
        self.assertEqual([row["status"] for row in repaired["tasks"]], ["ok", "skipped"])
        chosen = json.loads(summary_path.read_text())["selections"]["agg"]
        (Path(chosen["chosen_run_dir"]) / "answer.txt").unlink()
        repaired = self.run_stream()
        self.assertEqual([row["status"] for row in repaired["tasks"]], ["ok", "skipped"])

    def test_complete_evidence_and_library_prefix(self):
        (self.adapter.task_files("ledger-1") / "reference_answer.txt").write_text('"pass_rate": 1')
        self.run_stream()
        work = self.adapter.spec.runs_root / "demo-oc-seq2" / "ledger-1"
        reflection = work / "reflection/seq2"
        required = [
            "source/ledger.csv",
            "rollouts/base1.md",
            "rollouts/c2.md",
            "pool/inv1/harness.diff",
            "decisions/c1.md",
            "adoption.json",
            "base_harness/scripts/summarize.py",
            "harness/scripts/summarize.py",
        ]
        self.assertEqual([name for name in required if not (reflection / name).is_file()], [])
        self.assertTrue((work / "context/inv2/pool/c1/rollout.md").is_file())
        self.assertEqual(list(work.rglob("reference_answer.txt")), [])
        next_context = work.parent / "ledger-2/context/inv1/library/harnesses"
        self.assertEqual([path.name for path in next_context.iterdir()], ["ledger-1"])
        with self.assertRaises(ValueError):
            context.assert_tree_has_no_grader_verdicts({"EXPERIENCE.md": '"pass_rate": 1'})
        previous_sessions = []

        def run_opencode(command, **kwargs):
            database = Path(kwargs["env"]["XDG_DATA_HOME"]) / "opencode.db"
            previous_sessions.append(database.exists())
            database.parent.mkdir(parents=True, exist_ok=True)
            database.write_text("private evidence from this session")
            return Mock(stdout='{"type":"text","part":{"text":"done"}}', stderr="", returncode=0)

        proposer = Proposer(self.proposer.spec)
        with (
            patch("harness2.core.proposer._opencode_dir", return_value=self.root),
            patch("harness2.core.proposer.subprocess.run", side_effect=run_opencode),
        ):
            for index in range(2):
                proposer.call("Read this task only", None, f"isolation{index}")
        self.assertEqual(previous_sessions, [False, False])

    def test_invalid_proposal_retries_with_feedback_and_accumulates_cost(self):
        class RepairProposer(DemoProposer):
            feedback = ""

            def call(self, mission, context_dir, label, **kwargs):
                script = Path(context_dir) / "harness/scripts/summarize.py"
                if label.endswith(".a1"):
                    script.write_text("def broken(:\n")
                    return Reply(
                        label,
                        "invalid edit",
                        True,
                        cost_usd=1,
                        attempts=1,
                        workspace=script.parent.parent,
                    )
                self.feedback = mission
                script.write_text(BASE_SCRIPT)
                reply = super().call(mission, context_dir, label, **kwargs)
                reply.cost_usd = 2
                return reply

        proposer = RepairProposer(self.proposer.spec)
        stage = recursion._run_stage(
            label="inv1",
            dim="INV",
            job="Repair the script",
            adapter=self.adapter,
            prop=proposer,
            ctx_dir=self.root / "context",
            task_id="ledger-1",
            rollouts=[],
            fallback=tree.read_tree(self.adapter.substrate.surface(), self.adapter.base_harness()),
            harness_dir=self.adapter.base_harness(),
            library_dir=None,
        )
        self.assertEqual((stage.fell_back, stage.attempts, stage.usd), (False, 2, 3))
        self.assertIn("PREVIOUS ATTEMPT WAS REJECTED", proposer.feedback)

    def test_no_entry_stays_out_of_library_on_resume(self):
        class NoExperience(DemoProposer):
            def call(self, mission, context_dir, label, **kwargs):
                if label.startswith("experience."):
                    if label.endswith("seq1"):
                        raise ProposerError("reflection failed", attempts=2, cost_usd=2)
                    return Reply(label, "NO_ENTRY", True, cost_usd=1)
                return super().call(mission, context_dir, label, **kwargs)

        proposer = NoExperience(self.proposer.spec)
        self.run_stream(k=1, proposer=proposer)
        self.run_stream(k=1, proposer=proposer)
        for mode in ("agg", "seq"):
            library = Library(
                self.root / "library/1" / mode, "accounting", self.adapter.substrate.surface()
            )
            self.assertEqual(library.entries(), [])
        summary_path = self.adapter.spec.runs_root / "demo-oc-agg1/ledger-1/summary.json"
        selections = json.loads(summary_path.read_text())["selections"]
        self.assertEqual([selections[mode]["usd"] for mode in ("agg", "seq")], [1, 2])

    def test_failed_selection_is_never_judged(self):
        class FailedSelector(DemoProposer):
            def call(self, mission, context_dir, label, **kwargs):
                if label.startswith("select."):
                    raise ProposerError("selection failed", attempts=2, cost_usd=3)
                return super().call(mission, context_dir, label, **kwargs)

        self.run_stream(proposer=FailedSelector(self.proposer.spec))
        summary_path = self.adapter.spec.runs_root / "demo-oc-agg2/ledger-1/summary.json"
        summary = json.loads(summary_path.read_text())
        self.assertEqual([summary["selections"][mode]["usd"] for mode in ("agg", "seq")], [3, 3])
        self.adapter.score = Mock(return_value={"n_passed": 1, "n_criteria": 1})
        rows = judge.judge_selections(
            self.adapter,
            list(summary["selections"].values()),
        )
        self.adapter.score.assert_not_called()
        self.assertEqual((rows[0].graded, rows[0].n_passed), (False, None))


class WorkBuddySetupTests(unittest.TestCase):
    VERTEX_SOLVER = {
        "name": "google/gemini-x",
        "protocols": ["openai"],
        "backend_url_env": "HARNESS2_WB_VERTEX_URL",
        "backend_key_command": "gcloud auth application-default print-access-token",
        "backend_key_ttl": 1800,
        "thought_signature_passback": True,
        "params": {"max_output_tokens": 64000, "extra_body": {"reasoning_effort": "high"}},
    }

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.repo = Path(temporary.name)
        self.addCleanup(patch.stopall)
        patch.object(config, "wb_repo", return_value=self.repo).start()

    def configure(self, solver, judge, project="", solver_url="", judge_url=""):
        patch.multiple(
            config,
            SOLVER_MODEL_OC=solver,
            JUDGE_MODEL=judge,
            VERTEX_PROJECT=project,
            SOLVER_BASE_URL=solver_url,
            JUDGE_BASE_URL=judge_url,
        ).start()

    def route(self, role):
        path = self.repo / "configs/models/harness2" / f"{role}.yaml"
        return json.loads(path.read_text())["model"] if path.is_file() else None

    def test_no_endpoint_and_no_vertex_gemini_skips_both_routes(self):
        self.configure("openai/gpt-x", "claude-z")
        with patch("sys.stdout", new_callable=io.StringIO) as out:
            install.workbuddy_models()
        self.assertEqual((self.route("solver"), self.route("judge")), (None, None))
        self.assertIn("Skip WorkBuddy solver route", out.getvalue())
        self.assertIn("Skip WorkBuddy judge route", out.getvalue())

    def test_endpoints_name_variables_and_never_hold_keys(self):
        self.configure(
            "openai/gpt-x", "gpt-judge", solver_url="http://127.0.0.1:1/v1", judge_url="http://j/v1"
        )
        with patch.dict("os.environ", {"HARNESS2_SOLVER_API_KEY": "sk-secret-value"}):
            install.workbuddy_models()
        self.assertEqual(
            self.route("solver"),
            {
                "name": "gpt-x",
                "protocols": ["openai"],
                "backend_url_env": "HARNESS2_SOLVER_BASE_URL",
                "backend_key_env": "HARNESS2_SOLVER_API_KEY",
            },
        )
        self.assertEqual(self.route("judge")["name"], "gpt-judge")
        self.assertEqual(self.route("judge")["backend_key_env"], "HARNESS2_JUDGE_API_KEY")
        written = "".join(p.read_text() for p in self.repo.rglob("*.yaml"))
        self.assertNotIn("sk-secret-value", written)
        self.assertNotIn("http://127.0.0.1:1/v1", written)

    def test_vertex_gemini_routes_are_unchanged(self):
        self.configure("google-vertex/gemini-x", "gemini-y", project="demo-project")
        install.workbuddy_models()
        self.assertEqual(self.route("solver"), self.VERTEX_SOLVER)
        self.assertEqual(self.route("judge")["name"], "google/gemini-y")

    def test_endpoint_wins_over_vertex(self):
        self.configure(
            "google-vertex/gemini-x", "gemini-y", project="demo-project", solver_url="http://s/v1"
        )
        install.workbuddy_models()
        self.assertEqual(self.route("solver")["backend_url_env"], "HARNESS2_SOLVER_BASE_URL")
        self.assertEqual(self.route("solver")["name"], "gemini-x")
        self.assertEqual(self.route("judge")["backend_url_env"], "HARNESS2_WB_VERTEX_URL")

    def test_check_needs_the_judge_route_only_for_web(self):
        self.configure("openai/gpt-x", "", solver_url="http://s/v1")
        install.workbuddy_models()
        self.assertEqual(preflight.workbuddy_route_errors(self.repo, web=False), [])
        errors = preflight.workbuddy_route_errors(self.repo, web=True)
        self.assertEqual(len(errors), 1)
        self.assertIn("judge has no model to use", errors[0])

    def test_check_reports_a_route_written_for_other_settings(self):
        self.configure("google-vertex/gemini-x", "gemini-y", project="demo-project")
        install.workbuddy_models()
        patch.object(config, "SOLVER_BASE_URL", "http://s/v1").start()
        errors = preflight.workbuddy_route_errors(self.repo, web=False)
        self.assertEqual(
            errors, ["WorkBuddy solver configuration is stale; rerun harness2-setup wb"]
        )


class JobBenchJudgeTokenTests(unittest.TestCase):
    VERTEX = "https://aiplatform.googleapis.com/v1/projects/p/locations/global/endpoints/openapi"

    def setUp(self):
        self.addCleanup(patch.stopall)
        self.gcloud = patch.object(
            jobbench.subprocess, "run", return_value=Mock(stdout="ya29.google-token\n")
        ).start()

    def configure(self, base, key=""):
        patch.multiple(config, JB_JUDGE_API_BASE=base, JB_JUDGE_API_KEY=key).start()

    def test_no_gcloud_token_for_a_non_google_endpoint(self):
        self.configure("https://openrouter.ai/api/v1")
        token, why = jobbench.judge_token()
        self.assertEqual(token, "")
        self.assertIn("HARNESS2_JUDGE_API_KEY is required", why)
        self.gcloud.assert_not_called()

    def test_gcloud_token_only_for_a_google_endpoint(self):
        self.configure(self.VERTEX)
        self.assertEqual(jobbench.judge_token(), ("ya29.google-token", ""))
        self.gcloud.assert_called_once()

    def test_a_configured_key_is_always_used(self):
        self.configure("https://openrouter.ai/api/v1", key="sk-or-key")
        self.assertEqual(jobbench.judge_token(), ("sk-or-key", ""))
        self.gcloud.assert_not_called()

    def test_lookalike_hosts_are_not_google(self):
        for url in (
            "https://evilgoogleapis.com/v1",
            "https://googleapis.com.example.org/v1",
            "http://localhost:11434/v1",
            "",
        ):
            self.assertFalse(config.is_google_endpoint(url), url)
        self.assertTrue(config.is_google_endpoint(self.VERTEX))
        self.assertTrue(config.is_google_endpoint("https://us-east5-aiplatform.googleapis.com/v1"))


class CodexLoginTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.addCleanup(patch.stopall)
        self.home = Path(temporary.name) / "codex-home"
        dist = Path(temporary.name) / "dist"
        dist.mkdir()
        (dist / "codex").write_text("")
        patch.multiple(
            config,
            CODEX_AUTH="chatgpt",
            CODEX_HOME_DIR=self.home,
            CODEX_DIST=str(dist),
            SOLVER_MODEL_CX="gpt-test",
        ).start()

    def log_in(self, auth_mode="chatgpt", refresh_token="rt-secret"):
        self.home.mkdir(parents=True, exist_ok=True)
        tokens = {"refresh_token": refresh_token} if refresh_token else {}
        (self.home / "auth.json").write_text(json.dumps({"auth_mode": auth_mode, "tokens": tokens}))

    def test_setup_writes_a_login_config_without_an_endpoint(self):
        with patch("sys.stdout", new_callable=io.StringIO) as out:
            install.configure_codex()
        self.assertEqual(
            (self.home / "config.toml").read_text(),
            'model = "gpt-test"\ncli_auth_credentials_store = "file"\n',
        )
        self.assertFalse((self.home / "auth.json").exists())
        self.assertIn(f"CODEX_HOME={self.home}", out.getvalue())
        self.assertIn("codex login", out.getvalue())

    def test_login_check(self):
        self.assertIn("not logged in", codex_substrate.login_error())
        self.log_in(auth_mode="apikey")
        self.assertIn("not a ChatGPT login", codex_substrate.login_error())
        self.log_in(refresh_token="")
        self.assertIn("not a ChatGPT login", codex_substrate.login_error())
        self.log_in()
        self.assertIsNone(codex_substrate.login_error())

    def test_login_folder_inside_the_repository_is_refused(self):
        patch.object(config, "CODEX_HOME_DIR", config.ROOT / "dependencies" / "codex").start()
        self.assertIn("inside the repository", codex_substrate.login_error())

    def test_an_endpoint_config_is_stale_for_the_login(self):
        self.log_in()
        self.home.joinpath("config.toml").write_text(
            'model = "gpt-test"\nmodel_provider = "harness2"\n'
            '[model_providers.harness2]\nbase_url = "http://x/v1"\n'
        )
        self.assertIn("built-in OpenAI provider", install.codex_config_error())
        install.configure_codex()
        self.assertIsNone(install.codex_config_error())

    def test_preflight_uses_the_login_and_refuses_workbuddy(self):
        self.log_in()
        codex_substrate.CodexSubstrate().preflight()
        with self.assertRaisesRegex(config.ConfigError, "issues/8"):
            codex_substrate.CodexSubstrate().preflight(require_dist=True, require_bridge=True)

    def test_codex_runs_need_the_codex_and_improver_models_only(self):
        patch.multiple(config, SOLVER_MODEL_OC="", IMPROVER_MODEL="openrouter/a/b").start()
        config.validate_models(substrate="cx")
        patch.object(config, "IMPROVER_MODEL", "").start()
        with self.assertRaisesRegex(config.ConfigError, "HARNESS2_IMPROVER_MODEL"):
            config.validate_models(substrate="cx")
        patch.object(config, "CODEX_AUTH", "apikey").start()
        with self.assertRaisesRegex(config.ConfigError, "HARNESS2_CODEX_AUTH"):
            config.validate_models(substrate="cx")

    def test_lab_links_each_run_to_the_one_login(self):
        self.log_in()
        runner = config.ROOT / "third_party/lab_runner/adapt_runs_cx/run_codex_lab.py"
        stub = {
            name: Mock() for name in ("external", "external.tts_lab", "external.tts_lab.config")
        }
        env = {"ADAPT_LABCX_AUTH_FILE": str(self.home / "auth.json")}
        with patch.dict("sys.modules", stub), patch.dict("os.environ", env):
            spec = importlib.util.spec_from_file_location("run_codex_lab_test", runner)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        run_home = self.home.parent / "run-home"
        module.write_codex_home(run_home, "gpt-test", Path("/ws"))
        self.assertTrue((run_home / "auth.json").is_symlink())
        self.assertEqual((run_home / "auth.json").resolve(), (self.home / "auth.json").resolve())
        cfg = (run_home / "config.toml").read_text()
        self.assertIn('model = "gpt-test"', cfg)
        self.assertNotIn("model_provider", cfg)
        self.assertIn('[projects."/ws"]', cfg)


if __name__ == "__main__":
    unittest.main()
