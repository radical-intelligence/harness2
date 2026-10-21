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

"""The improvement agents on the claude_cli channel: settings, checks and sessions."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from harness2 import config
from harness2.core import proposer
from harness2.core.proposer import Proposer
from harness2.core.spec import ChannelSpec, resolve_channel_model

PROVIDER_ENV = {
    "ANTHROPIC_API_KEY": "sk-ant-test",
    "CLAUDE_CODE_USE_VERTEX": "1",
    "ANTHROPIC_VERTEX_PROJECT_ID": "project-test",
    "CLOUD_ML_REGION": "us-east5",
    "GOOGLE_CLOUD_PROJECT": "project-test",
    "GOOGLE_CLOUD_LOCATION": "global",
}


class ClaudeCliSettingsTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(patch.stopall)
        patch.multiple(
            config,
            CODEX_AUTH="",
            SOLVER_MODEL_OC="openrouter/anthropic/claude-test",
            SOLVER_MODEL_CX="gpt-test",
            IMPROVER_CHANNEL="claude_cli",
            IMPROVER_MODEL="sonnet",
            DEFAULT_PROPOSER_CHANNEL="claude_cli",
        ).start()
        patch.dict(config.PROPOSER_MODEL_BY_CHANNEL, {"claude_cli": "sonnet"}).start()
        patch.dict(os.environ, {"HARNESS2_SOLVER_MODEL": config.SOLVER_MODEL_OC}).start()

    def test_claude_model_names_pass_for_both_substrates(self):
        for model in ("sonnet", "opus", "claude-sonnet-4-5-20250929"):
            patch.object(config, "IMPROVER_MODEL", model).start()
            for substrate in ("oc", "cx"):
                with self.subTest(model=model, substrate=substrate):
                    config.validate_models(substrate=substrate)

    def test_provider_model_ids_are_refused(self):
        patch.object(config, "IMPROVER_MODEL", "openrouter/anthropic/claude-test").start()
        with self.assertRaisesRegex(config.ConfigError, "Claude model name such as sonnet"):
            config.validate_models()

    def test_the_model_is_required(self):
        patch.object(config, "IMPROVER_MODEL", "").start()
        with self.assertRaisesRegex(config.ConfigError, "HARNESS2_IMPROVER_MODEL is required"):
            config.validate_models()

    def test_an_unknown_channel_is_refused(self):
        patch.multiple(config, IMPROVER_CHANNEL="claude", DEFAULT_PROPOSER_CHANNEL="claude").start()
        with self.assertRaisesRegex(config.ConfigError, "HARNESS2_IMPROVER_CHANNEL"):
            config.validate_models()
        with self.assertRaisesRegex(ValueError, "HARNESS2_IMPROVER_CHANNEL"):
            resolve_channel_model(None, None)

    def test_opencode_still_needs_provider_model_ids(self):
        patch.multiple(
            config, IMPROVER_CHANNEL="opencode", DEFAULT_PROPOSER_CHANNEL="opencode"
        ).start()
        with self.assertRaisesRegex(config.ConfigError, "provider/model"):
            config.validate_models()

    def test_the_benchmarks_resolve_the_configured_channel(self):
        spec = resolve_channel_model(None, None, effort="high")
        self.assertEqual((spec.kind, spec.model), ("claude_cli", "sonnet"))
        with self.assertRaisesRegex(ValueError, "Claude model name"):
            resolve_channel_model("claude_cli", "anthropic/claude-test")


class ClaudeCliSessionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.addCleanup(patch.stopall)
        self.root = Path(temporary.name)
        patch.dict(os.environ, PROVIDER_ENV).start()
        patch.object(proposer.shutil, "which", return_value="/usr/local/bin/claude").start()

    def fake_claude(self, payload, returncode=0):
        return patch.object(
            proposer.subprocess,
            "run",
            return_value=Mock(returncode=returncode, stdout=json.dumps(payload), stderr=""),
        ).start()

    def assert_no_provider_settings(self, env):
        for name in PROVIDER_ENV:
            self.assertNotIn(name, env)

    def test_a_session_runs_claude_print_on_its_own_login(self):
        run = self.fake_claude(
            {"type": "result", "subtype": "success", "result": "edited", "total_cost_usd": 0.5}
        )
        context = self.root / "context"
        context.mkdir()
        (context / "notes.md").write_text("evidence")
        audit = self.root / "audit"
        session = Proposer(ChannelSpec("claude_cli", "sonnet", effort="high"), audit_dir=audit)
        reply = session.call("Improve the harness.", context, "inv1.a1", system="Be careful.")

        self.assertEqual((reply.text, reply.cost_usd, reply.channel), ("edited", 0.5, "claude_cli"))
        cmd = run.call_args.args[0]
        self.assertEqual(cmd[:2], ["/usr/local/bin/claude", "-p"])
        self.assertEqual(cmd[cmd.index("--model") + 1], "sonnet")
        self.assertEqual(cmd[cmd.index("--permission-mode") + 1], "acceptEdits")
        self.assertEqual(cmd[cmd.index("--setting-sources") + 1], "project,local")
        self.assertIn("--strict-mcp-config", cmd)
        self.assertIn("--no-session-persistence", cmd)
        self.assertNotIn("--dangerously-skip-permissions", cmd)
        self.assertEqual(run.call_args.kwargs["input"], "Improve the harness.")
        self.assert_no_provider_settings(run.call_args.kwargs["env"])

        written = sorted(p.name for p in audit.iterdir())
        self.assertEqual(len(written), 4)
        self.assertTrue(any(name.endswith(".mission.md") for name in written))
        meta = json.loads(next(audit.glob("*.meta.json")).read_text())
        self.assertEqual(
            (meta["channel"], meta["model"], meta["ok"]), ("claude_cli", "sonnet", True)
        )
        response = next(audit.glob("*.response.md")).read_text()
        self.assertEqual(response, "edited")

    def test_a_cli_error_is_not_a_reply(self):
        self.fake_claude({"type": "result", "subtype": "success", "is_error": True, "result": "x"})
        session = Proposer(ChannelSpec("claude_cli", "sonnet"), attempts=1)
        with patch.object(proposer.time, "sleep"):
            with self.assertRaisesRegex(proposer.ProposerError, "CLI error"):
                session.call("Improve the harness.", None, "inv1.a1")

    def test_the_login_check(self):
        run = self.fake_claude({"loggedIn": True, "authMethod": "claude.ai"})
        self.assertIsNone(proposer.claude_login_error())
        self.assertEqual(run.call_args.args[0][1:], ["auth", "status", "--json"])
        self.assert_no_provider_settings(run.call_args.kwargs["env"])

        self.fake_claude({"loggedIn": False, "authMethod": "none"}, returncode=1)
        self.assertIn("claude auth login", proposer.claude_login_error())

        patch.object(proposer.shutil, "which", return_value=None).start()
        self.assertIn("on PATH", proposer.claude_login_error())

    def test_opencode_sessions_keep_their_environment(self):
        env = Proposer(ChannelSpec("opencode", "openrouter/anthropic/claude-test")).env()
        self.assertNotIn("ANTHROPIC_API_KEY", env)
        self.assertEqual(env["GOOGLE_CLOUD_PROJECT"], "project-test")
        self.assertIn("GOOGLE_VERTEX_PROJECT", env)


if __name__ == "__main__":
    unittest.main()
