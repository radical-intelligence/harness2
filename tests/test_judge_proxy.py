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

"""The judge server: a chat-completions request in, one `claude -p` command out, the reply back.

`claude` itself is replaced by a fake that records the call.
"""

import json
import os
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest.mock import patch

from harness2.cli import judge_proxy

KEY = "test-key"
PNG = "iVBORw0KGgo="
SYSTEM = "You are an evaluation judge. You must return valid JSON only."
VERDICT = '{"criteria_results": [], "rubric_passed": true, "overall_reasoning": "ok"}'


def stream(*events: dict) -> str:
    return "".join(json.dumps(e) + "\n" for e in events)


INIT = {"type": "system", "subtype": "init", "model": "claude-test-1", "tools": []}
SUCCESS = {
    "type": "result",
    "subtype": "success",
    "is_error": False,
    "result": VERDICT,
    "stop_reason": "end_turn",
    "usage": {
        "input_tokens": 10,
        "cache_creation_input_tokens": 100,
        "cache_read_input_tokens": 5,
        "output_tokens": 7,
    },
}


def judge_request(**extra) -> dict:
    """A request shaped like the JobBench judge's vision request."""
    return {
        "model": "claude-test-1",
        "messages": [
            {"role": "system", "content": SYSTEM},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Evaluate the rubric."},
                    {"type": "text", "text": "Image 1: chart.png"},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{PNG}"}},
                ],
            },
        ],
        "max_completion_tokens": 65536,
        "temperature": 0.0,
        "reasoning_effort": "low",
        **extra,
    }


class FakeClaude:
    """Stands in for run_claude: records each call and returns a canned result."""

    def __init__(self, result=(0, stream(INIT, SUCCESS), "")):
        self.result, self.calls = result, []

    def __call__(self, cmd, stdin, cwd, files, *, timeout, gone):
        system = Path(cmd[cmd.index("--system-prompt-file") + 1]).read_text(encoding="utf-8")
        self.calls.append(
            {"cmd": cmd, "stdin": json.loads(stdin), "system": system, "cwd_files": os.listdir(cwd)}
        )
        return self.result


class JudgeProxyServerTests(unittest.TestCase):
    def setUp(self):
        workdir = tempfile.TemporaryDirectory()
        self.addCleanup(workdir.cleanup)
        self.proxy = judge_proxy.Proxy(
            "/usr/local/bin/claude", KEY, max_calls=2, timeout=30, workdir=Path(workdir.name)
        )
        server = judge_proxy.ThreadingHTTPServer(("127.0.0.1", 0), judge_proxy.Handler)
        server.daemon_threads = True
        server.proxy = self.proxy
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        self.url = f"http://127.0.0.1:{server.server_address[1]}/v1/chat/completions"
        self.fake = FakeClaude()
        patch.object(judge_proxy, "run_claude", self.fake).start()
        self.addCleanup(patch.stopall)

    def post(self, body, key=KEY) -> tuple[int, dict]:
        request = urllib.request.Request(
            self.url,
            data=json.dumps(body).encode(),
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=10) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    def test_a_request_becomes_one_claude_call_without_tools_or_settings(self):
        status, _ = self.post(judge_request())
        self.assertEqual(status, 200)
        self.assertEqual(len(self.fake.calls), 1)
        cmd = self.fake.calls[0]["cmd"]
        self.assertEqual(cmd[:2], ["/usr/local/bin/claude", "-p"])
        for flag, value in (
            ("--input-format", "stream-json"),
            ("--output-format", "stream-json"),
            ("--tools", ""),
            ("--setting-sources", ""),
        ):
            self.assertEqual(cmd[cmd.index(flag) + 1], value, flag)
        for flag in (
            "--strict-mcp-config",
            "--safe-mode",
            "--no-session-persistence",
            "--model=claude-test-1",
            "--effort=low",
        ):
            self.assertIn(flag, cmd)
        self.assertNotIn("--dangerously-skip-permissions", cmd)
        self.assertEqual(self.fake.calls[0]["cwd_files"], [])

    def test_system_prompt_and_images_reach_claude(self):
        self.post(judge_request())
        call = self.fake.calls[0]
        self.assertEqual(call["system"], SYSTEM)
        self.assertEqual(call["stdin"]["type"], "user")
        self.assertEqual(
            call["stdin"]["message"]["content"],
            [
                {"type": "text", "text": "Evaluate the rubric."},
                {"type": "text", "text": "Image 1: chart.png"},
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": "image/png", "data": PNG},
                },
            ],
        )

    def test_the_reply_is_a_chat_completion(self):
        status, body = self.post(judge_request())
        self.assertEqual(status, 200)
        self.assertEqual(body["object"], "chat.completion")
        self.assertEqual(body["model"], "claude-test-1")
        self.assertEqual(
            body["choices"][0],
            {
                "index": 0,
                "message": {"role": "assistant", "content": VERDICT},
                "finish_reason": "stop",
            },
        )
        self.assertEqual(
            body["usage"], {"prompt_tokens": 115, "completion_tokens": 7, "total_tokens": 122}
        )

    def test_a_wrong_key_is_refused_before_claude_runs(self):
        status, body = self.post(judge_request(), key="other")
        self.assertEqual(status, 401)
        self.assertIn("HARNESS2_JUDGE_API_KEY", body["error"]["message"])
        self.assertEqual(self.fake.calls, [])

    def test_requests_the_server_cannot_answer_are_refused(self):
        image_link = judge_request()
        image_link["messages"][1]["content"][2]["image_url"]["url"] = "https://example.com/a.png"
        assistant = judge_request()
        assistant["messages"].append({"role": "assistant", "content": "earlier answer"})
        for name, body in (
            ("streaming", judge_request(stream=True)),
            ("image link", image_link),
            ("assistant turn", assistant),
            ("no user message", {"messages": [{"role": "system", "content": SYSTEM}]}),
        ):
            status, reply = self.post(body)
            self.assertEqual(status, 400, name)
            self.assertEqual(reply["error"]["type"], "invalid_request_error", name)
        self.assertEqual(self.fake.calls, [])

    def test_claude_failures_become_http_errors(self):
        limited = dict(SUCCESS, is_error=True, api_error_status=429, result="usage limit reached")
        for result, status, text in (
            ((0, stream(INIT, limited), ""), 429, "usage limit reached"),
            ((1, "", "not logged in"), 502, "not logged in"),
            ((None, "", ""), 504, "stopped"),
        ):
            self.fake.result = result
            got, body = self.post(judge_request())
            self.assertEqual(got, status)
            self.assertIn(text, body["error"]["message"])

    def test_an_effort_claude_does_not_accept_is_left_out(self):
        _, _, options = judge_proxy.build_call(judge_request(reasoning_effort="minimal"))
        self.assertEqual(options, ["--model=claude-test-1"])

    def test_a_waiting_request_is_dropped_when_its_client_leaves(self):
        proxy = judge_proxy.Proxy("claude", KEY, max_calls=1, timeout=30, workdir=Path("."))
        proxy.slots.acquire()
        status, _ = proxy.answer(judge_request(), gone=lambda: True)
        self.assertEqual(status, 499)
        self.assertEqual(self.fake.calls, [])


class RunClaudeTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.files = Path(tmp.name)

    def test_the_input_is_sent_on_stdin_and_the_output_read_back(self):
        cmd = [sys.executable, "-c", "import sys; sys.stdout.write(sys.stdin.read().upper())"]
        got = judge_proxy.run_claude(
            cmd, b"abc\n", self.files, self.files, timeout=30, gone=lambda: False
        )
        self.assertEqual(got, (0, "ABC\n", ""))

    def test_claude_is_stopped_after_the_timeout(self):
        cmd = [sys.executable, "-c", "import time; time.sleep(30)"]
        started = time.monotonic()
        got = judge_proxy.run_claude(
            cmd, b"", self.files, self.files, timeout=0.2, gone=lambda: False
        )
        self.assertIsNone(got[0])
        self.assertLess(time.monotonic() - started, 10)

    def test_settings_that_bypass_the_login_are_removed(self):
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant", "CLAUDE_CODE_USE_VERTEX": "1"}):
            env = judge_proxy.claude_env()
        self.assertNotIn("ANTHROPIC_API_KEY", env)
        self.assertNotIn("CLAUDE_CODE_USE_VERTEX", env)
        self.assertEqual(env.get("HOME"), os.environ.get("HOME"))


if __name__ == "__main__":
    unittest.main()
