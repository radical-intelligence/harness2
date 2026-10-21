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

"""Serve an OpenAI chat-completions endpoint for the judge, answered by the local `claude` CLI.

Each request runs `claude -p` once, with the logged-in Claude Code account. The request's
system messages become the system prompt and its user messages, including base64 images,
become one user turn. claude runs in an empty folder with no tools, no MCP servers, and no
settings, CLAUDE.md files, skills, hooks or plugins from your configuration.

The server listens on 127.0.0.1 only and requires HARNESS2_JUDGE_API_KEY as a bearer key.
"""

from __future__ import annotations

import argparse
import hmac
import json
import os
import secrets
import select
import shutil
import signal
import socket
import subprocess
import tempfile
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from harness2 import config

HOST = "127.0.0.1"
DEFAULT_PORT = 8642
PATH = "/v1/chat/completions"
# Values `claude --effort` accepts; other reasoning_effort values are ignored.
EFFORTS = ("low", "medium", "high", "xhigh", "max")
# Settings that would make claude use an API key, another endpoint or a cloud provider
# instead of the Claude Code login.
DROP_ENV = (
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_BASE_URL",
    "CLAUDE_CODE_USE_BEDROCK",
    "CLAUDE_CODE_USE_VERTEX",
)
# claude processes still running, stopped when the server exits.
RUNNING: set[subprocess.Popen] = set()


class RequestError(ValueError):
    """A request the server cannot answer; the message is returned to the client."""


def _text(content) -> str:
    """The text of a system message, given as a string or a list of text parts."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(str(p.get("text", "")) for p in content if isinstance(p, dict))
    raise RequestError("message content must be a string or a list of parts")


def _image(part: dict) -> dict:
    """An OpenAI image_url part with a base64 data URL, as a claude image block."""
    image = part.get("image_url")
    url = image.get("url") if isinstance(image, dict) else image
    header, sep, data = str(url or "").partition(",")
    media_type, _, encoding = header.removeprefix("data:").partition(";")
    if not header.startswith("data:") or not sep or encoding != "base64":
        raise RequestError("images must be base64 data URLs (data:<type>;base64,...)")
    return {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": data}}


def _blocks(content) -> list[dict]:
    """A user message's content as claude content blocks."""
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    if not isinstance(content, list):
        raise RequestError("message content must be a string or a list of parts")
    out = []
    for part in content:
        kind = part.get("type") if isinstance(part, dict) else None
        if kind == "text":
            out.append({"type": "text", "text": str(part.get("text", ""))})
        elif kind == "image_url":
            out.append(_image(part))
        else:
            raise RequestError(f"unsupported content part type {kind!r}")
    return out


def build_call(body: dict) -> tuple[str, dict, list[str]]:
    """(system prompt, claude's stream-json user message, extra claude options) for a request."""
    if not isinstance(body, dict):
        raise RequestError("the request body must be a JSON object")
    if body.get("stream"):
        raise RequestError("streaming is not supported")
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        raise RequestError("messages must be a non-empty list")
    system, blocks = [], []
    for message in messages:
        role = message.get("role") if isinstance(message, dict) else None
        if role in ("system", "developer"):
            system.append(_text(message.get("content")))
        elif role == "user":
            blocks += _blocks(message.get("content"))
        else:
            raise RequestError(f"unsupported message role {role!r}; send system and user messages")
    if not blocks:
        raise RequestError("the request has no user message")
    options = []
    if body.get("model"):
        options.append(f"--model={body['model']}")
    if body.get("reasoning_effort") in EFFORTS:
        options.append(f"--effort={body['reasoning_effort']}")
    user = {"type": "user", "message": {"role": "user", "content": blocks}}
    return "\n\n".join(system), user, options


def claude_command(cli: str, system_file: Path, options: list[str]) -> list[str]:
    """The `claude -p` command for one call: no tools and none of the user's configuration."""
    return [
        cli,
        "-p",
        "--input-format",
        "stream-json",
        "--output-format",
        "stream-json",
        "--verbose",
        "--system-prompt-file",
        str(system_file),
        "--tools",
        "",
        "--strict-mcp-config",
        "--setting-sources",
        "",
        "--safe-mode",
        "--disable-slash-commands",
        "--no-session-persistence",
        *options,
    ]


def claude_env() -> dict[str, str]:
    return {k: v for k, v in os.environ.items() if k not in DROP_ENV}


def run_claude(cmd, stdin: bytes, cwd: Path, files: Path, *, timeout: float, gone) -> tuple:
    """(exit code, or None when stopped, stdout, stderr). Stops claude on timeout or gone()."""
    (files / "input.jsonl").write_bytes(stdin)
    with (
        (files / "input.jsonl").open("rb") as fin,
        (files / "output.jsonl").open("wb") as fout,
        (files / "error.txt").open("wb") as ferr,
    ):
        proc = subprocess.Popen(
            cmd,
            cwd=cwd,
            env=claude_env(),
            stdin=fin,
            stdout=fout,
            stderr=ferr,
            start_new_session=True,
        )
        RUNNING.add(proc)
        deadline = time.monotonic() + timeout
        code = None
        try:
            while code is None:
                try:
                    code = proc.wait(timeout=0.5)
                except subprocess.TimeoutExpired:
                    if gone() or time.monotonic() > deadline:
                        _stop(proc)
                        break
        finally:
            RUNNING.discard(proc)
    out = (files / "output.jsonl").read_text(encoding="utf-8", errors="replace")
    err = (files / "error.txt").read_text(encoding="utf-8", errors="replace")
    return code, out, err


def _stop(proc: subprocess.Popen) -> None:
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        proc.kill()
    proc.wait()


def parse_output(stdout: str) -> tuple[str, dict | None]:
    """(model claude used, its result event) from claude's stream-json output."""
    model, result = "", None
    for line in stdout.splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if not isinstance(event, dict):
            continue
        if event.get("type") == "system" and event.get("subtype") == "init":
            model = str(event.get("model") or "")
        elif event.get("type") == "result":
            result = event
    return model, result


def reply(result: dict, model: str) -> dict:
    """claude's result event as an OpenAI chat completion."""
    usage = result.get("usage") or {}
    prompt = sum(
        int(usage.get(k) or 0)
        for k in ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")
    )
    completion = int(usage.get("output_tokens") or 0)
    finish = "length" if result.get("stop_reason") == "max_tokens" else "stop"
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": str(result.get("result") or "")},
                "finish_reason": finish,
            }
        ],
        "usage": {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": prompt + completion,
        },
    }


def error(status: int, message: str) -> tuple[int, dict]:
    kind = "invalid_request_error" if status < 500 else "server_error"
    return status, {"error": {"message": message, "type": kind, "code": None}}


class Proxy:
    """Turns one chat-completions request into one `claude -p` call."""

    def __init__(self, cli: str, key: str, *, max_calls: int, timeout: float, workdir: Path):
        self.cli, self.key, self.timeout, self.workdir = cli, key, timeout, workdir
        self.slots = threading.BoundedSemaphore(max_calls)
        self.calls = 0
        self._lock = threading.Lock()

    def authorized(self, header: str | None) -> bool:
        return hmac.compare_digest(str(header or ""), f"Bearer {self.key}")

    def answer(self, body: dict, gone=lambda: False) -> tuple[int, dict]:
        """(HTTP status, JSON body) for one request. gone() says the client has left."""
        try:
            system, user, options = build_call(body)
        except RequestError as exc:
            return error(400, str(exc))
        while not self.slots.acquire(timeout=1):
            if gone():
                return error(499, "the client closed the connection")
        try:
            with self._lock:
                self.calls += 1
                number = self.calls
            images = sum(b["type"] == "image" for b in user["message"]["content"])
            started = time.monotonic()
            with tempfile.TemporaryDirectory(prefix="harness2-judge-call-") as tmp:
                files = Path(tmp)
                (files / "system.txt").write_text(system, encoding="utf-8")
                cmd = claude_command(self.cli, files / "system.txt", options)
                stdin = (json.dumps(user) + "\n").encode("utf-8")
                code, stdout, stderr = run_claude(
                    cmd, stdin, self.workdir, files, timeout=self.timeout, gone=gone
                )
            status, payload = self._result(code, stdout, stderr, body)
            detail = payload["model"] if status == 200 else payload["error"]["message"]
            seconds = time.monotonic() - started
            print(f"call {number}: {status} images={images} {seconds:.0f}s {detail}", flush=True)
            return status, payload
        finally:
            self.slots.release()

    def _result(self, code, stdout: str, stderr: str, body: dict) -> tuple[int, dict]:
        if code is None:
            return error(504, "claude was stopped: the client left or the call timed out")
        model, result = parse_output(stdout)
        if result is None:
            return error(502, f"claude exited {code} without a result: {stderr.strip()[-500:]}")
        if result.get("is_error") or result.get("subtype") != "success":
            status = result.get("api_error_status")
            status = status if isinstance(status, int) and 400 <= status < 600 else 502
            detail = result.get("result") or result.get("subtype") or "unknown error"
            return error(status, f"claude failed: {str(detail)[:500]}")
        return 200, reply(result, model or str(body.get("model") or ""))


def _client_gone(sock: socket.socket) -> bool:
    """Whether the client closed the connection, for example after its own timeout."""
    try:
        readable, _, _ = select.select([sock], [], [], 0)
        return bool(readable) and sock.recv(1, socket.MSG_PEEK) == b""
    except OSError:
        return True


class Handler(BaseHTTPRequestHandler):
    server_version = "harness2-judge-proxy"

    def do_POST(self):
        proxy: Proxy = self.server.proxy
        if urlparse(self.path).path != PATH:
            return self._send(*error(404, f"only POST {PATH} is served"))
        if not proxy.authorized(self.headers.get("Authorization")):
            return self._send(*error(401, "wrong or missing key; use HARNESS2_JUDGE_API_KEY"))
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)))
        except ValueError:
            return self._send(*error(400, "the request body is not JSON"))
        self._send(*proxy.answer(body, gone=lambda: _client_gone(self.connection)))

    def do_GET(self):
        self._send(*error(404, f"only POST {PATH} is served"))

    def _send(self, status: int, payload: dict) -> None:
        data = json.dumps(payload).encode("utf-8")
        try:
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        except OSError:
            pass

    def log_request(self, code="-", size="-"):
        """Proxy.answer prints one line per call instead."""


def _default_port() -> int:
    """The port in HARNESS2_JUDGE_BASE_URL when that points at this machine."""
    url = urlparse(config.JUDGE_BASE_URL)
    if url.hostname in ("127.0.0.1", "localhost") and url.port:
        return url.port
    return DEFAULT_PORT


def _login(cli: str) -> str:
    """How claude is logged in, or an empty string when it is not."""
    r = subprocess.run(
        [cli, "auth", "status", "--json"],
        env=claude_env(),
        capture_output=True,
        text=True,
        timeout=60,
    )
    try:
        status = json.loads(r.stdout)
    except ValueError:
        return ""
    if not isinstance(status, dict) or not status.get("loggedIn"):
        return ""
    return str(status.get("authMethod") or "a login")


def _interrupt(signum, frame):
    """Stop on SIGTERM the same way as on Ctrl-C, so running claude calls are stopped too."""
    raise KeyboardInterrupt


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="harness2-judge-proxy",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--port",
        type=int,
        default=_default_port(),
        help=f"port on 127.0.0.1; default: the one in HARNESS2_JUDGE_BASE_URL, else {DEFAULT_PORT}",
    )
    p.add_argument(
        "--max-calls",
        type=int,
        default=8,
        help="claude calls that run at once; further requests wait (default: 8)",
    )
    p.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="seconds before a claude call is stopped (default: 600)",
    )
    args = p.parse_args(argv)

    cli = shutil.which("claude")
    if not cli:
        p.exit(1, "claude is not on PATH; install Claude Code and log in first\n")
    login = _login(cli)
    if not login:
        p.exit(1, "claude is not logged in; run claude once and log in, then start again\n")
    version = subprocess.run([cli, "--version"], capture_output=True, text=True).stdout.strip()

    key = config.env("HARNESS2_JUDGE_API_KEY")
    settings = {"HARNESS2_JUDGE_BASE_URL": f"http://{HOST}:{args.port}/v1"}
    if not key:
        key = settings["HARNESS2_JUDGE_API_KEY"] = secrets.token_urlsafe(24)

    try:
        server = ThreadingHTTPServer((HOST, args.port), Handler)
    except OSError as exc:
        p.exit(1, f"cannot listen on {HOST}:{args.port}: {exc}; choose another --port\n")
    workdir = Path(tempfile.mkdtemp(prefix="harness2-judge-proxy-"))
    server.daemon_threads = True
    server.proxy = Proxy(cli, key, max_calls=args.max_calls, timeout=args.timeout, workdir=workdir)
    print(f"Serving {settings['HARNESS2_JUDGE_BASE_URL']} with {version}, logged in with {login}")
    print("Add these to .env, then run harness2-judge:")
    print("\n".join(f"{name}={value}" for name, value in settings.items()))
    if not config.JUDGE_MODEL:
        print("HARNESS2_JUDGE_MODEL=<a model name that claude --model accepts>")
    print(flush=True)
    signal.signal(signal.SIGTERM, _interrupt)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        for proc in list(RUNNING):
            _stop(proc)
        shutil.rmtree(workdir, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
