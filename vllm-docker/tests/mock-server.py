#!/usr/bin/env python3
"""Minimal OpenAI-compatible mock for validating tests/test-model.sh wiring."""
import json
import os
import time
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

MODEL = os.environ.get("MOCK_MODEL", "mock-model")
MAXLEN = int(os.environ.get("MOCK_MAX_LEN", "32768"))
PORT = int(os.environ.get("MOCK_PORT", "9999"))


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 — matches base signature
        pass

    def _send(self, obj):
        body = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self.send_response(200)
            self.end_headers()
        elif self.path == "/v1/models":
            self._send({"data": [{"id": MODEL, "max_model_len": MAXLEN}]})
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        time.sleep(0.3)
        msg: dict[str, Any] = {"role": "assistant", "content": "The sky is blue because of Rayleigh scattering."}
        finish, usage = "stop", {"completion_tokens": 12}
        if "response_format" in req:
            msg["content"] = json.dumps(
                {"title": "Login button misaligned on Safari", "severity": 7,
                 "tags": ["UI", "Safari"]}
            )
            usage = {"completion_tokens": 30}
        if "tools" in req:
            msg["content"] = None
            msg["tool_calls"] = [{
                "id": "call_" + uuid.uuid4().hex[:8], "type": "function",
                "function": {"name": "get_weather",
                             "arguments": json.dumps({"city": "Paris", "unit": "c"})}}]
            finish, usage = "tool_calls", {"completion_tokens": 18}
        if "Think step by step" in json.dumps(req.get("messages", [])):
            msg["reasoning_content"] = "Comparing decimals..."
            msg["content"] = "9.9 is greater than 9.11."
            usage = {"completion_tokens": 40}
        if "80 words" in json.dumps(req.get("messages", [])):
            msg["content"] = "word " * 80
            usage = {"completion_tokens": 80}
        self._send({
            "id": "chatcmpl-" + uuid.uuid4().hex[:8], "object": "chat.completion",
            "model": req.get("model", MODEL),
            "choices": [{"index": 0, "message": msg, "finish_reason": finish}],
            "usage": usage,
        })


HTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
