#!/usr/bin/env python3
"""Serve a local, CPU-only browser demo for the checked-out LeetGPU challenges.

The app deliberately runs only each challenge's small ``generate_example_test``
through its PyTorch reference implementation. It never contacts the LeetGPU
service or creates an accelerator submission.
"""

from __future__ import annotations

import argparse
import ast
from contextlib import contextmanager
import hashlib
import importlib.util
import json
import math
import sys
import threading
import time
from dataclasses import dataclass
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse


REPO_ROOT = Path(__file__).resolve().parents[1]
CHALLENGES_ROOT = REPO_ROOT / "challenges"
DEMO_ROOT = REPO_ROOT / "demo"
TENSOR_PREVIEW_LIMIT = 32
RUN_LOCK = threading.Lock()


@dataclass(frozen=True)
class ChallengeInfo:
    """A challenge discovered from the currently checked-out working tree."""

    challenge_id: str
    difficulty: str
    directory_name: str
    name: str
    path: Path

    @property
    def source_path(self) -> str:
        return self.path.relative_to(REPO_ROOT).as_posix()


def challenge_name(challenge_file: Path) -> str:
    """Read ``Challenge.name`` without importing the challenge or requiring torch."""
    try:
        module = ast.parse(challenge_file.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return challenge_file.parent.name.replace("_", " ").title()

    for node in module.body:
        if not isinstance(node, ast.ClassDef) or node.name != "Challenge":
            continue
        for statement in node.body:
            if not isinstance(statement, ast.Assign):
                continue
            if not any(
                isinstance(target, ast.Name) and target.id == "name" for target in statement.targets
            ):
                continue
            if isinstance(statement.value, ast.Constant) and isinstance(statement.value.value, str):
                return statement.value.value

    return challenge_file.parent.name.replace("_", " ").title()


def discover_challenges() -> dict[str, ChallengeInfo]:
    """Return all challenge definitions that exist in this checkout."""
    challenges: dict[str, ChallengeInfo] = {}
    for challenge_file in sorted(CHALLENGES_ROOT.glob("*/*/challenge.py")):
        difficulty = challenge_file.parent.parent.name
        directory_name = challenge_file.parent.name
        challenge_id = f"{difficulty}/{directory_name}"
        challenges[challenge_id] = ChallengeInfo(
            challenge_id=challenge_id,
            difficulty=difficulty,
            directory_name=directory_name,
            name=challenge_name(challenge_file),
            path=challenge_file,
        )
    return challenges


def load_challenge(challenge: ChallengeInfo) -> Any:
    """Load one challenge module with the repository's ``core`` package available."""
    challenges_path = str(CHALLENGES_ROOT)
    if challenges_path not in sys.path:
        sys.path.insert(0, challenges_path)

    module_name = "cpu_demo_" + hashlib.sha1(challenge.challenge_id.encode("utf-8")).hexdigest()
    spec = importlib.util.spec_from_file_location(module_name, challenge.path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {challenge.source_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module.Challenge(device="cpu")


def json_value(value: Any) -> Any:
    """Convert values that may appear in a test case into JSON-safe values."""
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, tuple):
        return [json_value(item) for item in value]
    if isinstance(value, list):
        return [json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    return str(value)


def summarize_value(value: Any, torch: Any) -> Any:
    """Return a compact, inspectable representation of a scalar or tensor."""
    if not torch.is_tensor(value):
        return json_value(value)

    tensor = value.detach().cpu()
    preview = tensor.reshape(-1)[:TENSOR_PREVIEW_LIMIT].tolist()
    return {
        "kind": "tensor",
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype).removeprefix("torch."),
        "values": json_value(preview),
        "preview_truncated": tensor.numel() > TENSOR_PREVIEW_LIMIT,
    }


@contextmanager
def emulate_uint32_when_needed(challenge_info: ChallengeInfo, torch: Any):
    """Represent uint32 as int64 when the installed PyTorch has no uint32 dtype.

    PyTorch does not provide ``torch.uint32`` in several CPU builds. Values remain
    in the unsigned 32-bit range; ``int64`` is only used as a lossless display and
    execution representation for the local example.
    """
    needs_emulation = not hasattr(
        torch, "uint32"
    ) and "torch.uint32" in challenge_info.path.read_text(encoding="utf-8")
    if not needs_emulation:
        yield False
        return

    torch.uint32 = torch.int64
    try:
        yield True
    finally:
        delattr(torch, "uint32")


def execute_reference(
    challenge_info: ChallengeInfo, challenge: Any, test_case: dict[str, Any], uint32_emulated: bool
) -> Any:
    """Run the reference implementation, including the one CPU dtype adaptation."""
    if uint32_emulated and challenge_info.challenge_id == "easy/24_rainbow_table":
        # ``Tensor.view(torch.uint32)`` cannot be emulated with int64 because it
        # would change the tensor's element width. ``fnv1a_hash`` already returns
        # the desired unsigned 32-bit values as non-negative int64 values.
        current = test_case["input"]
        for _ in range(test_case["R"]):
            current = challenge.fnv1a_hash(current)
        test_case["output"].copy_(current)
        return None
    return challenge.reference_impl(**test_case)


def run_example(challenge_info: ChallengeInfo) -> dict[str, Any]:
    """Execute one generated example through the reference implementation on CPU."""
    try:
        import torch
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "PyTorch is required for the CPU demo. Run "
            "`python -m pip install -r scripts/requirements.txt`."
        ) from error

    with RUN_LOCK, emulate_uint32_when_needed(challenge_info, torch) as uint32_emulated:
        challenge = load_challenge(challenge_info)
        test_case = challenge.generate_example_test()
        if not isinstance(test_case, dict):
            raise TypeError("generate_example_test() must return a dictionary of arguments")

        signature = challenge.get_solve_signature()
        output_names = [
            name
            for name, parameter in signature.items()
            if isinstance(parameter, tuple)
            and len(parameter) > 1
            and parameter[1] in {"out", "inout"}
        ]
        inputs = {
            name: summarize_value(value, torch)
            for name, value in test_case.items()
            if name not in output_names
        }

        started = time.perf_counter()
        return_value = execute_reference(challenge_info, challenge, test_case, uint32_emulated)
        elapsed_ms = (time.perf_counter() - started) * 1_000

        outputs = {
            name: summarize_value(test_case[name], torch)
            for name in output_names
            if name in test_case
        }
        if return_value is not None:
            outputs["return_value"] = summarize_value(return_value, torch)

    return {
        "challenge": challenge_info.name,
        "device": "cpu",
        "elapsed_ms": round(elapsed_ms, 3),
        "execution_note": (
            "Unsigned 32-bit values are represented as int64 because this local PyTorch build "
            "does not provide torch.uint32."
            if uint32_emulated
            else "PyTorch reference implementation."
        ),
        "inputs": inputs,
        "outputs": outputs,
    }


class DemoRequestHandler(SimpleHTTPRequestHandler):
    """Serve the browser app plus a small local-only JSON API."""

    server_version = "LeetGPUCPUDemo/1.0"

    @property
    def challenges(self) -> dict[str, ChallengeInfo]:
        return self.server.challenges  # type: ignore[attr-defined]

    def log_message(self, format: str, *args: Any) -> None:
        print(f"{self.address_string()} - {format % args}")

    def send_json(self, status: HTTPStatus, data: Any) -> None:
        payload = json.dumps(data, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def send_file(self, file_path: Path, content_type: str) -> None:
        payload = file_path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def challenge_for_request(self, prefix: str, suffix: str = "") -> ChallengeInfo | None:
        requested = unquote(urlparse(self.path).path.removeprefix(prefix)).lstrip("/")
        if suffix and requested.endswith(suffix):
            requested = requested[: -len(suffix)].rstrip("/")
        return self.challenges.get(requested)

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path in {"/", "/index.html"}:
            self.send_file(DEMO_ROOT / "index.html", "text/html; charset=utf-8")
            return
        if path == "/api/challenges":
            self.send_json(
                HTTPStatus.OK,
                {
                    "challenges": [
                        {
                            "id": challenge.challenge_id,
                            "difficulty": challenge.difficulty,
                            "name": challenge.name,
                            "source_path": challenge.source_path,
                        }
                        for challenge in self.challenges.values()
                    ]
                },
            )
            return
        if path.startswith("/api/challenges/"):
            challenge = self.challenge_for_request("/api/challenges/")
            if challenge is None:
                self.send_json(HTTPStatus.NOT_FOUND, {"error": "Challenge not found"})
                return
            description = challenge.path.with_name("challenge.html")
            self.send_json(
                HTTPStatus.OK,
                {
                    "id": challenge.challenge_id,
                    "difficulty": challenge.difficulty,
                    "name": challenge.name,
                    "source_path": challenge.source_path,
                    "description_html": (
                        description.read_text(encoding="utf-8")
                        if description.exists()
                        else "<p>No challenge description is available in this checkout.</p>"
                    ),
                },
            )
            return
        self.send_json(HTTPStatus.NOT_FOUND, {"error": "Not found"})

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if not path.startswith("/api/challenges/") or not path.endswith("/run-example"):
            self.send_json(HTTPStatus.NOT_FOUND, {"error": "Not found"})
            return

        challenge = self.challenge_for_request("/api/challenges/", suffix="/run-example")
        if challenge is None:
            self.send_json(HTTPStatus.NOT_FOUND, {"error": "Challenge not found"})
            return

        try:
            self.send_json(HTTPStatus.OK, run_example(challenge))
        except Exception as error:
            self.send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(error)})


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve LeetGPU reference examples locally on CPU.")
    parser.add_argument("--host", default="127.0.0.1", help="Host interface (default: 127.0.0.1)")
    parser.add_argument("--port", default=8000, type=int, help="Port number (default: 8000)")
    args = parser.parse_args()

    challenges = discover_challenges()
    if not challenges:
        parser.error(f"No challenge.py files found under {CHALLENGES_ROOT}")

    server = ThreadingHTTPServer((args.host, args.port), DemoRequestHandler)
    server.challenges = challenges  # type: ignore[attr-defined]
    print(f"LeetGPU CPU demo: http://{args.host}:{args.port}")
    print(f"Discovered {len(challenges)} challenges from this checkout. Press Ctrl-C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nCPU demo stopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
