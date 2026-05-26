from __future__ import annotations

# Verifies generated Lean proof completions against a local Lean project and emits per-row pass/fail results.
#
# PYTHONPATH=src python -m lean_eval.harness \
#     --dataset data/processed/test.jsonl \
#     --predictions outputs/predictions.cleaned.jsonl \
#     --output outputs/verified.jsonl \
#     --summary outputs/verified.summary.json \
#     --lake-project-dir lean

import argparse
import concurrent.futures
import dataclasses
import hashlib
import json
import multiprocessing
import os
import queue
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

# Default system prompt retained for rows or tools that still expect a canonical prompt string.
DEFAULT_SYSTEM = (
    "You are an expert Lean 4 theorem prover. Your task is to complete the formal proof by "
    "providing the tactic or sequence of tactics required to complete the proof from the "
    "`sorry`'s position and close the goal."
)
# Ordered fallback fields searched when a row does not specify an explicit completion field.
DEFAULT_COMPLETION_FIELDS = (
    "formal_completion",
    "prediction",
    "completion",
    "generated_text",
    "output",
    "text",
)
# Matches the start of a Lean declaration so code can be split into header and body sections.
DECL_RE = re.compile(
    r"(?m)^(?:@[^\n]*\n\s*)*(?:private\s+|protected\s+|noncomputable\s+|unsafe\s+|partial\s+)*"
    r"(?:theorem|lemma|example|def|abbrev|instance)\b"
)

# Captures the name of a named Lean declaration so verification can safely rename it.
NAMED_DECL_RE = re.compile(
    r"(?m)^(?P<prefix>\s*(?:@[^\n]*\n\s*)*"
    r"(?:(?:private|protected|noncomputable|unsafe|partial)\s+)*"
    r"(?:theorem|lemma|def|abbrev|instance)\s+)"
    r"(?P<name>«[^»]+»|[^\s(:{]+)"
    r"(?P<suffix>(?:\s|$))"
)

# Per-process REPL instance reused by worker processes in the multiprocessing backend.
_WORKER: LeanRepl | None = None
# Per-process timeout used for normal verification commands in worker mode.
_WORKER_TIMEOUT = 20.0
# Per-process timeout used for full-file or header setup work in worker mode.
_WORKER_HEADER_TIMEOUT = 120.0
# Per-process split mode used by worker processes when verifying through the REPL.
_WORKER_SPLIT_MODE = "full"


# Raised when the long-lived Lean REPL process fails in a non-domain-specific way.
class LeanReplError(RuntimeError):
    pass


# Immutable verification input passed between the file-loading and verification layers.
@dataclasses.dataclass(frozen=True)
class VerificationRequest:
    item_id: str
    original_id: str
    formal_context_with_sorry: str
    completion: str
    metadata: dict[str, Any]


# Read a JSONL file into a list of row dictionaries.
# path: JSONL file to read.
def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: expected a JSON object")
            rows.append(row)
    return rows


# Write an iterable of row dictionaries to a JSONL file.
# path: Destination JSONL file to create or overwrite.
# rows: Row dictionaries to serialize one per line.
def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


# Pick the completion string from a verification row.
# row: Row containing one of the supported completion fields.
# explicit_field: Optional field name that should be used exclusively.
def pick_completion(row: dict[str, Any], explicit_field: str | None = None) -> str:
    fields = (explicit_field,) if explicit_field else DEFAULT_COMPLETION_FIELDS
    for field in fields:
        if field and field in row and row[field] is not None:
            value = row[field]
            if isinstance(value, str):
                return value
            raise ValueError(f"completion field {field!r} must be a string")
    raise ValueError(
        "could not find a completion field; use --completion-field or one of "
        f"{', '.join(DEFAULT_COMPLETION_FIELDS)}"
    )


# Build normalized verification requests from input rows or dataset-plus-predictions pairs.
# input_rows: Pre-merged rows when using the single-input mode.
# dataset_rows: Dataset rows when merging dataset and prediction files.
# prediction_rows: Prediction rows to merge onto dataset rows.
# completion_field: Optional explicit completion field name to read.
def build_requests(
    *,
    input_rows: list[dict[str, Any]] | None,
    dataset_rows: list[dict[str, Any]] | None,
    prediction_rows: list[dict[str, Any]] | None,
    completion_field: str | None,
) -> list[VerificationRequest]:
    if input_rows is not None:
        rows = input_rows
    else:
        if dataset_rows is None or prediction_rows is None:
            raise ValueError("expected either --input or both --dataset and --predictions")
        by_item = {str(row["item_id"]): row for row in dataset_rows}
        rows = []
        for pred in prediction_rows:
            item_id = str(pred.get("item_id", ""))
            if item_id not in by_item:
                raise ValueError(f"prediction item_id {item_id!r} is absent from dataset")
            merged = dict(by_item[item_id])
            merged.update(pred)
            rows.append(merged)

    requests: list[VerificationRequest] = []
    for idx, row in enumerate(rows):
        try:
            context = row["formal_context_with_sorry"]
            if not isinstance(context, str):
                raise ValueError("formal_context_with_sorry must be a string")
            item_id = str(row.get("item_id", idx))
            original_id = str(row.get("original_id", item_id))
            completion = pick_completion(row, completion_field)
        except Exception as exc:
            raise ValueError(f"row {idx}: {exc}") from exc
        requests.append(
            VerificationRequest(
                item_id=item_id,
                original_id=original_id,
                formal_context_with_sorry=context,
                completion=completion,
                metadata={
                    key: row[key]
                    for key in ("source_dataset", "is_geometry", "state_after")
                    if key in row
                },
            )
        )
    return requests


# Replace the single `sorry` placeholder with a generated completion.
# formal_context_with_sorry: Lean code containing exactly one `sorry`.
# completion: Replacement Lean code to splice in.
def substitute_completion(formal_context_with_sorry: str, completion: str) -> str:
    count = formal_context_with_sorry.count("sorry")
    if count != 1:
        raise ValueError(f"expected exactly one sorry in formal_context_with_sorry, found {count}")
    return formal_context_with_sorry.replace("sorry", completion, 1)


# Convert an item id into a safe Lean declaration name suffix.
# item_id: Dataset item identifier to sanitize.
def safe_decl_name(item_id: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_]", "_", item_id)
    safe = re.sub(r"_+", "_", safe).strip("_")
    if not safe or safe[0].isdigit():
        safe = f"item_{safe}"
    return f"verify_{safe[:80]}"


# Rename the named declaration preceding `sorry` so verification does not clash on repeated names.
# code: Lean code containing the declaration to rename.
# new_name: Replacement declaration name.
def rename_declaration_containing_sorry(code: str, new_name: str) -> tuple[str, str | None]:
    sorry_idx = code.find("sorry")
    if sorry_idx == -1:
        return code, None
    matches = [match for match in NAMED_DECL_RE.finditer(code) if match.start() < sorry_idx]
    if not matches:
        return code, None
    match = matches[-1]
    old_name = match.group("name")
    renamed = code[: match.start("name")] + new_name + code[match.end("name") :]
    return renamed, old_name


# Rename the first named declaration in a Lean code block.
# code: Lean code whose first declaration should be renamed.
# new_name: Replacement declaration name.
def rename_first_named_declaration(code: str, new_name: str) -> tuple[str, str | None]:
    match = NAMED_DECL_RE.search(code)
    if match is None:
        return code, None
    old_name = match.group("name")
    renamed = code[: match.start("name")] + new_name + code[match.end("name") :]
    return renamed, old_name


# Split Lean code into an import/comment header and the remaining declaration body.
# code: Lean code to split.
def split_header_body(code: str) -> tuple[str, str]:
    match = DECL_RE.search(code)
    if not match:
        return "", code
    preamble = code[: match.start()]
    body = code[match.start() :]

    lines = preamble.splitlines(keepends=True)
    split_idx = 0
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith("--"):
            split_idx = idx + 1
            continue
        if stripped.startswith("import "):
            split_idx = idx + 1
            continue
        break

    header = "".join(lines[:split_idx])
    body = "".join(lines[split_idx:]) + body
    return header, body


# Detect whether a REPL response reports an error or unresolved `sorry`.
# response: JSON response object returned by the Lean REPL.
def response_has_error_or_sorry(response: dict[str, Any]) -> bool:
    for message in response.get("messages", []) or []:
        severity = str(message.get("severity", "")).lower()
        data = str(message.get("data", ""))
        if severity == "error":
            return True
        if "declaration uses sorry" in data:
            return True
    return False


# Hash a header string into a stable cache key.
# header: Header text whose cache key should be computed.
def stable_header_key(header: str) -> str:
    return hashlib.sha256(header.encode("utf-8")).hexdigest()


# Verify a completion by writing a temporary Lean file and invoking `lake env lean`.
# request: Normalized verification request to check.
# lake_project_dir: Lean project directory where verification should run.
# timeout: Maximum time allowed for the Lean process.
# keep_failures_dir: Optional directory for copying failing temporary files.
def verify_with_lean_file(
    request: VerificationRequest,
    *,
    lake_project_dir: Path,
    timeout: float,
    keep_failures_dir: Path | None = None,
) -> dict[str, Any]:
    start = time.perf_counter()
    try:
        renamed_context, old_name = rename_declaration_containing_sorry(
            request.formal_context_with_sorry,
            safe_decl_name(request.item_id),
        )
        code = substitute_completion(renamed_context, request.completion)
    except ValueError as exc:
        return result_row(request, passed=False, status="invalid_input", error=str(exc), start=start)

    safe_id = re.sub(r"[^A-Za-z0-9_.-]", "_", request.item_id)[:80]
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        suffix=".lean",
        prefix=f"verify_{safe_id}_",
        dir=lake_project_dir,
        delete=False,
    ) as f:
        temp_path = Path(f.name)
        f.write(code)
        if not code.endswith("\n"):
            f.write("\n")

    keep_path: Path | None = None
    try:
        proc = subprocess.run(
            ["lake", "env", "lean", str(temp_path.name)],
            cwd=lake_project_dir,
            text=True,
            capture_output=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        if keep_failures_dir is not None:
            keep_failures_dir.mkdir(parents=True, exist_ok=True)
            keep_path = keep_failures_dir / temp_path.name
            shutil.copyfile(temp_path, keep_path)
        return result_row(
            request,
            passed=False,
            status="timeout",
            error=f"lake env lean timed out after {timeout:.1f}s",
            stderr=((exc.stderr or "") + (exc.stdout or "")).splitlines(),
            kept_file=str(keep_path) if keep_path else None,
            start=start,
        )
    finally:
        pass

    output = "\n".join(part for part in (proc.stdout, proc.stderr) if part)
    uses_sorry = "declaration uses sorry" in output
    passed = proc.returncode == 0 and not uses_sorry
    status = "pass" if passed else "lean_error"
    if not passed and keep_failures_dir is not None:
        keep_failures_dir.mkdir(parents=True, exist_ok=True)
        keep_path = keep_failures_dir / temp_path.name
        shutil.copyfile(temp_path, keep_path)
    try:
        temp_path.unlink()
    except FileNotFoundError:
        pass
    return result_row(
        request,
        passed=passed,
        status=status,
        error=None if passed else f"lake env lean exited with code {proc.returncode}",
        stderr=output.splitlines()[-80:],
        kept_file=str(keep_path) if keep_path else None,
        renamed_from=old_name,
        start=start,
    )


class LeanRepl:
    # Initialize a Lean REPL wrapper and start the subprocess immediately.
    # self: REPL wrapper instance being initialized.
    # lake_project_dir: Lean project directory where the REPL should run.
    # repl_cmd: Command used to launch the REPL process.
    def __init__(self, lake_project_dir: Path, repl_cmd: list[str]):
        self.lake_project_dir = lake_project_dir
        self.repl_cmd = repl_cmd
        self.proc: subprocess.Popen[str] | None = None
        self.stdout_queue: queue.Queue[str | None] = queue.Queue()
        self.stderr_lines: list[str] = []
        self.env_cache: dict[str, int] = {}
        self.restart_count = 0
        self.start()

    # Start or restart the underlying REPL subprocess and its I/O threads.
    # self: REPL wrapper instance to start.
    def start(self) -> None:
        self.close()
        self.stdout_queue = queue.Queue()
        self.stderr_lines = []
        self.proc = subprocess.Popen(
            self.repl_cmd,
            cwd=self.lake_project_dir,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )
        assert self.proc.stdout is not None
        assert self.proc.stderr is not None
        threading.Thread(target=self._drain_stdout, daemon=True).start()
        threading.Thread(target=self._drain_stderr, daemon=True).start()

    # Terminate the REPL subprocess if it is still running.
    # self: REPL wrapper instance to close.
    def close(self) -> None:
        if self.proc is None:
            return
        if self.proc.poll() is None:
            self.proc.kill()
            self.proc.wait(timeout=5)
        self.proc = None

    # Restart the REPL subprocess and clear any cached environments.
    # self: REPL wrapper instance to restart.
    def restart(self) -> None:
        self.restart_count += 1
        self.env_cache.clear()
        self.start()

    # Return the most recent stderr lines captured from the REPL process.
    # self: REPL wrapper instance whose stderr buffer is queried.
    def recent_stderr(self) -> list[str]:
        return list(self.stderr_lines[-20:])

    # Continuously drain stdout into a queue for JSON response parsing.
    # self: REPL wrapper instance whose stdout should be drained.
    def _drain_stdout(self) -> None:
        assert self.proc is not None and self.proc.stdout is not None
        for line in self.proc.stdout:
            if line.strip():
                self.stdout_queue.put(line)
        self.stdout_queue.put(None)

    # Continuously retain a short rolling window of stderr output.
    # self: REPL wrapper instance whose stderr should be drained.
    def _drain_stderr(self) -> None:
        assert self.proc is not None and self.proc.stderr is not None
        for line in self.proc.stderr:
            line = line.rstrip("\n")
            if line:
                self.stderr_lines.append(line)
                del self.stderr_lines[:-20]

    # Send one JSON payload to the REPL and parse the next JSON response.
    # self: REPL wrapper instance used for communication.
    # payload: JSON command payload to send.
    # timeout: Maximum time to wait for a response.
    def send(self, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
        if self.proc is None or self.proc.poll() is not None:
            raise LeanReplError("REPL process is not running")
        assert self.proc.stdin is not None
        try:
            self.proc.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n\n")
            self.proc.stdin.flush()
        except BrokenPipeError as exc:
            raise LeanReplError("REPL stdin closed") from exc

        decoder = json.JSONDecoder()
        chunks: list[str] = []
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"Lean REPL timed out after {timeout:.1f}s")
            try:
                line = self.stdout_queue.get(timeout=remaining)
            except queue.Empty as exc:
                raise TimeoutError(f"Lean REPL timed out after {timeout:.1f}s") from exc

            if line is None:
                stderr = "\n".join(self.stderr_lines[-5:])
                raise LeanReplError(f"REPL exited before producing a response: {stderr}")

            chunks.append(line)
            text = "".join(chunks)
            try:
                response, end = decoder.raw_decode(text)
            except json.JSONDecodeError:
                continue
            if text[end:].strip():
                raise LeanReplError(f"REPL returned trailing non-JSON output: {text[end:]!r}")
            break

        if not isinstance(response, dict):
            raise LeanReplError(f"expected REPL response object, got {type(response).__name__}")
        return response

    # Verify a single request through the REPL backend.
    # self: REPL wrapper instance used for verification.
    # request: Normalized verification request to check.
    # timeout: Timeout for body verification work.
    # header_timeout: Timeout for full-file or header setup work.
    # split_mode: Verification mode controlling full-file versus split execution.
    def verify(
        self,
        request: VerificationRequest,
        timeout: float,
        header_timeout: float,
        split_mode: str,
    ) -> dict[str, Any]:
        start = time.perf_counter()
        try:
            renamed_context, old_name = rename_declaration_containing_sorry(
                request.formal_context_with_sorry,
                safe_decl_name(request.item_id),
            )
            code = substitute_completion(renamed_context, request.completion)
        except ValueError as exc:
            return result_row(request, passed=False, status="invalid_input", error=str(exc), start=start)

        try:
            phase = "file"
            if split_mode == "full":
                response = self.send({"cmd": code}, header_timeout)
            elif split_mode == "imports":
                header, body = split_header_body(code)
                header_key = stable_header_key(header)
                env_id: int | None = None
                phase = "body"
                if header.strip():
                    env_id = self.env_cache.get(header_key)
                if header.strip() and env_id is None:
                    header_payload: dict[str, Any] = {"cmd": header}
                    phase = "header"
                    header_resp = self.send(header_payload, header_timeout)
                    if response_has_error_or_sorry(header_resp):
                        return result_row(
                            request,
                            passed=False,
                            status="header_error",
                            response=header_resp,
                            start=start,
                        )
                    env_id = int(header_resp.get("env", 0))
                    self.env_cache[header_key] = env_id

                phase = "body"
                body_payload: dict[str, Any] = {"cmd": body}
                if env_id is not None:
                    body_payload["env"] = env_id
                response = self.send(body_payload, timeout)
            else:
                raise ValueError(f"unknown split_mode {split_mode!r}")
        except TimeoutError as exc:
            message = str(exc)
            status = "header_timeout" if phase in {"header", "file"} else "timeout"
            stderr = self.recent_stderr()
            self.restart()
            return result_row(
                request,
                passed=False,
                status=status,
                error=message,
                stderr=stderr,
                start=start,
            )
        except Exception as exc:
            stderr = self.recent_stderr()
            self.restart()
            return result_row(
                request,
                passed=False,
                status="repl_error",
                error=str(exc),
                stderr=stderr,
                start=start,
            )

        passed = not response_has_error_or_sorry(response)
        return result_row(
            request,
            passed=passed,
            status="pass" if passed else "lean_error",
            response=response,
            renamed_from=old_name,
            start=start,
        )


# Build one verification result row from the request and execution outcome.
# request: Original verification request being reported on.
# passed: Whether verification succeeded.
# status: Machine-readable verification status label.
# start: Timer start used to compute elapsed time.
# response: Optional REPL response object to include.
# error: Optional error message to include.
# stderr: Optional stderr lines to include.
# kept_file: Optional saved failing Lean file path.
# renamed_from: Optional original declaration name that was renamed.
def result_row(
    request: VerificationRequest,
    *,
    passed: bool,
    status: str,
    start: float,
    response: dict[str, Any] | None = None,
    error: str | None = None,
    stderr: list[str] | None = None,
    kept_file: str | None = None,
    renamed_from: str | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "item_id": request.item_id,
        "original_id": request.original_id,
        "pass": passed,
        "status": status,
        "elapsed_sec": round(time.perf_counter() - start, 6),
    }
    row.update(request.metadata)
    if response is not None:
        row["messages"] = response.get("messages", [])
        if "env" in response:
            row["env"] = response["env"]
    if error is not None:
        row["error"] = error
    if stderr:
        row["stderr"] = stderr
    if kept_file:
        row["kept_file"] = kept_file
    if renamed_from:
        row["renamed_from"] = renamed_from
    return row


# Group verification requests by original proof id.
# requests: Verification requests to group.
def grouped_requests(requests: list[VerificationRequest]) -> list[list[VerificationRequest]]:
    groups: dict[str, list[VerificationRequest]] = defaultdict(list)
    for request in requests:
        groups[request.original_id].append(request)
    return list(groups.values())


# Initialize one process-pool worker with a persistent Lean REPL instance.
# lake_project_dir: Lean project directory where the REPL should run.
# repl_cmd: Command used to launch the REPL.
# timeout: Timeout for body verification work.
# header_timeout: Timeout for full-file or header setup work.
# split_mode: Verification mode controlling full-file versus split execution.
def _init_worker(
    lake_project_dir: str,
    repl_cmd: list[str],
    timeout: float,
    header_timeout: float,
    split_mode: str,
) -> None:
    global _WORKER, _WORKER_TIMEOUT, _WORKER_HEADER_TIMEOUT, _WORKER_SPLIT_MODE
    _WORKER_TIMEOUT = timeout
    _WORKER_HEADER_TIMEOUT = header_timeout
    _WORKER_SPLIT_MODE = split_mode
    _WORKER = LeanRepl(Path(lake_project_dir), repl_cmd)


# Verify one grouped batch of requests inside a worker process.
# group: Requests sharing an original proof id.
def _verify_group(group: list[VerificationRequest]) -> list[dict[str, Any]]:
    if _WORKER is None:
        raise RuntimeError("worker was not initialized")
    return [
        _WORKER.verify(request, _WORKER_TIMEOUT, _WORKER_HEADER_TIMEOUT, _WORKER_SPLIT_MODE)
        for request in group
    ]


# Verify a batch of requests with either the file-based or REPL-based backend.
# requests: Verification requests to execute.
# lake_project_dir: Lean project directory where verification should run.
# repl_cmd: Command used to launch the REPL backend.
# workers: Number of worker processes to use.
# timeout: Timeout for body verification work.
# header_timeout: Timeout for full-file or header setup work.
# split_mode: Verification mode controlling full-file versus split execution.
# backend: Verification backend name.
# keep_failures_dir: Optional directory for copying failing temporary files.
def verify_requests(
    requests: list[VerificationRequest],
    *,
    lake_project_dir: Path,
    repl_cmd: list[str],
    workers: int,
    timeout: float,
    header_timeout: float,
    split_mode: str,
    backend: str,
    keep_failures_dir: Path | None,
) -> list[dict[str, Any]]:
    if workers < 1:
        raise ValueError("workers must be at least 1")
    if backend == "lean":
        if workers == 1:
            return [
                verify_with_lean_file(
                    request,
                    lake_project_dir=lake_project_dir,
                    timeout=header_timeout,
                    keep_failures_dir=keep_failures_dir,
                )
                for request in requests
            ]
        context = multiprocessing.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
            futures = [
                pool.submit(
                    verify_with_lean_file,
                    request,
                    lake_project_dir=lake_project_dir,
                    timeout=header_timeout,
                    keep_failures_dir=keep_failures_dir,
                )
                for request in requests
            ]
            return [row for future in concurrent.futures.as_completed(futures) for row in [future.result()]]
    if backend != "repl":
        raise ValueError(f"unknown backend {backend!r}")

    groups = grouped_requests(requests)
    results: list[dict[str, Any]] = []
    if workers == 1:
        repl = LeanRepl(lake_project_dir, repl_cmd)
        try:
            for group in groups:
                results.extend(
                    repl.verify(request, timeout, header_timeout, split_mode) for request in group
                )
        finally:
            repl.close()
        return results

    context = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=workers,
        mp_context=context,
        initializer=_init_worker,
        initargs=(str(lake_project_dir), repl_cmd, timeout, header_timeout, split_mode),
    ) as pool:
        futures = [pool.submit(_verify_group, group) for group in groups]
        for future in concurrent.futures.as_completed(futures):
            results.extend(future.result())
    return results


# Compute a compact verification summary from result rows.
# results: Verification result rows to summarize.
def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)
    passed = sum(1 for row in results if row.get("pass") is True)
    by_status: dict[str, int] = defaultdict(int)
    for row in results:
        by_status[str(row.get("status", "unknown"))] += 1
    return {
        "total": total,
        "passed": passed,
        "pass_rate": passed / total if total else 0.0,
        "by_status": dict(sorted(by_status.items())),
    }


# Load files, verify all requests, and write both detailed results and an optional summary.
# input_path: Optional single input JSONL containing contexts and completions together.
# dataset_path: Optional dataset JSONL used with predictions_path.
# predictions_path: Optional prediction JSONL used with dataset_path.
# output_path: Destination JSONL path for detailed verification results.
# summary_path: Optional JSON path for the aggregate summary.
# lake_project_dir: Lean project directory where verification should run.
# repl_cmd: Command used to launch the REPL backend.
# workers: Number of worker processes to use.
# timeout: Timeout for body verification work.
# header_timeout: Timeout for full-file or header setup work.
# split_mode: Verification mode controlling full-file versus split execution.
# backend: Verification backend name.
# keep_failures_dir: Optional directory for copying failing temporary files.
# completion_field: Optional explicit completion field name to read.
def verify_files(
    *,
    input_path: Path | None,
    dataset_path: Path | None,
    predictions_path: Path | None,
    output_path: Path,
    summary_path: Path | None,
    lake_project_dir: Path,
    repl_cmd: list[str],
    workers: int = 4,
    timeout: float = 20.0,
    header_timeout: float = 120.0,
    split_mode: str = "full",
    backend: str = "lean",
    keep_failures_dir: Path | None = None,
    completion_field: str | None = None,
) -> dict[str, Any]:
    requests = build_requests(
        input_rows=read_jsonl(input_path) if input_path else None,
        dataset_rows=read_jsonl(dataset_path) if dataset_path else None,
        prediction_rows=read_jsonl(predictions_path) if predictions_path else None,
        completion_field=completion_field,
    )
    results = verify_requests(
        requests,
        lake_project_dir=lake_project_dir,
        repl_cmd=repl_cmd,
        workers=workers,
        timeout=timeout,
        header_timeout=header_timeout,
        split_mode=split_mode,
        backend=backend,
        keep_failures_dir=keep_failures_dir,
    )
    results.sort(key=lambda row: row["item_id"])
    write_jsonl(output_path, results)
    summary = summarize(results)
    if summary_path is not None:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


# Parse CLI arguments for the verification harness.
def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify Lean proof completions with the Lean REPL.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path, help="JSONL containing contexts and generated completions.")
    source.add_argument("--dataset", type=Path, help="Dataset JSONL containing formal_context_with_sorry.")
    parser.add_argument("--predictions", type=Path, help="Prediction JSONL; required with --dataset.")
    parser.add_argument("--output", type=Path, required=True, help="Output JSONL of verification results.")
    parser.add_argument("--summary", type=Path, help="Optional JSON summary output path.")
    parser.add_argument("--lake-project-dir", type=Path, default=Path("lean"))
    parser.add_argument(
        "--repl-cmd",
        default=os.environ.get("LEAN_REPL_CMD", "lake env repl"),
        help="Command used to launch the REPL inside --lake-project-dir.",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument(
        "--header-timeout",
        type=float,
        default=120.0,
        help="Timeout for import/header environment setup. Keep this larger than --timeout.",
    )
    parser.add_argument(
        "--split-mode",
        choices=("full", "imports"),
        default="full",
        help="Use full-file verification by default; imports mode caches imports but is less robust.",
    )
    parser.add_argument(
        "--backend",
        choices=("lean", "repl"),
        default="lean",
        help="Verification backend. 'lean' writes a temp file and runs lake env lean; 'repl' uses JSON REPL.",
    )
    parser.add_argument(
        "--keep-failures-dir",
        type=Path,
        help="Optional directory where failed generated .lean files are copied.",
    )
    parser.add_argument("--completion-field", help="Field containing the model completion.")
    args = parser.parse_args(argv)
    if args.dataset and not args.predictions:
        parser.error("--predictions is required when --dataset is used")
    return args


# Load the requested inputs, run verification, and print the aggregate summary.
def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    summary = verify_files(
        input_path=args.input,
        dataset_path=args.dataset,
        predictions_path=args.predictions,
        output_path=args.output,
        summary_path=args.summary,
        lake_project_dir=args.lake_project_dir,
        repl_cmd=shlex.split(args.repl_cmd),
        workers=args.workers,
        timeout=args.timeout,
        header_timeout=args.header_timeout,
        split_mode=args.split_mode,
        backend=args.backend,
        keep_failures_dir=args.keep_failures_dir,
        completion_field=args.completion_field,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
