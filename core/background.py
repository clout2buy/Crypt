"""Background shell task manager."""
from __future__ import annotations

import os
import signal
import subprocess
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from . import runtime, settings


@dataclass
class Job:
    id: str
    command: str
    cwd: str
    output_path: Path
    started_at: float
    process: subprocess.Popen
    description: str = ""


_JOBS: dict[str, Job] = {}


def _jobs_dir() -> Path:
    session_id = runtime.session_id() or "no-session"
    path = settings.APP_DIR / "tasks" / session_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def start(command: str, cwd: str | None = None, description: str = "") -> Job:
    job_id = str(uuid.uuid4())[:8]
    out_path = _jobs_dir() / f"{job_id}.log"
    workdir = str(Path(cwd or runtime.cwd()).expanduser().resolve())
    f = out_path.open("ab")
    header = (
        f"$ {command}\n"
        f"[crypt background job {job_id} started {time.strftime('%Y-%m-%d %H:%M:%S')}]\n\n"
    )
    f.write(header.encode("utf-8", errors="replace"))
    f.flush()
    settings.restrict_file_permissions(out_path)
    popen_kwargs = {}
    if os.name == "nt":
        popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        popen_kwargs["preexec_fn"] = os.setsid
    try:
        proc = subprocess.Popen(
            command,
            shell=True,
            cwd=workdir,
            stdout=f,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            **popen_kwargs,
        )
    finally:
        f.close()
    job = Job(job_id, command, workdir, out_path, time.time(), proc, description)
    _JOBS[job_id] = job
    return job


def get(job_id: str) -> Job | None:
    return _JOBS.get(job_id)


def forget(job_id: str) -> None:
    _JOBS.pop(job_id, None)


def list_jobs() -> list[Job]:
    return sorted(_JOBS.values(), key=lambda j: j.started_at, reverse=True)


def status(job: Job) -> str:
    rc = job.process.poll()
    if rc is None:
        return "running"
    return f"exited {rc}"


def poll(job_id: str, tail_lines: int = 80) -> str:
    job = get(job_id)
    if job is None:
        raise KeyError(f"unknown background job: {job_id}")
    body = _tail(job.output_path, tail_lines)
    return (
        f"job {job.id}: {status(job)}\n"
        f"cwd: {job.cwd}\n"
        f"output: {job.output_path}\n\n"
        f"{body or '(no output yet)'}"
    )


def kill(job_id: str) -> str:
    job = get(job_id)
    if job is None:
        raise KeyError(f"unknown background job: {job_id}")
    if job.process.poll() is not None:
        return f"job {job.id} already {status(job)}"
    if os.name == "nt":
        try:
            subprocess.run(
                ["taskkill", "/PID", str(job.process.pid), "/T", "/F"],
                capture_output=True,
                text=True,
                timeout=5,
            )
        except Exception:
            job.process.send_signal(signal.CTRL_BREAK_EVENT)
        time.sleep(0.5)
        if job.process.poll() is None:
            job.process.kill()
    else:
        try:
            os.killpg(os.getpgid(job.process.pid), signal.SIGTERM)
        except Exception:
            job.process.terminate()
        time.sleep(0.5)
        if job.process.poll() is None:
            try:
                os.killpg(os.getpgid(job.process.pid), signal.SIGKILL)
            except Exception:
                job.process.kill()
    return f"job {job.id} killed"


def _tail(path: Path, lines: int) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    parts = text.splitlines()
    if lines <= 0:
        lines = 80
    return "\n".join(parts[-lines:])
