"""FastAPI web app for the anime storyboard generator.

This file serves the exported frontend in ``000103/000103`` and exposes JSON
APIs that call the existing anime_pipeline_graph backend.
"""

from __future__ import annotations

import argparse
import errno
import json
import os
import socket
import sys
import threading
import traceback
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel


PROJECT_ROOT = Path(__file__).resolve().parent
FRONTEND_DIR = PROJECT_ROOT / "000103" / "000103"
RUNS_URL_PREFIX = "/runs"
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from anime_pipeline_graph.cli.main import run_pipeline  # noqa: E402


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}


class GenerateRequest(BaseModel):
    prompt: str
    dry_run: bool | None = None


class JobStore:
    """Thread-safe in-memory status store for long-running generation jobs."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jobs: dict[str, dict[str, Any]] = {}

    def create(self) -> str:
        job_id = f"job_{uuid.uuid4().hex[:12]}"
        now = utc_now()
        with self._lock:
            self._jobs[job_id] = {
                "job_id": job_id,
                "status": "queued",
                "created_at": now,
                "updated_at": now,
            }
        return job_id

    def update(self, job_id: str, **fields: Any) -> None:
        with self._lock:
            job = self._jobs.setdefault(job_id, {"job_id": job_id})
            job.update(fields)
            job["updated_at"] = utc_now()

    def get(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return dict(job) if job else None


class AppState:
    def __init__(
        self,
        backend: str,
        output_dir: str,
        dry_run_default: bool,
        workers: int,
    ) -> None:
        self.backend = backend
        self.output_dir = output_dir
        self.dry_run_default = dry_run_default
        self.jobs = JobStore()
        self.executor = ThreadPoolExecutor(max_workers=max(1, workers))
        self.futures: dict[str, Future[Any]] = {}

    def submit_generation(self, prompt: str, dry_run: bool | None) -> str:
        job_id = self.jobs.create()
        use_dry_run = self.dry_run_default if dry_run is None else bool(dry_run)

        def worker() -> None:
            self.jobs.update(job_id, status="running")
            try:
                active_backend = set_runtime_env(self.backend, self.output_dir)
                record = run_pipeline(prompt, dry_run=use_dry_run, project_root=PROJECT_ROOT)
                result = build_generation_result(record, prompt=prompt)
                result["backend"] = active_backend
                result["dry_run"] = use_dry_run
                self.jobs.update(job_id, status="succeeded", result=result)
            except Exception as exc:
                self.jobs.update(
                    job_id,
                    status="failed",
                    error=str(exc),
                    traceback=traceback.format_exc(),
                )

        self.futures[job_id] = self.executor.submit(worker)
        return job_id

    def shutdown(self) -> None:
        self.executor.shutdown(wait=False, cancel_futures=True)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def set_runtime_env(backend: str, output_dir: str) -> str:
    """Match the runtime setup used by scripts/run_full_auto_real.py."""

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("FORCE_CPU_OFFLOAD", "1")
    if output_dir.strip():
        os.environ["DEFAULT_OUTPUT_DIR"] = output_dir.strip()

    normalized = backend.strip().lower()
    use_seedream_api = normalized in {
        "seedream4_api",
        "seedream_api",
        "ark_seedream4",
        "ark",
        "doubao_api",
    }
    if use_seedream_api:
        os.environ["IMAGE_BACKEND"] = "seedream4_api"
        os.environ["FORCE_NO_KONTEXT"] = "1"
        os.environ["FORCE_BASE_ONLY"] = "1"
        os.environ["SINGLE_LORA_PER_FRAME"] = "0"
        os.environ["LORA_PER_FRAME"] = "0"
    else:
        os.environ["IMAGE_BACKEND"] = "local"
        os.environ.pop("FORCE_NO_KONTEXT", None)
        os.environ.pop("FORCE_BASE_ONLY", None)
    return os.environ["IMAGE_BACKEND"]


def output_root() -> Path:
    return PROJECT_ROOT / os.getenv("DEFAULT_OUTPUT_DIR", "runs")


def safe_relative(path: Path, base: Path) -> str:
    return path.resolve().relative_to(base.resolve()).as_posix()


def run_dir_for(run_id: str) -> Path:
    if "/" in run_id or "\\" in run_id or not run_id.startswith("run_"):
        raise ValueError("Invalid run id")
    run_dir = output_root() / run_id
    resolved = run_dir.resolve()
    resolved.relative_to(output_root().resolve())
    return resolved


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def natural_key(path: Path) -> tuple[Any, ...]:
    parts: list[Any] = []
    current = ""
    for ch in path.stem:
        if ch.isdigit():
            current += ch
        else:
            if current:
                parts.append(int(current))
                current = ""
            parts.append(ch.lower())
    if current:
        parts.append(int(current))
    parts.append(path.suffix.lower())
    return tuple(parts)


def generated_image_paths(run_dir: Path) -> list[Path]:
    generated_dir = run_dir / "images" / "generated"
    if not generated_dir.exists():
        return []
    return sorted(
        [p for p in generated_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS],
        key=natural_key,
    )


def image_payload(path: Path) -> dict[str, Any]:
    rel = safe_relative(path, output_root())
    stat = path.stat()
    return {
        "filename": path.name,
        "url": f"{RUNS_URL_PREFIX}/{rel}",
        "size_bytes": stat.st_size,
        "modified_at": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
    }


def prompt_from_run(run_dir: Path, record: dict[str, Any] | None = None) -> str:
    metadata = read_json(run_dir / "web_metadata.json")
    if metadata.get("original_prompt"):
        return str(metadata["original_prompt"])
    if metadata.get("prompt"):
        return str(metadata["prompt"])

    record = record if record is not None else read_json(run_dir / "run_record.json")
    frame_descriptions = (((record or {}).get("task_spec") or {}).get("frame_descriptions") or [])
    if frame_descriptions:
        return " ".join(str(item) for item in frame_descriptions)

    for pre_file in sorted((run_dir / "steps").glob("*/pre.json")):
        data = read_json(pre_file)
        prompt = (((data.get("step") or {}).get("inputs") or {}).get("prompt"))
        if prompt:
            return str(prompt)
    return ""


def run_summary(run_dir: Path) -> dict[str, Any] | None:
    record_path = run_dir / "run_record.json"
    if not record_path.exists():
        return None

    record = read_json(record_path)
    images = generated_image_paths(run_dir)
    created = datetime.fromtimestamp(record_path.stat().st_mtime, tz=timezone.utc).isoformat()
    prompt = prompt_from_run(run_dir, record)
    task_spec = record.get("task_spec") or {}
    return {
        "run_id": run_dir.name,
        "prompt": prompt,
        "pipeline_prompt": metadata_value(run_dir, "prompt"),
        "preprocessed_dialogue": bool(metadata_value(run_dir, "preprocessed_dialogue")),
        "created_at": created,
        "num_images": len(images),
        "num_frames": task_spec.get("num_frames") or len(images),
        "task_type": task_spec.get("task_type"),
        "generated_images": [image_payload(p) for p in images],
    }


def list_history(query: str = "", limit: int = 80) -> list[dict[str, Any]]:
    runs_root = output_root()
    if not runs_root.exists():
        return []
    items = []
    query_lc = query.strip().lower()
    for run_dir in runs_root.iterdir():
        if not run_dir.is_dir() or not run_dir.name.startswith("run_"):
            continue
        summary = run_summary(run_dir)
        if not summary:
            continue
        haystack = f"{summary['run_id']} {summary.get('prompt') or ''} {summary.get('pipeline_prompt') or ''}".lower()
        if query_lc and query_lc not in haystack:
            continue
        items.append(summary)
    items.sort(key=lambda item: item.get("created_at") or "", reverse=True)
    return items[:limit]


def metadata_value(run_dir: Path, key: str) -> Any:
    return read_json(run_dir / "web_metadata.json").get(key)


def build_generation_result(
    record: Any,
    prompt: str,
    original_prompt: str | None = None,
    preprocess: dict[str, Any] | None = None,
) -> dict[str, Any]:
    run_id = record.run_id
    run_dir = run_dir_for(run_id)
    backend_preprocess = read_json(run_dir / "dialogue_preprocess.json")
    effective_preprocess = preprocess or backend_preprocess
    pipeline_prompt = str(effective_preprocess.get("pipeline_prompt") or prompt)
    raw_prompt = str(original_prompt or effective_preprocess.get("original_prompt") or prompt)
    metadata = {
        "prompt": pipeline_prompt,
        "original_prompt": raw_prompt,
        "preprocessed_dialogue": bool(effective_preprocess.get("preprocessed")),
        "preprocess": effective_preprocess,
        "created_at": utc_now(),
    }
    (run_dir / "web_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    images = generated_image_paths(run_dir)
    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "prompt": raw_prompt,
        "pipeline_prompt": pipeline_prompt,
        "preprocessed_dialogue": bool(effective_preprocess.get("preprocessed")),
        "preprocess": effective_preprocess,
        "num_images": len(images),
        "generated_images": [image_payload(p) for p in images],
    }


def env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.lower() in {"1", "true", "yes", "on"}


def default_state() -> AppState:
    return AppState(
        backend=os.getenv("IMAGE_BACKEND", "local"),
        output_dir=os.getenv("DEFAULT_OUTPUT_DIR", "runs"),
        dry_run_default=env_bool("WEB_DRY_RUN", False),
        workers=int(os.getenv("APP_WORKERS", "1")),
    )


def get_state(request: Request) -> AppState:
    return request.app.state.backend_state


def create_app(state: AppState | None = None) -> FastAPI:
    fastapi_app = FastAPI(title="Anime Storyboard Generator", version="0.1.0")
    fastapi_app.state.backend_state = state or default_state()

    fastapi_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @fastapi_app.get("/api/health")
    def health() -> dict[str, Any]:
        return {"ok": True, "time": utc_now()}

    @fastapi_app.get("/api/storyboard/history")
    def history(
        q: str = "",
        limit: int = Query(default=80, ge=1, le=250),
    ) -> dict[str, Any]:
        return {"items": list_history(query=q, limit=limit)}

    @fastapi_app.get("/api/storyboard/jobs/{job_id}")
    def job_status(job_id: str, request: Request) -> dict[str, Any]:
        job = get_state(request).jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        return job

    @fastapi_app.get("/api/storyboard/{run_id}/generated")
    def generated(run_id: str) -> dict[str, Any]:
        try:
            summary = run_summary(run_dir_for(run_id))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if not summary:
            raise HTTPException(status_code=404, detail="Run not found")
        return summary

    @fastapi_app.post("/api/storyboard/generate_async", status_code=202)
    def generate_async(payload: GenerateRequest, request: Request) -> dict[str, str]:
        prompt = payload.prompt.strip()
        if not prompt:
            raise HTTPException(status_code=400, detail="Prompt is empty")
        job_id = get_state(request).submit_generation(prompt, dry_run=payload.dry_run)
        return {"job_id": job_id, "status": "queued"}

    @fastapi_app.post("/api/storyboard/generate")
    def generate(payload: GenerateRequest, request: Request) -> dict[str, Any]:
        prompt = payload.prompt.strip()
        if not prompt:
            raise HTTPException(status_code=400, detail="Prompt is empty")
        state = get_state(request)
        use_dry_run = state.dry_run_default if payload.dry_run is None else bool(payload.dry_run)
        active_backend = set_runtime_env(state.backend, state.output_dir)
        record = run_pipeline(prompt, dry_run=use_dry_run, project_root=PROJECT_ROOT)
        result = build_generation_result(record, prompt=prompt)
        result["backend"] = active_backend
        result["dry_run"] = use_dry_run
        return result

    @fastapi_app.on_event("shutdown")
    def shutdown_executor() -> None:
        fastapi_app.state.backend_state.shutdown()

    @fastapi_app.get("/")
    def index() -> FileResponse:
        return FileResponse(FRONTEND_DIR / "index.html")

    @fastapi_app.get("/runs/{file_path:path}")
    def run_file(file_path: str) -> FileResponse:
        base = output_root().resolve()
        target = (base / file_path).resolve()
        try:
            target.relative_to(base)
        except ValueError as exc:
            raise HTTPException(status_code=403, detail="Forbidden") from exc
        if not target.exists() or not target.is_file():
            raise HTTPException(status_code=404, detail="File not found")
        return FileResponse(target)

    fastapi_app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")

    return fastapi_app


app = create_app()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the FastAPI anime storyboard web app.")
    parser.add_argument("--host", default=os.getenv("APP_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("APP_PORT", "8000")))
    parser.add_argument("--backend", default=os.getenv("IMAGE_BACKEND", "local"), help="local | seedream4_api")
    parser.add_argument("--output-dir", default=os.getenv("DEFAULT_OUTPUT_DIR", "runs"))
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=env_bool("WEB_DRY_RUN", False),
        help="Use mock providers by default for browser generation requests.",
    )
    parser.add_argument("--workers", type=int, default=int(os.getenv("APP_WORKERS", "1")))
    return parser.parse_args()


def find_available_port(host: str, start_port: int, attempts: int = 20) -> int:
    for port in range(start_port, start_port + attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind((host, port))
            except OSError as exc:
                if exc.errno == errno.EADDRINUSE:
                    continue
                raise
            return port
    raise RuntimeError(
        f"Ports {start_port}-{start_port + attempts - 1} are all in use. "
        "Please pass --port with a free port."
    )


def main() -> None:
    args = parse_args()
    active_backend = set_runtime_env(args.backend, args.output_dir)
    app.state.backend_state = AppState(
        backend=args.backend,
        output_dir=args.output_dir,
        dry_run_default=args.dry_run,
        workers=args.workers,
    )

    import uvicorn

    port = find_available_port(args.host, args.port)
    print(f"Serving AniBoard at http://{args.host}:{port}")
    print(
        f"Backend={active_backend} output_dir={os.environ['DEFAULT_OUTPUT_DIR']} "
        f"dry_run_default={args.dry_run} dialogue_preprocess={os.getenv('PREPROCESS_DIALOGUE', '1')}"
    )
    uvicorn.run(app, host=args.host, port=port)


if __name__ == "__main__":
    main()
