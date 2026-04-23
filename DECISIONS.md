# Architectural Decisions

## DECISION-001: Pause Stage 2/3 In Active Runtime And Remove Server

- Date: 2026-04-21
- Status: Implemented

### Context

The repository had two active runtime paths:
- CLI pipeline (`main.py`) that chained transcription, quantization, and rendering.
- Web server pipeline (`server.py`) with FastAPI endpoints and static UI.

Current project priorities require focusing on transcription quality and MIDI output first. Quantization and notation remain valuable but are not part of the immediate runtime scope.

### Decision

1. Remove server functionality from the codebase.
2. Remove static web UI assets.
3. Remove server-only dependencies from project configuration.
4. Deactivate Stage 2 (quantization) and Stage 3 (rendering) from the default runtime path in `main.py`.
5. Keep `quantize.py` and `render.py` in the repository as deferred modules for future reactivation.

### Changes Implemented

- Deleted: `server.py`
- Deleted: `static/index.html` and `static/` directory
- Updated: `main.py` to run transcription only (`audio -> MIDI`)
- Updated: `pyproject.toml` to remove `fastapi`, `uvicorn[standard]`, and `python-multipart`
- Updated docs to reflect current active behavior and deferred stages

### Rationale

- Reduces maintenance surface area while core transcription work is prioritized.
- Avoids keeping a partially supported server stack.
- Preserves Stage 2/3 code for later without blocking current development.

### Stage 2/3 Reactivation Checklist

1. Rewire `main.py` to call `quantize()` and `render()` again.
2. Decide and document whether quantization uses drum stem or full mix for beat tracking input.
3. Restore CLI arguments needed by quantization/rendering (`--src`, `--out`) if required.
4. Verify output contracts for PDF and tempo-mapped MIDI.
5. Re-run docs pass so README and summaries match restored behavior.

### Server Reactivation Checklist

1. Reintroduce a server module and static UI as needed.
2. Restore dependencies (`fastapi`, `uvicorn[standard]`, `python-multipart`) in project config.
3. Add endpoint/API contract tests for job creation, status, and artifact download.
4. Document startup, deployment assumptions, and supported workflows.
