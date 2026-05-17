# AGENTS.md - Repository Guide

CLIMA-UEMA is a meteorological early-warning system for Costa Rica.
It includes LSTM/autoencoder model code, station data pipelines, and PDF-based
emergency alert extraction.

Use this file as the source of truth for agentic work in this repo.

## Repository Notes

- This repo currently has no `.cursor/rules/`, `.cursorrules`, or
  `.github/copilot-instructions.md` files.
- Keep this file updated if those rules appear later.
- Prefer small, focused changes that match the existing code style.

## Setup

```bash
pip install -r requirements.txt
```

- Install extra tooling if needed for local checks:

```bash
pip install pytest black flake8 mypy
```

## Build / Run Commands

- Download station data:

```bash
python data/stations/raw/ucr_uema_data_downloader.py
```

- Extract emergency alerts from PDFs:

```bash
python -m preprocessing.emergency_alerts.extract_alerts_data --google-api-key YOUR_KEY
```

- Show CLI help:

```bash
python -m preprocessing.emergency_alerts.extract_alerts_data -h
```

## Testing Commands

```bash
pytest
pytest -v
pytest -k "pattern"
pytest tests/test_file.py
pytest tests/test_file.py::test_function_name
```

- Single test file: `pytest tests/test_file.py`
- Single test function: `pytest tests/test_file.py::test_function_name`
- Prefer descriptive test names like `test_<function>_<expected_behavior>`.
- Add tests for public functions, edge cases, and failure paths.

## Quality Checks

```bash
black .
flake8 .
mypy .
```

- Run `black` before submitting code.
- Run `pytest` after meaningful changes.
- Run `flake8` and `mypy` when touching shared logic or type-heavy code.

## Code Style

- Follow PEP 8.
- Use 4-space indentation.
- Keep lines to 100 characters or fewer.
- Use Black-compatible formatting and trailing commas in multiline literals.
- Prefer explicit, readable code over clever shortcuts.

## Imports

- Order imports as: stdlib, third-party, local.
- Sort alphabetically within each group.
- Prefer absolute imports within the project when practical.
- Avoid unused imports and circular dependencies.

Example order:

```python
import csv
from pathlib import Path

import pandas as pd

from preprocessing.stations.config import RAW_DATA_DIR
```

## Type Hints

- Add type hints to all function signatures.
- Use `Optional[X]` instead of `X | None` for compatibility with older Python.
- Use `dict[str, Any]` for heterogeneous dictionaries.
- Keep return types explicit.
- Prefer `Path` over raw strings for filesystem inputs.

## Naming Conventions

- `snake_case` for functions, variables, and modules.
- `PascalCase` for classes and Pydantic models.
- `UPPERCASE` for constants.
- Use descriptive names; avoid single-letter variables except in short loops.
- Name tests for behavior, not implementation details.

## Docstrings

- Use Google-style docstrings for public functions and classes.
- Document args, returns, and raised exceptions when relevant.
- Keep internal helper docstrings short unless behavior is non-obvious.
- For pipelines and CLIs, document side effects and output paths.

## Error Handling

- Raise specific exceptions with actionable messages.
- Validate inputs early, near the boundary.
- Handle exceptions at the right layer; do not swallow unexpected errors silently.
- Prefer returning `None` only when that is an intentional API choice.
- Include context in error messages, especially file paths and station names.

## Logging

- Use `logging` or `loguru`; keep usage consistent within a module.
- Log with contextual details such as file names, station names, and alert IDs.
- Use INFO for normal progress, WARNING for recoverable issues, ERROR for failures.
- Do not log secrets, API keys, or full private payloads.

## Data Handling

- Use `pandas` for tabular data and `numpy` for numerical work.
- Use `pathlib.Path` for paths, not `os.path`.
- Preserve the repo's raw/processed data layout under `data/`.
- Treat missing data explicitly; do not assume columns or rows exist.
- Be careful with datetime parsing, resampling, and time zones.

## Validation

- Use Pydantic models for structured extracted data.
- Add `Field(description="...")` to important schema fields.
- Handle `pydantic.ValidationError` explicitly where parsing can fail.
- Keep schemas strict enough to catch malformed alert payloads.

## CLI Guidance

- Use `argparse` or `click` for user-facing scripts.
- Provide `-h` and `--help` support.
- Read API keys from environment variables or CLI flags; never hardcode them.
- Make default paths and outputs obvious.

## Project Structure

```text
src/                Model code and utilities
preprocessing/      Station pipelines and PDF alert extraction
tests/              Pytest suite
data/               Raw and processed datasets
```

## Domain + Environment

- `preprocessing/emergency_alerts/` extracts structured data from CNE PDFs.
- It uses docling OCR and Google Gemini, and writes outputs under
  `data/emergency_alerts/processed/`.
- Station processing should preserve the existing severity/region matching
  behavior.
- CUDA is disabled by default in alert extraction; keep docling CPU-first.
- Use `.env` files locally, but never commit secrets or API keys.

## Git + Agent Behavior

- Keep changes atomic and purpose-driven.
- Do not overwrite unrelated user changes.
- Prefer clear commit messages that explain why the change exists.
- Read relevant files before editing and match existing style patterns.
- When changing public APIs, update tests and docs together.
- If a task touches tests or pipeline behavior, verify with the most targeted
  test command first.
