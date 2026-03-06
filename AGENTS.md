# AGENTS.md - Guidelines for AI Agents

CLIMA-µEMA is a deep learning early warning system (LSTM-Autoencoders) for meteorological anomaly detection using micro-station data from Costa Rica.

## Build, Lint, and Test Commands

### Dependencies
```bash
pip install -r requirements.txt
pip install pytest black flake8 mypy
```

### Running Scripts
```bash
python data/stations/raw/ucr_uema_data_downloader.py
```

### Testing
```bash
pytest                          # Run all tests
pytest tests/test_file.py       # Run single test file
pytest tests/test_file.py::test_function_name  # Run single test
pytest -v                       # Verbose output
pytest -k "pattern"             # Match pattern
```

### Code Quality
```bash
black .     # Format code
flake8 .    # Lint
mypy .      # Type check
```

## Code Style Guidelines

### General
- Follow PEP 8 style guidelines
- Use type hints for all function signatures
- Add Google-style docstrings to public functions and classes

### Imports (order: stdlib, third-party, local; alphabetical within groups)
```python
import csv
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import requests
import torch
from torch import nn

from src.utils import helpers
```

### Formatting
- 4 spaces for indentation (no tabs)
- Maximum line length: 100 characters
- Use Black for automatic formatting
- Trailing commas in multi-line structures

### Type Hints
- Use `Optional[X]` instead of `X | None` for Python < 3.10 compatibility
- Use `dict[str, Any]` for dictionaries with mixed value types

```python
def train_model(
    data: torch.Tensor,
    hidden_size: int,
    learning_rate: float = 0.001,
) -> nn.Module:
```

### Naming Conventions
- `snake_case` for functions, variables, modules
- `PascalCase` for classes
- `UPPERCASE` for constants
- Descriptive names, avoid single-letter variables (except loops)

### Docstrings
```python
def function_name(param1: str, param2: int) -> bool:
    """Short description of what the function does.

    Args:
        param1: Description of param1.
        param2: Description of param2.

    Returns:
        Description of what the function returns.

    Raises:
        ValueError: Description of when this error is raised.
    """
```

### Error Handling
- Use specific exception types
- Include informative error messages
- Handle exceptions at the appropriate level

```python
try:
    response = requests.post(url, auth=auth)
    response.raise_for_status()
except requests.exceptions.RequestException as e:
    log(f"HTTP Error: {e}")
```

### Constants
- Group related constants at module level
- Use uppercase with underscores
- Comment non-obvious constants

```python
CR_TZ = timezone(timedelta(hours=-6))  # Costa Rica UTC-6

FEATURE_COLS = ["temperature", "humidity", "pressure"]
```

### File Organization
- Keep related functionality together
- Separate concerns (data, models, training, evaluation, utils)
- Main execution code behind `if __name__ == "__main__":` guard

### Project Structure
```
src/
  evaluation/    # Model evaluation metrics
  models/        # Neural network architectures
  training/      # Training loops and logic
  utils/         # Helper functions
tests/           # Test files
data/
  emergency_alerts/  # Alert PDFs
  stations/          # Weather station CSV data
```

### Testing
- Write tests for all public functions
- Use descriptive names: `test_<function>_<expected_behavior>`
- Use pytest fixtures for setup/teardown
- Test edge cases and error conditions

### Data Files
- Raw data: `data/raw/`
- Processed data: `data/processed/`
- Descriptive file names with prefixes

### Git Practices
- Atomic commits
- Clear commit messages
- Don't commit secrets (API keys, passwords)
- Use `.gitignore`
