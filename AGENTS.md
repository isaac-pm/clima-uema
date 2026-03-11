# AGENTS.md - Guidelines for AI Agents

CLIMA-µEMA is a deep learning early warning system (LSTM-Autoencoders) for meteorological anomaly detection using micro-station data from Costa Rica. It also includes a pipeline for extracting structured data from Costa Rican emergency weather alert PDFs.

## Build, Lint, and Test Commands

### Dependencies

```bash
pip install -r requirements.txt
pip install pytest black flake8 mypy
```

### Running Scripts

```bash
# Download micro-station data
python data/stations/raw/ucr_uema_data_downloader.py

# Extract emergency alerts from PDFs (requires Google API key)
python -m preprocessing.emergency_alerts.extract_alerts_data --google-api-key YOUR_KEY
python -m preprocessing.emergency_alerts.extract_alerts_data -h  # Show all options
```

### Testing

```bash
pytest                          # Run all tests
pytest tests/test_file.py       # Run single test file
pytest tests/test_file.py::test_function_name  # Run single test
pytest -v                       # Verbose output
pytest -k "pattern"             # Match pattern
```

- Write tests for all public functions with descriptive names: `test_<function>_<expected_behavior>`
- Use pytest fixtures for setup/teardown, test edge cases and error conditions

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
- 4 spaces for indentation (no tabs), max line length: 100 characters
- Use Black for automatic formatting with trailing commas in multi-line structures

### Imports

Order: stdlib, third-party, local (alphabetical within groups). Example: `csv`, `os`, `datetime` -> `requests`, `torch` -> `from src.utils import helpers`

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

### Error Handling

- Use specific exception types with informative error messages
- Handle exceptions at the appropriate level

```python
try:
    response = requests.post(url, auth=auth)
    response.raise_for_status()
except requests.exceptions.RequestException as e:
    log(f"HTTP Error: {e}")
```

### Constants & File Organization

- Group related constants at module level with uppercase and underscores
- Keep related functionality together, separate concerns
- Main execution code behind `if __name__ == "__main__":` guard

```python
CR_TZ = timezone(timedelta(hours=-6))  # Costa Rica UTC-6
FEATURE_COLS = ["temperature", "humidity", "pressure"]
```

### Project Structure

```
src/
  evaluation/    # Model evaluation metrics
  models/        # Neural network architectures
  training/      # Training loops and logic
  utils/         # Helper functions
preprocessing/
  emergency_alerts/   # Emergency alert PDF extraction pipeline
tests/           # Test files
data/
  emergency_alerts/  # Alert PDFs (raw/, processed/)
  stations/          # Weather station CSV data (raw/, processed/)
```

### Preprocessing Module

The `preprocessing/emergency_alerts/` module extracts structured data from Costa Rican emergency weather alert PDFs using docling for OCR and Google Gemini LLM for structured extraction. Output goes to CSV format, logs to `data/emergency_alerts/processed/alert_processing.log`.

### Git Practices

- Atomic commits with clear commit messages
- Don't commit secrets (API keys, passwords)
- Use `.gitignore`

### Logging

- Use `loguru` for logging (already in requirements.txt) or standard `logging` module
- Configure loguru with appropriate levels: DEBUG for development, INFO for production
- Include contextual information in log messages: `logger.info("Processing file: {}", file_path)`

### Data Validation

- Use `pydantic` for data validation and schema definition (already in requirements.txt)
- Define models with descriptive field descriptions using `Field(description="...")`
- Handle `PydanticValidationError` explicitly

### CLI Tools

- Use `click` or `argparse` for command-line interfaces
- Add `--help` support and `-h` short flag
- Use environment variables for API keys (never hardcode)

### Data Handling

- Use `pandas` for CSV data processing
- Use `numpy` for numerical operations
- Use `pathlib.Path` for file path operations (not os.path)
- Follow the raw/processed directory structure in `data/`
- Handle missing data explicitly with clear error messages

### Environment

- CUDA is disabled by default (`CUDA_VISIBLE_DEVICES=""`) to avoid library conflicts
- Uses CPU for docling OCR to prevent CUDA runtime errors
- Set environment variables in `.env` files (not committed to git)
