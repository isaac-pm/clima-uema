"""Extract structured emergency-alert data from Costa Rican weather PDFs.

Pipeline:
1. Extract PDF text with docling.
2. Extract structured fields with Google Gemini.
3. Validate with Pydantic and append to CSV.
"""

import csv
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from pydantic import BaseModel, Field
from pydantic import ValidationError as PydanticValidationError

RESET = "\033[0m"
RED = "\033[91m"
YELLOW = "\033[93m"
GREEN = "\033[92m"


class ColoredFormatter(logging.Formatter):
    """Colored log formatter for terminal output."""

    COLORS = {
        "WARNING": YELLOW,
        "ERROR": RED,
        "INFO": GREEN,
    }

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, RESET)
        record.levelname = f"{color}{record.levelname}{RESET}"
        return super().format(record)


root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)

LOG_DIR: Path = Path("logs")
LOG_FILE: Path = LOG_DIR / "emergency_alerts" / "alert_processing.log"

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(
    ColoredFormatter("%(asctime)s - %(levelname)s - %(message)s")
)
root_logger.addHandler(console_handler)

LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
file_handler = logging.FileHandler(LOG_FILE, mode="w")
file_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
)
root_logger.addHandler(file_handler)

log = logging.getLogger(__name__)

os.environ["CUDA_VISIBLE_DEVICES"] = ""


class AlertSchema(BaseModel):
    """Pydantic model for Costa Rican emergency alerts."""

    alert_number: str = Field(..., description="Alert identifier number")
    alert_category: str = Field(
        ..., description="Alert level: Green, Yellow, Orange, Red, or Cancellation"
    )
    issue_date: str = Field(..., description="Date in YYYY-MM-DD format")
    issue_time: str = Field(..., description="Time of issue in HH:MM format")
    meteorological_event: str = Field(
        default="", description="Meteorological event description (after 'ANTE:')"
    )
    regions_green_alert: Optional[list[str]] = Field(
        default=None, description="Regions under green alert"
    )
    regions_yellow_alert: Optional[list[str]] = Field(
        default=None, description="Regions under yellow alert"
    )
    regions_orange_alert: Optional[list[str]] = Field(
        default=None, description="Regions under orange alert"
    )
    regions_red_alert: Optional[list[str]] = Field(
        default=None, description="Regions under red alert"
    )
    summary_of_conditions: str = Field(
        default="", description="Summary of meteorological conditions"
    )


# --- Configuration ---

GOOGLE_API_KEY: Optional[str] = None
GEMINI_MODEL: str = "gemini-3.1-flash-lite-preview"

PDF_INPUT_DIR: Path = Path("data/emergency_alerts/raw")
OUTPUT_CSV: Path = Path("data/emergency_alerts/processed/alerts_data.csv")
FAILED_LOG: Path = LOG_DIR / "emergency_alerts" / "failed_pdfs.log"

LLM_SYSTEM_PROMPT: str = """You are an expert data extraction assistant specializing in Costa Rican emergency weather alerts.

CONTEXT: These alerts come from CNE (Comisión Nacional de Emergencias - National Emergency Commission). The documents are in Spanish.

KEY SPANISH TERMS YOU WILL SEE:
- "ANTE:" = "BECAUSE:" or "GIVEN:" - followed by the meteorological event description
- "SE INFORMA" = "IT IS INFORMED" or "REPORT" - followed by conditions summary
- "COMENTARIO" = "COMMENT" - section with weather commentary
- "ALERTA VERDE" = Green Alert (lowest level)
- "ALERTA AMARILLA" = Yellow Alert (moderate level)
- "ALERTA NARANJA" = Orange Alert (high level)
- "ALERTA ROJA" = Red Alert (maximum level)
- "LEVANTAMIENTO DE ALERTA" = Lifting/Cancellation of Alert
- "ZONA NORTE" = Northern Zone (region)
- "CARIBE" = Caribbean region
- "PACÍFICO" = Pacific region
- "CENTRAL" = Central region
- "ALAJUELA", "SAN JOSÉ", "HEREDIA", "CARTAGO", "PUNTARENAS", "LIMÓN", "GUANACASTE" = Province names

Your task: Extract structured data from the alert text below into valid JSON.

Instructions:
1. Only respond with valid JSON, no additional text
2. Use null for empty or not found fields
3. For region lists, include only the region names mentioned
4. Date must be in YYYY-MM-DD format
5. Time must be in HH:MM format (24-hour clock)
6. For "LEVANTAMIENTO" (cancellation) alerts, use "Cancellation" as alert_category
7. The alert_number must match exactly (e.g., "01-24", "02-25", "001-26")

Fields to extract:
- alert_number: Alert identifier (e.g., "01-24", "02-25")
- alert_category: Green, Yellow, Orange, Red, or Cancellation
- issue_date: Date of issue (YYYY-MM-DD)
- issue_time: Time of issue (HH:MM)
- meteorological_event: Meteorological event (text after "ANTE:" section)
- regions_green_alert: List of regions under green alert
- regions_yellow_alert: List of regions under yellow alert
- regions_orange_alert: List of regions under orange alert
- regions_red_alert: List of regions under red alert
- summary_of_conditions: Summary of conditions (2-3 sentences max from COMENTARIO/SE INFORMA)
"""

LLM_EXTRACTION_PROMPT: str = """Extract the following emergency alert data. Respond only with valid JSON:

{extracted_text}

Respond only with valid JSON."""


# --- PDF Extraction ---


def extract_text_from_pdf(pdf_path: Path) -> Optional[str]:
    """Extract text from a PDF file using docling.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        Extracted text as string, or None if extraction fails.
    """
    try:
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption

        pipeline_options = PdfPipelineOptions(do_ocr=True, do_table_structure=False)
        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )

        result = converter.convert(str(pdf_path))
        text = result.document.export_to_markdown()

        return text if text else None

    except ImportError as e:
        log.error(f"docling not installed: {e}")
        return None
    except Exception as e:
        log.error(f"Failed to extract text from {pdf_path.name}: {e}")
        return None


# --- LLM Extraction ---


def call_google_ai(
    prompt: str,
    system_prompt: str = LLM_SYSTEM_PROMPT,
    temperature: float = 0.1,
    max_tokens: int = 2048,
) -> Optional[str]:
    """Call Google AI Studio API to extract structured data from text.

    Args:
        prompt: User prompt with the extracted text.
        system_prompt: System prompt for the LLM.
        temperature: Sampling temperature.
        max_tokens: Maximum tokens to generate.

    Returns:
        LLM response text, or None if the call fails.
    """
    try:
        from google import genai

        client = genai.Client(api_key=GOOGLE_API_KEY)

        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config={
                "system_instruction": system_prompt,
                "temperature": temperature,
                "max_output_tokens": max_tokens,
                "response_mime_type": "application/json",
            },
        )

        return response.text

    except ImportError:
        log.error("google-genai package not installed")
        return None
    except Exception as e:
        log.error(f"Google AI API call failed: {e}")
        return None


def extract_structured_data(
    extracted_text: str,
) -> Optional[AlertSchema]:
    """Extract structured alert data using Google AI.

    Args:
        extracted_text: Raw text extracted from PDF.

    Returns:
        AlertSchema instance, or None if extraction fails.
    """
    prompt = LLM_EXTRACTION_PROMPT.format(extracted_text=extracted_text[:8000])

    response = call_google_ai(prompt)

    if not response:
        return None

    data: dict[str, Any] = {}
    try:
        data = json.loads(response)
        return AlertSchema(**data)

    except json.JSONDecodeError as e:
        log.error(f"Failed to parse LLM JSON response: {e}")
        log.debug(f"Raw response: {response[:500]}")
        return None
    except PydanticValidationError as e:
        log.error(f"Pydantic validation failed: {e}")
        log.debug(f"Raw data: {data}")
        return None


# --- CSV Export ---


def save_to_csv(alerts: list[AlertSchema], output_path: Path) -> None:
    """Save alerts to CSV file.

    Args:
        alerts: List of AlertSchema instances.
        output_path: Path to output CSV file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not alerts:
        log.warning("No alerts to save")
        return

    data = [alert.model_dump() for alert in alerts]
    df = pd.DataFrame(data)

    df.to_csv(output_path, index=False, encoding="utf-8")
    log.info(f"Saved {len(alerts)} alerts to {output_path}")


def append_to_csv(alert: AlertSchema, output_path: Path) -> None:
    """Append a single alert to CSV file.

    Args:
        alert: AlertSchema instance to append.
        output_path: Path to output CSV file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    file_exists = output_path.exists()

    with open(output_path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=AlertSchema.model_fields.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(alert.model_dump())


# --- Error Logging ---


def log_failed_pdf(pdf_path: Path, reason: str, failed_log: Path) -> None:
    """Log a failed PDF processing attempt.

    Args:
        pdf_path: Path to the failed PDF.
        reason: Reason for failure.
        failed_log: Path to the failure log file.
    """
    failed_log.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(failed_log, mode="w", encoding="utf-8") as f:
        f.write(f"{timestamp} | {pdf_path.name} | {reason}\n")
    log.warning(f"Logged failure for {pdf_path.name}: {reason}")


# --- Main Processing ---


def process_pdfs(
    input_dir: Path,
    output_csv: Path,
    failed_log: Path,
    year_filter: Optional[str] = None,
) -> list[AlertSchema]:
    """Process all PDFs in a directory recursively.

    Args:
        input_dir: Directory containing PDF files (searched recursively).
        output_csv: Path to output CSV file.
        failed_log: Path to failed PDFs log.
        year_filter: Optional year subdirectory to filter (e.g., "2024").

    Returns:
        List of successfully extracted AlertSchema instances.
    """
    if year_filter:
        input_dir = input_dir / year_filter

    pdf_files = sorted(input_dir.rglob("*.pdf"))

    if not pdf_files:
        log.warning(f"No PDF files found in {input_dir}")
        return []

    log.info(f"Found {len(pdf_files)} PDF files to process")

    alerts: list[AlertSchema] = []

    for pdf_path in pdf_files:
        log.info(f"Processing: {pdf_path.name}")

        extracted_text = None
        try:
            extracted_text = extract_text_from_pdf(pdf_path)

            if not extracted_text or len(extracted_text.strip()) < 50:
                log_failed_pdf(
                    pdf_path, "Empty or insufficient text extracted", failed_log
                )
                continue

        except Exception as e:
            log_failed_pdf(pdf_path, f"PDF extraction error: {e}", failed_log)
            continue

        alert = None
        try:
            alert = extract_structured_data(extracted_text)

            if not alert:
                log_failed_pdf(pdf_path, "LLM extraction returned no data", failed_log)
                continue

        except Exception as e:
            log_failed_pdf(pdf_path, f"LLM processing error: {e}", failed_log)
            continue

        try:
            append_to_csv(alert, output_csv)
            alerts.append(alert)
            log.info(f"Successfully processed: {pdf_path.name}")

        except Exception as e:
            log_failed_pdf(pdf_path, f"CSV write error: {e}", failed_log)
            continue

    return alerts


def main() -> None:
    """Main entry point for the PDF processing pipeline using Google AI."""
    global GOOGLE_API_KEY, GEMINI_MODEL
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract structured data from Costa Rican emergency alert PDFs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=PDF_INPUT_DIR,
        help="Directory containing PDF files (searched recursively)",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=OUTPUT_CSV,
        help="Output CSV file path",
    )
    parser.add_argument(
        "--year",
        type=str,
        default=None,
        choices=["2024", "2025", "2026"],
        help="Filter by year subdirectory",
    )
    parser.add_argument(
        "--google-api-key",
        type=str,
        default=None,
        help="Google AI Studio API key",
    )
    parser.add_argument(
        "--gemini-model",
        type=str,
        default=GEMINI_MODEL,
        help="Gemini model name",
    )

    args = parser.parse_args()

    if not args.google_api_key:
        parser.print_help()
        print("\nError: --google-api-key is required")
        sys.exit(1)

    GOOGLE_API_KEY = args.google_api_key
    GEMINI_MODEL = args.gemini_model

    log.info("=" * 60)
    log.info("Starting PDF Processing Pipeline")
    log.info(f"Input: {args.input_dir}")
    log.info(f"Output: {args.output_csv}")
    log.info(f"Model: Google AI Studio ({GEMINI_MODEL})")
    log.info("=" * 60)

    failed_log = FAILED_LOG

    try:
        alerts = process_pdfs(
            input_dir=args.input_dir,
            output_csv=args.output_csv,
            failed_log=failed_log,
            year_filter=args.year,
        )

        log.info("=" * 60)
        log.info(f"Processing complete: {len(alerts)} alerts extracted")
        log.info(f"Output saved to: {args.output_csv}")
        log.info(f"Failures logged to: {failed_log}")
        log.info("=" * 60)

    except KeyboardInterrupt:
        log.warning("Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        log.error(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
