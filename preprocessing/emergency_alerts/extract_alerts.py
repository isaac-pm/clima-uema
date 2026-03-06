"""
Costa Rican Emergency Alert PDF Processor.

Processes PDF emergency alerts from IMN (Instituto Meteorológico Nacional)
using docling for PDF extraction and vLLM for structured data extraction.

================================================================================
USAGE INSTRUCTIONS
================================================================================

Prerequisites:
    1. Install dependencies:
        pip install -r requirements.txt

    2. Start vLLM server with a suitable model for your hardware.
       For RTX 2050 + 32GB RAM, recommended models:
        - qwen/qwen2.5-1.5b-instruct (fast, good quality)
        - qwen/qwen2.5-3b-instruct (better quality, slower)
        - llama3.2-3b-instruct (good alternative)

    Example vLLM startup:
        vllm serve qwen/qwen2.5-1.5b-instruct --dtype half

Basic Usage:
    python preprocessing/extract_alerts.py

Process specific year:
    python preprocessing/extract_alerts.py --year 2024

Process single year with verbose output:
    python preprocessing/extract_alerts.py --year 2025 -v

Custom vLLM settings:
    python preprocessing/extract_alerts.py --vllm-url http://localhost:8000/v1 --vllm-model qwen/qwen2.5-1.5b-instruct

Arguments:
    --input-dir    Directory containing PDF files (default: data/emergency_alerts/raw)
    --output-csv   Output CSV file path (default: data/emergency_alerts/processed/alerts_data.csv)
    --year         Filter by year subdirectory: 2024, 2025, or 2026
    --vllm-url     vLLM API base URL (default: http://localhost:8000/v1)
    --vllm-model   vLLM model name (default: qwen/qwen2.5-1.5b-instruct)

Output:
    - CSV file with extracted alert data
    - Log file: alert_processing.log
    - Failed PDFs log: data/emergency_alerts/processed/failed_pdfs.txt

Notes:
    - The script processes PDFs incrementally and appends to CSV
    - Failed PDFs are logged with timestamps for retry
    - Use Ctrl+C to safely interrupt processing
================================================================================
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("alert_processing.log", mode="a"),
    ],
)
log = logging.getLogger(__name__)


class AlertSchema(BaseModel):
    """Pydantic model for Costa Rican emergency alerts."""

    alert_number: str = Field(..., description="Alert identifier number")
    alert_category: str = Field(
        ..., description="Alert level: Green, Yellow, Orange, Red, or Cancellation"
    )
    issue_date: str = Field(..., description="Date in YYYY-MM-DD format")
    issue_time: str = Field(..., description="Time of issue in HH:MM format")
    signer: str = Field(..., description="Name of the signing authority")
    meteorological_event: str = Field(
        ..., description="Meteorological event description (after 'ANTE:')"
    )
    regions_green_alert: list[str] = Field(
        default_factory=list, description="Regions under green alert"
    )
    regions_yellow_alert: list[str] = Field(
        default_factory=list, description="Regions under yellow alert"
    )
    regions_orange_alert: list[str] = Field(
        default_factory=list, description="Regions under orange alert"
    )
    regions_red_alert: list[str] = Field(
        default_factory=list, description="Regions under red alert"
    )
    summary_of_conditions: str = Field(
        ..., description="Summary of meteorological conditions"
    )


# --- Configuration ---

VLLM_BASE_URL: str = "http://localhost:8000/v1"
VLLM_MODEL: str = "qwen/qwen2.5-1.5b-instruct"

PDF_INPUT_DIR: Path = Path("data/emergency_alerts/raw")
OUTPUT_CSV: Path = Path("data/emergency_alerts/processed/alerts_data.csv")
FAILED_LOG: Path = Path("data/emergency_alerts/processed/failed_pdfs.txt")

LLM_SYSTEM_PROMPT: str = """You are an expert data extraction assistant specializing in Costa Rican emergency weather alerts.

CONTEXT: These alerts come from IMN (Instituto Meteorológico Nacional - National Meteorological Institute) or CNE (Comisión Nacional de Emergencias - National Emergency Commission). The documents are in Spanish.

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
- signer: Name of the signing authority
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
        from docling.document_converter import DocumentConverter
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions

        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfPipelineOptions(
                    do_ocr=True,
                    do_table=False,
                )
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


def call_vllm(
    prompt: str,
    system_prompt: str = LLM_SYSTEM_PROMPT,
    temperature: float = 0.1,
    max_tokens: int = 2048,
) -> Optional[str]:
    """Call vLLM API to extract structured data from text.

    Args:
        prompt: User prompt with the extracted text.
        system_prompt: System prompt for the LLM.
        temperature: Sampling temperature.
        max_tokens: Maximum tokens to generate.

    Returns:
        LLM response text, or None if the call fails.
    """
    try:
        from openai import OpenAI

        client = OpenAI(
            base_url=VLLM_BASE_URL,
            api_key="dummy-key",
        )

        response = client.chat.completions.create(
            model=VLLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )

        return response.choices[0].message.content

    except ImportError:
        log.error("openai package not installed (required for vLLM client)")
        return None
    except Exception as e:
        log.error(f"vLLM API call failed: {e}")
        return None


def extract_structured_data(extracted_text: str) -> Optional[AlertSchema]:
    """Extract structured alert data using LLM.

    Args:
        extracted_text: Raw text extracted from PDF.

    Returns:
        AlertSchema instance, or None if extraction fails.
    """
    prompt = LLM_EXTRACTION_PROMPT.format(extracted_text=extracted_text[:8000])

    response = call_vllm(prompt)
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
    with open(failed_log, mode="a", encoding="utf-8") as f:
        f.write(f"{timestamp} | {pdf_path.name} | {reason}\n")
    log.warning(f"Logged failure for {pdf_path.name}: {reason}")


# --- Main Processing ---


def process_pdfs(
    input_dir: Path,
    output_csv: Path,
    failed_log: Path,
    year_filter: Optional[str] = None,
) -> list[AlertSchema]:
    """Process all PDFs in a directory.

    Args:
        input_dir: Directory containing PDF files.
        output_csv: Path to output CSV file.
        failed_log: Path to failed PDFs log.
        year_filter: Optional year subdirectory to filter (e.g., "2024").

    Returns:
        List of successfully extracted AlertSchema instances.
    """
    if year_filter:
        input_dir = input_dir / year_filter

    pdf_files = sorted(input_dir.glob("*.pdf"))

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
    """Main entry point for the PDF processing pipeline."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Process Costa Rican emergency alert PDFs"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=PDF_INPUT_DIR,
        help="Directory containing PDF files",
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
        "--vllm-url",
        type=str,
        default=VLLM_BASE_URL,
        help="vLLM API base URL",
    )
    parser.add_argument(
        "--vllm-model",
        type=str,
        default=VLLM_MODEL,
        help="vLLM model name",
    )

    args = parser.parse_args()

    global VLLM_BASE_URL, VLLM_MODEL
    VLLM_BASE_URL = args.vllm_url
    VLLM_MODEL = args.vllm_model

    log.info("=" * 60)
    log.info("Starting PDF Processing Pipeline")
    log.info(f"Input: {args.input_dir}")
    log.info(f"Output: {args.output_csv}")
    log.info(f"vLLM: {VLLM_BASE_URL}/{VLLM_MODEL}")
    log.info("=" * 60)

    failed_log = args.output_csv.parent / "failed_pdfs.txt"

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
