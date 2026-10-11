from pathlib import Path

from pipeline.docx_processor import process_docx
from pipeline.pdf_processor import process_pdf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEMP_DIR = PROJECT_ROOT / "temp"

def process_document(
    file_path,
    detector=None,
    recognizer=None,
    temp_dir=TEMP_DIR
):

    file_path = Path(file_path)

    suffix = file_path.suffix.lower()

    if suffix == ".docx":

        return process_docx(file_path)

    if suffix == ".pdf":
        return process_pdf(
            file_path,
            detector=detector,
            recognizer=recognizer,
            temp_dir=temp_dir
        )

    raise ValueError(
        f"Unsupported file format: {suffix}"
    )