from pathlib import Path

from pipeline.docx_processor import process_docx
from pipeline.pdf_processor import process_pdf

def process_document(
    file_path,
    detector=None,
    recognizer=None,
    temp_dir=Path("temp")
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