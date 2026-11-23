from pathlib import Path
import argparse
import shutil
import os
import torch
import stat
from pipeline.postprocessor import format_lines
from output.output import save_results
from detector.dbnet import DBNetDetector
from ocr.recognizer import OCRRecognizer
from pipeline.router import process_document

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = Path(__file__).resolve().parent
TEMP_DIR = BASE_DIR / "temp"

def main(file_path, output_path=None):

    file_path = Path(file_path)

    if output_path is None:
        output_path = BASE_DIR / "output.json"
    else:
        output_path = Path(output_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Input file does not exist: {file_path}"
        )

    if not file_path.is_file():
        raise ValueError(
            f"Input path is not a file: {file_path}"
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    checkpoint = torch.load(
        BASE_DIR / "models" / "ocr_v3.0_best.pt",
        map_location="cpu",
        weights_only=False
    )

    detector = DBNetDetector()

    recognizer = OCRRecognizer(
        checkpoint["model_state_dict"],
        device=device
    )

    try:

        results = process_document(
            file_path,
            detector=detector,
            recognizer=recognizer
        )

        print(f"\nLiczba regionów: {len(results)}\n")
        print(format_lines(results))

        save_results(
            results,
            output_path
        )

        print(f"\nJSON zapisany: {output_path}")

        return results

    finally:
        cleanup_temp(TEMP_DIR)
        pass

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Document AI OCR"
    )

    parser.add_argument(
        "file_path",
        type=Path,
        help="Path to input PDF or DOCX file"
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to output JSON file"
    )

    args = parser.parse_args()

    main(args.file_path, args.output)