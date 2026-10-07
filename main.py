from pathlib import Path
import torch
from pipeline.postprocessor import format_lines
from detector.dbnet import DBNetDetector
from ocr.recognizer import OCRRecognizer

from pipeline.router import process_document


BASE_DIR = Path(__file__).resolve().parent

checkpoint = torch.load(
    BASE_DIR / "models" / "ocr_v2.1_best.pt",
    map_location="cpu",
    weights_only=False
)

detector = DBNetDetector()

recognizer = OCRRecognizer(
    checkpoint["model_state_dict"],
    device="cuda"
)

file_path = Path(
    r"C:\Users\matwi\Downloads\umowa_ocr_test_1_strona.pdf"
)

results = process_document(
    file_path,
    detector=detector,
    recognizer=recognizer
)

print(f"Liczba regionów: {len(results)}")

print(format_lines(results))