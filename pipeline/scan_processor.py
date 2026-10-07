from pathlib import Path
from PIL import Image
from ocr.dataset import OCRInferenceDataset
from detector.dbnet import DBNetDetector
from torch.utils.data import DataLoader
from ocr.dataset import collate_fn_inference
from ocr.recognizer import OCRRecognizer
from schemas.output import TextRegion

def save_crops(image, detections, temp_page_dir):

    temp_page_dir = Path(temp_page_dir)
    temp_page_dir.mkdir(parents=True, exist_ok=True)

    width, height = image.size

    crop_records = []

    for crop_id, detection in enumerate(detections):

        x1,y1, x2,y2, confidence = detection

        x1 = round(x1 * width)
        y1 = round(y1 * height)
        x2 = round(x2 * width)
        y2 = round(y2 * height)

        crop = image.crop((x1,y1,x2,y2))

        crop_path = temp_page_dir / f"crop_{crop_id:05d}.png"

        crop.save(crop_path)

        crop_records.append({
            "path": crop_path,
            "bbox": [x1, y1, x2, y2],
            "confidence":float(confidence),
            "crop_id": crop_id
        })

    return crop_records

def process_scanned_page(image, page_number, temp_page_dir, detector, recognizer):

    detections = detector.predict(image)

    detections = detections[0]["words"]

    crop_records = save_crops(
        image,
        detections,
        temp_page_dir
    )

    dataset = OCRInferenceDataset(crop_records)

    loader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=False,
        collate_fn=collate_fn_inference
    )

    results = []

    for images, input_lengths, indices in loader:

        texts = recognizer.forward(
            images,
            input_lengths
        )

        for text, index in zip(texts, indices):

            record = crop_records[index.item()]

            results.append(
                TextRegion(
                    page=page_number,
                    text=text,
                    bbox=record["bbox"],
                    source="ocr",
                    confidence=record["confidence"],
                    crop_id=record["crop_id"]
                )
            )

    return results