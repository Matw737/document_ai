from pathlib import Path
from PIL import Image
from ocr.dataset import OCRInferenceDataset
from detector.dbnet import DBNetDetector
from torch.utils.data import DataLoader
from ocr.dataset import collate_fn_inference
from ocr.recognizer import OCRRecognizer
from schemas.output import TextRegion
import random

def save_crops(image, detections, temp_page_dir):

    temp_page_dir = Path(temp_page_dir)
    temp_page_dir.mkdir(parents=True, exist_ok=True)

    width, height = image.size

    crop_records = []

    page = {}

    for crop_id, detection in enumerate(detections):

        x1,y1, x2,y2, confidence = detection

        x1 = round(x1 * width)
        y1 = round(y1 * height)
        x2 = round(x2 * width)
        y2 = round(y2 * height)

        page[crop_id]={
            "x1":x1,
            "y1":y1,
            "x2":x2,
            "y2":y2
        }

        # crop = image.crop((x1,y1,x2,y2))

        # crop_path = temp_page_dir / f"crop_{crop_id:05d}.png"

        # crop.save(crop_path)

        # crop_records.append({
        #     "path": crop_path,
        #     "bbox": [x1, y1, x2, y2],
        #     "confidence":float(confidence),
        #     "crop_id": crop_id
        # })
    sorted_page=dict(sorted(page.items(), key=lambda item: item[1]['y1']))
    current_x1=width
    current_y1=height
    current_x2=0
    current_y2=0
    previous = height
    id_crop=1
    for d,box in sorted_page.items():
        if box["y1"] < previous:
            previous=box["y2"]
            current_x1=min(box['x1'],current_x1)
            current_y1=min(box['y1'],current_y1)
            current_x2=max(box['x2'],current_x2)
            current_y2=max(box['y2'],current_y2)
        else:
            crop=image.crop((current_x1,current_y1,current_x2,current_y2))
            crop_path=temp_page_dir /f"crop_{id_crop:05d}.png"
            crop.save(crop_path)
            crop_records.append({
                'path': crop_path,
                'bbox': [current_x1,current_y1,current_x2,current_y2],
                'confidence': random.randint(1,100),
                "crop_id":id_crop
            })
            id_crop +=1
            current_x1=width
            current_y1=height
            current_x2=0
            current_y2=0
            previous=box["y2"]
            current_x1=min(box['x1'],current_x1)
            current_y1=min(box['y1'],current_y1)
            current_x2=max(box['x2'],current_x2)
            current_y2=max(box['y2'],current_y2)
    crop = image.crop((
    current_x1,
    current_y1,
    current_x2,
    current_y2
    ))

    crop_path = temp_page_dir / f"crop_{id_crop:05d}.png"
    crop.save(crop_path)

    crop_records.append({
        "path": crop_path,
        "bbox": [current_x1, current_y1, current_x2, current_y2],
        'confidence': random.randint(1,100),
        "crop_id": id_crop
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