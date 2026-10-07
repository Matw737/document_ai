from dataclasses import dataclass
from typing import Optional

@dataclass
class TextRegion:
    page: int
    text: str
    bbox: list[float]
    source: str
    confidence: Optional[float] = None
    crop_id: Optional[int] = None
    line_id: Optional[int] = None