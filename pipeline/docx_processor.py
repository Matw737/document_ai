from docx import Document

from schemas.output import TextRegion

def process_docx(docx_path):

    document = Document(docx_path)

    regions = []

    for paragraph in document.paragraphs:

        text = paragraph.text.strip()

        if not text:
            continue

        regions.append(
            TextRegion(
                page=1,
                text = text,
                bbox = [],
                source = "docx"
            )
        )

    return regions