import pymupdf
from PIL import Image


MIN_TEXT_CHARS = 30


def has_text_layer(page, min_chars=MIN_TEXT_CHARS):
    """True, gdy strona ma użyteczną warstwę tekstową, a nie tylko drobny
    tekst (stopka, znak wodny) na skanie."""
    text = page.get_text("text").strip()

    if len(text) >= min_chars:
        return True

    if not text:
        return False

   
    return not page.get_images(full=True)


def render_page(page, scale=2.5):
    pix = page.get_pixmap(
        matrix=pymupdf.Matrix(scale, scale),
        colorspace=pymupdf.csRGB,
        alpha=False
    )

    # bez kodowania do PNG i z powrotem
    return Image.frombytes(
        "RGB",
        (pix.width, pix.height),
        pix.samples
    )


def _mark_source(page_results, is_ocr):
    """Oznacza regiony: czy pochodzą z OCR (potrzebne w format_pages, żeby
    korekty wielkości liter nie dotykały tekstu z warstwy tekstowej PDF)."""
    for item in page_results:
        if isinstance(item, dict):
            item["ocr"] = is_ocr


def process_pdf(
    pdf_path,
    detector,
    recognizer,
    temp_dir
):

    from pipeline.text_processor import process_text_page
    from pipeline.scan_processor import process_scanned_page

    results = []

    with pymupdf.open(pdf_path) as document:

        total = len(document)

        print(f"PDF: {pdf_path}")
        print(f"Liczba stron: {total}")

        for page_number, page in enumerate(document, start=1):
            print(f"Przetwarzam stronę {page_number}/{total}")

            if has_text_layer(page):

                page_results = process_text_page(
                    page,
                    page_number
                )
                is_ocr = False

            else:

                image = render_page(page)

                page_temp_dir = temp_dir / f"page_{page_number:03d}"

                page_results = process_scanned_page(
                    image=image,
                    page_number=page_number,
                    temp_page_dir=page_temp_dir,
                    detector=detector,
                    recognizer=recognizer
                )
                is_ocr = True

            _mark_source(page_results, is_ocr)

            results.extend(page_results)

    return results