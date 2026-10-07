import pymupdf


def has_text_layer(page):

    text = page.get_text("text")

    return bool(text.strip())


def render_page(page, scale=2.5):

    matrix = pymupdf.Matrix(scale, scale)

    pix = page.get_pixmap(
        matrix=matrix,
        alpha=False
    )

    image_bytes = pix.tobytes("png")

    from PIL import Image
    import io

    image = Image.open(
        io.BytesIO(image_bytes)
    )

    return image


def process_pdf(
    pdf_path,
    detector,
    recognizer,
    temp_dir
):

    from pipeline.text_processor import process_text_page
    from pipeline.scan_processor import process_scanned_page

    document = pymupdf.open(pdf_path)

    results = []

    for page_number, page in enumerate(document, start=1):

        if has_text_layer(page):

            page_results = process_text_page(
                page,
                page_number
            )

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

        results.extend(page_results)

    document.close()

    return results