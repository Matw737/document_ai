from schemas.output import TextRegion

def extract_text(page):

    text = page.get_text("text")

    return text.strip()

def process_text_page(page, page_number):

    words = page.get_text("words")

    regions = []

    for word in words:

        x0, y0, x1, y1, text, block, line, word_no = word

        regions.append(
            TextRegion(
                page=page_number,
                text=text,
                bbox=[x0, y0, x1, y1],
                source="native"
            )
        )

    return regions