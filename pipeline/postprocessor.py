# def sort_reading_order(regions):
#     return sorted(
#         regions,
#         key=lambda r: (
#             r.page,
#             r.bbox[1] if r.bbox else 0,
#             r.bbox[0] if r.bbox else 0
#         )
#     )


# def format_text(regions):

#     regions = sort_reading_order(regions)

#     output = []
#     current_page = None

#     for region in regions:

#         if region.page != current_page:

#             if output:
#                 output.append("")

#             output.append(f"--- STRONA {region.page} ---")
#             current_page = region.page

#         if region.text.strip():
#             output.append(region.text.strip())

#     return "\n".join(output)

def format_pages(regions):

    pages = {}

    for region in regions:
        pages.setdefault(region.page, []).append(region)

    output = []

    for page in sorted(pages):

        lines = group_into_lines(pages[page])

        page_lines = []

        for line in lines:

            text = " ".join(
                region.text.strip()
                for region in line
                if region.text.strip()
            )

            if text:
                page_lines.append(text)

        output.append({
            "page": page,
            "text": "\n".join(page_lines)
        })

    return output

def group_into_lines(regions, y_threshold=0.5):

    lines = []

    regions = sorted(
        regions,
        key=lambda r: r.bbox[1]
    )

    for region in regions:

        x1, y1, x2, y2 = region.bbox

        center_y = (y1 + y2) / 2
        height = y2 - y1

        best_line = None
        best_distance = float("inf")

        for line in lines:

            line_center = line["center_y"]

            distance = abs(center_y - line_center)

            # tolerancja zależna od wysokości tekstu
            tolerance = max(
                height,
                line["height"]
            ) * y_threshold

            if distance <= tolerance and distance < best_distance:
                best_line = line
                best_distance = distance

        if best_line is None:

            lines.append({
                "center_y": center_y,
                "height": height,
                "regions": [region]
            })

        else:

            best_line["regions"].append(region)

            # aktualizujemy środek linii
            ys = [
                (r.bbox[1] + r.bbox[3]) / 2
                for r in best_line["regions"]
            ]

            best_line["center_y"] = sum(ys) / len(ys)

            heights = [
                r.bbox[3] - r.bbox[1]
                for r in best_line["regions"]
            ]

            best_line["height"] = sum(heights) / len(heights)

    # sortowanie linii od góry
    lines.sort(
        key=lambda line: line["center_y"]
    )

    # sortowanie słów w linii od lewej
    for line_id, line in enumerate(lines):

        line["regions"].sort(
            key=lambda r: r.bbox[0]
        )

        for region in line["regions"]:
            region.line_id = line_id

    return [
        line["regions"]
        for line in lines
    ]

def format_lines(regions):

    pages = format_pages(regions)

    output = []

    for page in pages:

        if output:
            output.append("")

        output.append(
            f"--- STRONA {page['page']} ---"
        )

        output.append(page["text"])

    return "\n".join(output)

import matplotlib.pyplot as plt
import matplotlib.patches as patches


def plot_regions(image, regions):
    fig, ax = plt.subplots(figsize=(14, 20))

    ax.imshow(image)

    for region in regions:
        if not region.bbox:
            continue

        x1, y1, x2, y2 = region.bbox

        rect = patches.Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            fill=False,
            linewidth=1
        )

        ax.add_patch(rect)

        line_id = getattr(region, "line_id", "?")

        ax.text(
            x1,
            y1,
            f"{line_id}: {region.text}",
            fontsize=6
        )

    ax.axis("off")
    plt.tight_layout()
    plt.show()

    