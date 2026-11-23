

def format_pages(regions):

    pages = {}

    for region in regions:
        pages.setdefault(region.page, []).append(region)

    output = []

    for page in sorted(pages):

        lines = group_into_lines(pages[page])

        page_lines = []
        line_data = []

        for line in lines:

            text = " ".join(
                region.text.strip()
                for region in line["regions"]
                if region.text.strip()
            )

            if text:
                page_lines.append(text)

                line_data.append({
                    "text": text,
                    "bbox": line["bbox"],
                    "center_y": line["center_y"],
                    "height": line["height"]
                })
        for i in range(len(line_data) - 1):

            current = line_data[i]
            next_line = line_data[i + 1]

            gap = next_line["bbox"][1] - current["bbox"][3]

            avg_height = (
                current["height"] + next_line["height"]
            ) / 2

            current["gap_after"] = gap
            current["gap_ratio"] = (
                round(gap / avg_height, 2)
                if avg_height else 0
            )
        output.append({
            "page": page,
            "text": "\n".join(page_lines),
            "lines": line_data
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

        line["bbox"] = (
            min(r.bbox[0] for r in line["regions"]),
            min(r.bbox[1] for r in line["regions"]),
            max(r.bbox[2] for r in line["regions"]),
            max(r.bbox[3] for r in line["regions"])
        )

    return lines

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

    