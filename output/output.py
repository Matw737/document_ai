import json
from pathlib import Path

from pipeline.postprocessor import format_pages


def save_results(results, output_path):

    output_path = Path(output_path)

    data = format_pages(results)

    with open(output_path, "w", encoding="utf-8") as f:

        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2
        )

    return output_path