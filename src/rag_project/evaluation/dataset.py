import re
from pathlib import Path


def load_golden(path: Path) -> list[dict]:
    """
    Load the golden evaluation dataset.

    Expected format:

        ID: 001
        Question (EN): ...
        Answer (EN): ...
        Question (AR): ...
        Answer (AR): ...
        Source: Article 147
    """

    prefixes = {
        "q_en": "Question (EN):",
        "a_en": "Answer (EN):",
        "q_ar": "Question (AR):",
        "a_ar": "Answer (AR):",
    }

    items = []
    current = None

    with open(path, encoding="utf-8") as file:

        for raw_line in file:

            line = raw_line.strip()

            if not line:
                continue

            # -----------------------------
            # New test case
            # -----------------------------

            if line.startswith("ID:"):

                current = {
                    "id": line.split(":", 1)[1].strip()
                }

                items.append(current)

                continue

            if current is None:
                continue

            # -----------------------------
            # Questions / answers
            # -----------------------------

            for key, prefix in prefixes.items():

                if line.startswith(prefix):

                    current[key] = (
                        line[len(prefix):].strip()
                    )

                    break

            # -----------------------------
            # Source article
            # -----------------------------

            if line.startswith("Source:"):

                match = re.search(r"\d+", line)

                if match:
                    current["article"] = int(
                        match.group()
                    )

    return items