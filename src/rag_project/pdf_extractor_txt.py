import re

import pdfplumber

DIGIT_RUN = re.compile(r"[0-9٠-٩۰-۹]+(?:[.,٫٬][0-9٠-٩۰-۹]+)*")

def has_arabic_letter(s: str) -> bool:
    return any(0x0621 <= ord(ch) <= 0x064A for ch in s)
MIRROR_MAP = str.maketrans("()[]{}<>", ")(][}{><")


def is_digit_token(token: str) -> bool:
    return not has_arabic_letter(token)


def fix_word(token: str) -> str:
    # pdfplumber extracts each *Arabic-letter* word's characters in
    # reversed order (it walks glyphs left-to-right by x-position,
    # which is backward for RTL script). Pure digit/punctuation/paren
    # tokens (numbers, dates, "(13)") are LTR-embedded runs and are
    # already extracted in correct order — reversing them would break
    # them, so only reverse when the token actually contains Arabic
    # letters.
    if not has_arabic_letter(token):
        return token

    reversed_token = token[::-1].translate(MIRROR_MAP)
    # If a digit run got caught up in a mixed token (e.g. Arabic
    # letters glued directly to a number with no space), restore that
    # run's correct left-to-right digit order.
    return DIGIT_RUN.sub(lambda m: m.group()[::-1], reversed_token)


def extract_arabic_column(crop) -> str:
    words = crop.extract_words()

    # Group words into lines by vertical position ('top'), since
    # extract_words() gives individual words, not lines.
    lines = []
    tolerance = 3  # points; widen/narrow if lines merge or split wrongly
    for w in sorted(words, key=lambda w: w["top"]):
        placed = False
        for line in lines:
            if abs(line[0]["top"] - w["top"]) <= tolerance:
                line.append(w)
                placed = True
                break
        if not placed:
            lines.append([w])

    page_lines = []
    for line in lines:
        # Right-to-left word order (rightmost word = read first)
        line_sorted = sorted(line, key=lambda w: -w["x0"])
        fixed_words = [fix_word(w["text"]) for w in line_sorted]
        page_lines.append(" ".join(fixed_words))

    return "\n".join(page_lines)


def process_bilingual_pdf(
    pdf_path: str,
    arabic_out_path: str = "arabic_output.txt",
    english_out_path: str = "english_output.txt",
):
    arabic_pages = []
    english_pages = []

    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            width = page.width
            height = page.height

            # Crop Right Column (Arabic)
            crop_ar = page.crop((width / 2, 0, width, height))
            fixed_ar = extract_arabic_column(crop_ar)

            # Crop Left Column (English) — unchanged
            crop_en = page.crop((0, 0, width / 2, height))
            fixed_en = crop_en.extract_text() or ""

            # Save page contents
            arabic_pages.append(f"--- PAGE {i+1} ---\n" + fixed_ar)
            english_pages.append(f"--- PAGE {i+1} ---\n" + fixed_en)

    # Write processed Arabic text
    with open(arabic_out_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(arabic_pages))

    # Write extracted English text
    with open(english_out_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(english_pages))

    print(
        f"Processing complete! Saved to '{arabic_out_path}' and '{english_out_path}'."
    )


if __name__ == "__main__":
    PDF_FILE = "1576751803.pdf"

    process_bilingual_pdf(PDF_FILE)