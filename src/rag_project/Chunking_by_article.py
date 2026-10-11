import re
import json
import arabic_reshaper
from bidi.algorithm import get_display


def show_ar(text):
    print(get_display(arabic_reshaper.reshape(text)))


PAGE_MARKER = re.compile(r"^\s*--- PAGE (\d+) ---\s*$")

# ---------------------------------------------------------------------
# English
# ---------------------------------------------------------------------

LEVEL_PATTERN_EN = re.compile(r"^(Book|Part|Title|Chapter|Section)\s+[IVXLCDM\d]+\s*$", re.IGNORECASE)
ARTICLE_LINE_EN = re.compile(r"^Article\s+(\d+)\s*$")

def split_with_headings_en(path):
    articles = {}
    current_num = None
    body_lines = []
    current = {"book": None, "chapter": None, "section": None}
    pending_level = None
    page = None

    def flush():
        if current_num is not None:
            articles[current_num] = {
                "text": "\n".join(body_lines).strip(),
                "source_page": page,
                **current,
            }

    with open(path, encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()

            pm = PAGE_MARKER.match(line)
            if pm:
                page = int(pm.group(1))
                continue

            if not line:
                continue

            m = ARTICLE_LINE_EN.match(line)
            if m:
                flush()
                current_num = int(m.group(1))
                body_lines = []
                continue

            lvl = LEVEL_PATTERN_EN.match(line)
            if lvl:
                word = lvl.group(1).lower()
                field = "book" if word in ("book", "part", "title") else word
                pending_level = field
                continue

            if pending_level:
                current[pending_level] = line
                pending_level = None
                continue

            if current_num is not None:
                body_lines.append(line)

    flush()
    return articles


en_articles = split_with_headings_en("english_output.txt")

# ---------------------------------------------------------------------
# Arabic
# ---------------------------------------------------------------------

GAP = r"[\s\u200e\u200f]*"
PAREN = r"[()]*"
ARTICLE_LINE_AR = re.compile(
    rf"^{GAP}مادة{GAP}{PAREN}{GAP}([٠-٩0-9]+){GAP}{PAREN}{GAP}$",
    flags=re.MULTILINE,
)
LEVEL_PATTERN_AR = re.compile(r"^(الكتاب|الباب|الفصل)\s+\S+\s*$")
AR_FIELD = {"الكتاب": "book", "الباب": "chapter", "الفصل": "section"}
AR_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")

def split_with_headings_ar(path):
    with open(path, encoding="utf-8") as f:
        text = f.read()

    articles = {}
    current_num = None
    body_lines = []
    current = {"book": None, "chapter": None, "section": None}
    pending_level = None
    page = None

    for raw in text.split("\n"):
        line = raw.strip()

        pm = PAGE_MARKER.match(line)
        if pm:
            page = int(pm.group(1))
            continue

        if not line:
            continue

        m = ARTICLE_LINE_AR.match(line)
        if m:
            if current_num is not None:
                articles[current_num] = {
                    "text": "\n".join(body_lines).strip(),
                    "source_page": page,
                    **current,
                }
            current_num = int(m.group(1).translate(AR_DIGITS))
            body_lines = []
            continue

        lvl = LEVEL_PATTERN_AR.match(line)
        if lvl:
            pending_level = AR_FIELD[lvl.group(1)]
            continue

        if pending_level:
            current[pending_level] = line
            pending_level = None
            continue

        if current_num is not None:
            body_lines.append(line)

    if current_num is not None:
        articles[current_num] = {
            "text": "\n".join(body_lines).strip(),
            "source_page": page,
            **current,
        }
    return articles


ar_articles = split_with_headings_ar("arabic_output.txt")

# ---------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------

print(en_articles[89])
show_ar(ar_articles[89]["text"])
print(ar_articles[89]["book"], ar_articles[89]["chapter"], ar_articles[89]["section"], ar_articles[89]["source_page"])

only_en = sorted(set(en_articles) - set(ar_articles))
only_ar = sorted(set(ar_articles) - set(en_articles))
print("only in English:", only_en)
print("only in Arabic:", only_ar)

# ---------------------------------------------------------------------
# Merge + save
# ---------------------------------------------------------------------

records = []
for num, en in en_articles.items():
    ar = ar_articles.get(num, {"text": "", "book": None, "chapter": None, "section": None, "source_page": None})
    records.append({
        "article_number": num,
        "book_en": en["book"],
        "chapter_en": en["chapter"],
        "section_en": en["section"],
        "book_ar":ar["book"],
        "chapter_ar": ar["chapter"],
        "section_ar": ar["section"],        
        "source_page_en": en["source_page"],
        "source_page_ar": ar["source_page"],
        "text_en": en["text"],
        "text_ar": ar["text"],
        "citation": f"Egyptian Civil Code, Article {num}",
    })

records.sort(key=lambda r: r["article_number"])

with open("civil_code.json", "w", encoding="utf-8") as f:
    json.dump(records, f, ensure_ascii=False, indent=2)

print(f"saved {len(records)} records to civil_code.json")