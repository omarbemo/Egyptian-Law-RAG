import re
from collections import Counter

def article_numbers(path, pattern):
    text = open(path, encoding="utf-8").read()
    return [int(m.group(1)) for m in re.finditer(pattern, text, flags=re.MULTILINE)]

en = article_numbers("english_output.txt", r"^Article\s+(\d+)\s*$")

counts_en = Counter(en)
duplicates_en = sorted(n for n, c in counts_en.items() if c > 1)
missing_en = sorted(set(range(1, max(en) + 1)) - set(en))
print("English:")
print("duplicates:", duplicates_en)
print("missing:", len(missing_en), missing_en)


AR_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")

def arabic_article_numbers(path):
    text = open(path, encoding="utf-8").read()
    GAP = r"[\s\u200e\u200f]*"
    PAREN = r"[()]*"
    pattern = rf"^{GAP}مادة{GAP}{PAREN}{GAP}([٠-٩0-9]+){GAP}{PAREN}{GAP}$"
    return [int(m.group(1).translate(AR_DIGITS))
            for m in re.finditer(pattern, text, flags=re.MULTILINE)]

ar = arabic_article_numbers("arabic_output.txt")
counts_ar = Counter(ar)
duplicates_ar = sorted(n for n, c in counts_ar.items() if c > 1)
missing_ar = sorted(set(range(1, max(en) + 1)) - set(ar))

print("Arabic:")
print("duplicates:", duplicates_ar)
print("missing:", len(missing_ar), missing_ar)

print(set(missing_ar)-set(missing_en))
