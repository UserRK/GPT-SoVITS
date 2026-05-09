import re
from text.symbols2 import symbols

# Ukrainian Cyrillic → ARPA phoneme mapping
# Ukrainian is a phonetic language — spelling closely matches pronunciation
_uk_to_arpa = {
    # Vowels
    "а": ["AA1"], "А": ["AA1"],
    "е": ["EH1"], "Е": ["EH1"],
    "є": ["Y", "EH1"], "Є": ["Y", "EH1"],
    "и": ["IH1"], "И": ["IH1"],
    "і": ["IY1"], "І": ["IY1"],
    "ї": ["Y", "IY1"], "Ї": ["Y", "IY1"],
    "о": ["AO1"], "О": ["AO1"],
    "у": ["UW1"], "У": ["UW1"],
    "ю": ["Y", "UW1"], "Ю": ["Y", "UW1"],
    "я": ["Y", "AA1"], "Я": ["Y", "AA1"],
    # Consonants
    "б": ["B"], "Б": ["B"],
    "в": ["V"], "В": ["V"],
    "г": ["HH"], "Г": ["HH"],
    "ґ": ["G"], "Ґ": ["G"],
    "д": ["D"], "Д": ["D"],
    "ж": ["ZH"], "Ж": ["ZH"],
    "з": ["Z"], "З": ["Z"],
    "й": ["Y"], "Й": ["Y"],
    "к": ["K"], "К": ["K"],
    "л": ["L"], "Л": ["L"],
    "м": ["M"], "М": ["M"],
    "н": ["N"], "Н": ["N"],
    "п": ["P"], "П": ["P"],
    "р": ["R"], "Р": ["R"],
    "с": ["S"], "С": ["S"],
    "т": ["T"], "Т": ["T"],
    "ф": ["F"], "Ф": ["F"],
    "х": ["HH"], "Х": ["HH"],
    "ц": ["T", "S"], "Ц": ["T", "S"],
    "ч": ["CH"], "Ч": ["CH"],
    "ш": ["SH"], "Ш": ["SH"],
    "щ": ["SH", "CH"], "Щ": ["SH", "CH"],
    # Soft/hard sign — slight pause/separator, skip
    "ь": [], "Ь": [],
    "ъ": [], "Ъ": [],
    # Apostrophe (separator between consonant and iotated vowel)
    "'": [], "ʼ": [], "’": [],
}

_punctuation_map = {
    ",": ",", ".": ".", "!": "!", "?": "?",
    "…": "…", "-": "-", "—": ",", "–": ",",
    ";": ",", ":": ",", "\n": ".",
}

_digit_map = {
    "0": "нуль", "1": "один", "2": "два", "3": "три", "4": "чотири",
    "5": "п'ять", "6": "шість", "7": "сім", "8": "вісім", "9": "дев'ять",
}


def text_normalize(text):
    # Replace digits with Ukrainian words
    text = re.sub(r"\d", lambda m: _digit_map[m.group(0)] + " ", text)
    # Collapse multiple spaces
    text = re.sub(r" {2,}", " ", text).strip()
    return text


def post_replace_ph(ph):
    if ph in symbols:
        return ph
    return "UNK"


def g2p(text):
    phones = []
    for char in text:
        if char in _uk_to_arpa:
            phones.extend(_uk_to_arpa[char])
        elif char in _punctuation_map:
            phones.append(_punctuation_map[char])
        elif char == " ":
            # add a short pause between words
            if phones and phones[-1] not in (",", ".", "!", "?", "…"):
                phones.append(",")
        # skip unknown characters (latin letters, etc.)
    # ensure list ends cleanly
    if phones and phones[-1] == ",":
        phones.pop()
    phones = [post_replace_ph(p) for p in phones]
    return phones


if __name__ == "__main__":
    test = "Привіт! Як справи?"
    print(text_normalize(test))
    print(g2p(text_normalize(test)))
