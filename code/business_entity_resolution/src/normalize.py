"""
Text and Address Normalization Module for Business Entity Resolution.
Supports US, India (including Devanagari and Tamil transliteration), and France.
Handles all 30 competition edge cases: abbreviations, script transliterations,
numeric anchors, legal forms, noise removal, and country-specific rules.
"""

import re
import unicodedata
from typing import Dict, List, Set, Tuple

# Devanagari to Latin phonetic transliteration map
DEVA_MAP = {
    'क': 'k', 'ख': 'kh', 'ग': 'g', 'घ': 'gh', 'ङ': 'n',
    'च': 'ch', 'छ': 'chh', 'ज': 'j', 'झ': 'jh', 'ञ': 'n',
    'ट': 't', 'ठ': 'th', 'ड': 'd', 'ढ': 'dh', 'ण': 'n',
    'त': 't', 'थ': 'th', 'द': 'd', 'ध': 'dh', 'न': 'n',
    'प': 'p', 'फ': 'f', 'ब': 'b', 'भ': 'bh', 'म': 'm',
    'य': 'y', 'र': 'r', 'ल': 'l', 'व': 'v', 'श': 'sh', 'ष': 'sh', 'स': 's', 'ह': 'h',
    'ा': 'a', 'ि': 'i', 'ी': 'i', 'ु': 'u', 'ू': 'u', 'े': 'e', 'ै': 'ai', 'ो': 'o', 'ौ': 'au',
    '्': '', 'ं': 'n', 'ँ': 'n', 'अ': 'a', 'आ': 'a', 'इ': 'i', 'ई': 'i', 'उ': 'u', 'ऊ': 'u',
    'ए': 'e', 'ऐ': 'ai', 'ओ': 'o', 'औ': 'au', 'ऋ': 'ri', 'ॠ': 'ri'
}

# Tamil to Latin phonetic transliteration map
TAMIL_MAP = {
    'க': 'k', 'ங': 'n', 'ச': 's', 'ஞ': 'n', 'ட': 't', 'ண': 'n',
    'த': 't', 'ந': 'n', 'ப': 'p', 'ம': 'm', 'ய': 'y', 'ர': 'r',
    'ல': 'l', 'வ': 'v', 'ழ': 'zh', 'ள': 'l', 'ற': 'r', 'ன': 'n',
    'ஜ': 'j', 'ஷ': 'sh', 'ஸ': 's', 'ஹ': 'h',
    'ா': 'a', 'ி': 'i', 'ீ': 'i', 'ு': 'u', 'ூ': 'u', 'ெ': 'e', 'ே': 'e', 'ை': 'ai',
    'ொ': 'o', 'ோ': 'o', 'ௌ': 'au', '்': '',
    'அ': 'a', 'ஆ': 'a', 'இ': 'i', 'ஈ': 'i', 'உ': 'u', 'ஊ': 'u',
    'எ': 'e', 'ஏ': 'e', 'ஐ': 'ai', 'ஒ': 'o', 'ஓ': 'o', 'ஔ': 'au'
}

def transliterate_indic(text: str) -> str:
    """Transliterate Indic scripts (Devanagari, Tamil) into phonetic Latin characters."""
    if not text:
        return ""
    res = []
    for c in text:
        if c in DEVA_MAP:
            res.append(DEVA_MAP[c])
        elif c in TAMIL_MAP:
            res.append(TAMIL_MAP[c])
        else:
            res.append(c)
    return "".join(res)


# Number words to digits
NUMBER_WORDS = {
    "first": "1", "1st": "1",
    "second": "2", "2nd": "2",
    "third": "3", "3rd": "3",
    "fourth": "4", "4th": "4",
    "fifth": "5", "5th": "5",
    "sixth": "6", "6th": "6",
    "seventh": "7", "7th": "7",
    "eighth": "8", "8th": "8",
    "ninth": "9", "9th": "9",
    "tenth": "10", "10th": "10",
    "eleventh": "11", "11th": "11",
    "twelfth": "12", "12th": "12",
}

# Legal suffixes to strip
LEGAL_REGEX = re.compile(
    r"\b(pvt\s+ltd|private\s+limited|praivet\s+limited|praivet|pvt|private|"
    r"incorporated|inc|corporation|corp|company|co|limited|ltd|"
    r"llc|llp|pllc|lp|pc|elelpi|sarl|sas|sa|sci|eurl|snc|gie|groupe|"
    r"holdings|enterprises)\b",
    re.IGNORECASE
)

# Address abbreviations mapping
ADDRESS_ABBR_GENERIC = {
    "rd": "road", "street": "street",
    "ave": "avenue", "blvd": "boulevard",
    "ct": "court", "ln": "lane", "dr": "drive",
    "hwy": "highway", "pkwy": "parkway", "sq": "square",
    "ste": "suite", "apt": "apt", "apartment": "apt",
    "fl": "floor", "flr": "floor", "bldg": "building",
    "po": "pobox", "box": "pobox", "p.o.": "pobox", "pobox": "pobox",
    "township": "twp", "twn": "town",
}

# French specific address abbreviations
FRENCH_ADDR_ABBR = {
    "bd": "boulevard", "bvd": "boulevard",
    "av": "avenue", "ave": "avenue",
    "r": "rue", "rue": "rue",
    "pl": "place",
    "chem": "chemin",
    "rte": "route",
    "all": "allee",
    "imp": "impasse",
    "fbg": "faubourg",
    "st": "saint",
    "ste": "sainte",
}

# State abbreviations map
STATE_MAP = {
    # US
    "ny": "new york", "ca": "california", "tx": "texas", "fl": "florida",
    "il": "illinois", "pa": "pennsylvania", "oh": "ohio", "ga": "georgia",
    "nc": "north carolina", "mi": "michigan", "nj": "new jersey", "va": "virginia",
    "wa": "washington", "az": "arizona", "ma": "massachusetts", "tn": "tennessee",
    "in": "indiana", "mo": "missouri", "md": "maryland", "wi": "wisconsin",
    "co": "colorado", "mn": "minnesota", "sc": "south carolina", "al": "alabama",
    "la": "louisiana", "ky": "kentucky", "or": "oregon", "ok": "oklahoma",
    "ct": "connecticut", "ut": "utah", "ia": "iowa", "nv": "nevada",
    "ar": "arkansas", "ms": "mississippi", "ks": "kansas", "nm": "new mexico",
    "ne": "nebraska", "id": "idaho", "wv": "west virginia", "hi": "hawaii",
    "nh": "new hampshire", "me": "maine", "ri": "rhode island", "mt": "montana",
    "de": "delaware", "sd": "south dakota", "nd": "north dakota", "ak": "alaska",
    "vt": "vermont", "wy": "wyoming",
    # India
    "mh": "maharashtra", "dl": "delhi", "ka": "karnataka", "tn": "tamil nadu",
    "up": "uttar pradesh", "gj": "gujarat", "wb": "west bengal", "rj": "rajasthan",
    "tg": "telangana", "ap": "andhra pradesh", "mp": "madhya pradesh", "kl": "kerala",
    "pb": "punjab", "hr": "haryana", "br": "bihar", "or": "odisha",
    "jh": "jharkhand", "as": "assam", "ch": "chandigarh", "ct": "chhattisgarh",
    "uk": "uttarakhand", "ua": "uttarakhand", "goa": "goa", "hp": "himachal pradesh",
}

DOMAIN_REGEX = re.compile(
    r"^(?:https?://)?(?:www\.)?([a-z0-9\-]+)(?:\.[a-z]{2,})+(?:/.*)?$",
    re.IGNORECASE
)


def remove_accents(text: str) -> str:
    """Normalize unicode accents (e.g. Léarning -> Learning, Àmicale -> Amicale)."""
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def normalize_business_name(name: str) -> str:
    """Normalize business name: transliterates scripts, removes legal forms, noise, domains."""
    if not name:
        return ""

    text = name.strip()

    # Transliterate Indic scripts first (e.g. राम -> ram, ராஜ் -> raj)
    text = transliterate_indic(text)

    # Strip prefix noise like '<< ', '-- ', '** '
    text = re.sub(r"^[\<\>\-\*\#\s\.\,\']+", "", text)
    text = re.sub(r"[\<\>\-\*\#\s\.\,\']+$", "", text)

    # Unicode accent strip
    text = remove_accents(text)
    text = text.lower()

    # Split DBA / AKA (e.g. 'Company A D.B.A. Company B' -> 'Company B')
    dba_split = re.split(r"\b(?:d\.?b\.?a\.?|a\.?k\.?a\.?|t/a|trading as)\b", text)
    if len(dba_split) > 1:
        # If DBA present, prefer the secondary trade name if non-empty, otherwise primary
        trade_name = dba_split[-1].strip()
        if len(trade_name) >= 3:
            text = trade_name
        else:
            text = dba_split[0].strip()

    # Domain name check (e.g. earnosethroat.com -> earnosethroat)
    domain_match = DOMAIN_REGEX.match(text)
    if domain_match:
        text = domain_match.group(1)

    # Standardize & and +
    text = text.replace("&", " and ").replace("+", " plus ").replace("@", " at ")

    # Strip periods inside acronyms (s.a.s -> sas, l.l.c -> llc)
    text = re.sub(r"(?<=\b[a-zA-Z])\.(?=[a-zA-Z]\b)", "", text)
    text = text.replace(".", " ")

    # Strip legal entity suffixes
    text = LEGAL_REGEX.sub(" ", text)

    # Clean out non-alphanumeric except spaces
    cleaned = []
    for c in text:
        if c.isalnum() or c.isspace():
            cleaned.append(c)
        else:
            cleaned.append(" ")
    text = "".join(cleaned)

    # Strip leading noise articles
    tokens = text.split()
    while tokens and tokens[0] in {"the", "ms", "shri", "sri", "le", "la", "les"}:
        tokens.pop(0)

    return " ".join(tokens)


def normalize_address(address: str, country: str = "US") -> str:
    """Normalize business address with country-specific abbreviation expansions."""
    if not address:
        return ""

    text = address.strip()

    # Transliterate Indic characters if present
    text = transliterate_indic(text)

    # Strip accents
    text = remove_accents(text)
    text = text.lower()

    # Remove literal 'null'
    text = re.sub(r"\bnull\b", " ", text)

    # Replace punctuation
    text = text.replace(",", " ").replace(".", " ").replace(";", " ")
    text = text.replace("(", " ").replace(")", " ").replace("/", " ").replace("\\", " ")
    text = text.replace("#", " ").replace("-", " ")

    tokens = text.split()
    norm_tokens = []
    is_france = (country == "France")

    for tok in tokens:
        # Ordinals & number words (e.g. 1st -> 1, eleventh -> 11)
        if tok in NUMBER_WORDS:
            tok = NUMBER_WORDS[tok]
        elif re.match(r"^\d+(?:st|nd|rd|th)$", tok):
            tok = re.sub(r"[a-z]+$", "", tok)

        # Strip leading zeros on numeric tokens (0520 -> 520, 006413 -> 6413)
        if tok.isdigit():
            tok = str(int(tok))

        # Country-specific abbreviation handling (e.g. St in France is Saint, in US is Street)
        if is_france:
            expanded = FRENCH_ADDR_ABBR.get(tok, tok)
        else:
            if tok == "st":
                expanded = "street"
            else:
                expanded = ADDRESS_ABBR_GENERIC.get(tok, tok)

        # Standardize state if known
        expanded = STATE_MAP.get(expanded, expanded)
        norm_tokens.append(expanded)

    return " ".join(norm_tokens)


def extract_address_anchors(address_tokens: List[str]) -> Set[str]:
    """Extract distinctive anchor tokens: clean house numbers, PIN/ZIP codes, plot IDs."""
    anchors = set()
    for tok in address_tokens:
        if tok.isdigit():
            anchors.add(str(int(tok)))
        elif any(c.isdigit() for c in tok) and len(tok) >= 2:
            # Strip leading zeros inside mixed tokens if any
            clean_tok = re.sub(r"^0+", "", tok)
            anchors.add(clean_tok if clean_tok else tok)
    return anchors
