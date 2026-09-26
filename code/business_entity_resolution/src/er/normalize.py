"""Normalisation of business names and addresses.

Both sides of every comparison go through the same functions, so the goal is
consistency rather than linguistic correctness: every spelling variant of a
word (``Street``/``St``/``ST.``, ``Private``/``Pvt``/``praivet``) is mapped to
one canonical token.

Pipeline for a raw string:
  1. Indian scripts are transliterated phonetically (``er.indic``); accents are
     stripped and any remaining non-ASCII text goes through ``anyascii``.
  2. Names: rename markers (formerly / f/k/a / d.b.a. / a.k.a.) split the name
     into alternative parts; website labels and @handles become extra parts;
     junk such as ``(ID: 123)`` or ``#70318`` is removed; legal-form words
     (LLC, Pvt, Ltd, SARL, ...) are set aside; abbreviations are expanded.
  3. Addresses: filler (NULL, PMB/PO box), ordinals and number words are
     normalised; street types and directions are contracted to one short form;
     state names are mapped to codes for countries whose states we know.

Country only selects extra state tables. Unknown countries get the generic
treatment and are never dropped.
"""

import re
import unicodedata

from anyascii import anyascii
from rapidfuzz import fuzz, process

from .indic import has_indic, transliterate

# --------------------------------------------------------------------------
# Character level
# --------------------------------------------------------------------------

_ORDINAL_SIGNS = str.maketrans({"º": " ", "°": " ", "ª": " "})


def to_ascii_lower(text):
    """Transliterate/strip accents and lowercase; returns plain ASCII."""
    if not text:
        return ""
    text = text.translate(_ORDINAL_SIGNS)
    if has_indic(text):
        text = transliterate(text)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    if not text.isascii():
        text = anyascii(text)
    return text.lower()


_TOKEN_RE = re.compile(r"[a-z0-9]+")
# "d.b.a." / "f/k/a" / "l.l.c." / "o.d." -> "dba" / "fka" / "llc" / "od"
_ACRONYM_DOTS_RE = re.compile(r"(?<=\b[a-z])[./]\s*(?=[a-z]\b)")
_APOSTROPHE_RE = re.compile(r"['’`]")


def _tokens(text):
    return _TOKEN_RE.findall(text)


def _dedupe_consecutive(tokens):
    out = []
    for t in tokens:
        if not out or out[-1] != t:
            out.append(t)
    return out


# --------------------------------------------------------------------------
# Phonetic skeleton (used to compare transliterated with English spellings)
# --------------------------------------------------------------------------

_SKEL_PAIRS = [
    ("ph", "f"), ("bh", "b"), ("kh", "k"), ("gh", "g"), ("th", "t"), ("dh", "d"),
    ("sh", "s"), ("chh", "c"), ("ch", "c"), ("ck", "k"), ("q", "k"), ("x", "ks"),
    ("z", "j"), ("w", "v"), ("c", "k"),
]
_VOWELS_RE = re.compile(r"[aeiouy]")
_REPEAT_RE = re.compile(r"(.)\1+")


def skeleton(token):
    """Consonant skeleton of a token: 'private'->'prvt', 'praivet'->'prvt'."""
    if not token or token.isdigit():
        return token
    s = token
    for a, b in _SKEL_PAIRS:
        s = s.replace(a, b)
    # a leading vowel is kept only as a neutral marker ("indian"/"indiyan" -> "andn",
    # "agro"/"egro" -> "agr"), since transliteration often changes it
    head = "a" if s[0] in "aeiouy" else s[0]
    s = head + _VOWELS_RE.sub("", s[1:])
    return _REPEAT_RE.sub(r"\1", s)


# --------------------------------------------------------------------------
# Names
# --------------------------------------------------------------------------

LEGAL_FORMS = {
    # English
    "llc": "llc", "inc": "inc", "incorporated": "inc", "corp": "corp", "corporation": "corp",
    "co": "co", "company": "co", "cos": "co", "ltd": "ltd", "limited": "ltd", "ltda": "ltd",
    "pvt": "pvt", "private": "pvt", "plc": "plc", "llp": "llp", "lp": "lp", "lllp": "lllp",
    "pc": "pc", "pllc": "pllc", "pa": "pa", "opc": "opc", "pty": "pty", "pte": "pte",
    # French / European
    "sa": "sa", "sas": "sas", "sasu": "sasu", "sarl": "sarl", "eurl": "eurl", "sci": "sci",
    "snc": "snc", "scop": "scop", "selarl": "selarl", "gmbh": "gmbh", "ag": "ag", "bv": "bv",
    "nv": "nv", "srl": "srl",  # not "spa": far more often a day spa than an S.p.A.
    # common transliterations of Indian legal words (Private / Limited / Company / LLP)
    "praivet": "pvt", "praibhet": "pvt", "piraivet": "pvt", "prayvet": "pvt", "pravet": "pvt",
    "limitet": "ltd", "limitad": "ltd", "limited": "ltd", "elaelapi": "llp", "kampani": "co",
    "kanpani": "co", "kampni": "co",
}
NAME_STOPWORDS = {"and", "the", "of", "a", "an", "et", "de", "du", "des", "la", "le", "les", "l", "d"}
NAME_ABBREV = {
    "intl": "international", "int'l": "international", "mgmt": "management", "mgt": "management",
    "svcs": "services", "svc": "service", "srvc": "service", "assoc": "associates",
    "assn": "association", "bros": "brothers", "mfg": "manufacturing", "univ": "university",
    "hosp": "hospital", "ctr": "center", "cntr": "center", "centre": "center", "natl": "national",
    "grp": "group", "hldgs": "holdings", "st": "saint", "ste": "sainte", "mt": "mount",
    "dept": "department", "inst": "institute", "tech": "technologies", "techs": "technologies",
    "technology": "technologies", "sys": "systems", "sols": "solutions", "soln": "solutions",
    "ent": "enterprises", "entp": "enterprises", "enterprise": "enterprises",
    "&": "and",
}

_RENAME_RE = re.compile(
    r"\b(?:formerly known as|formerly|now known as|fka|nka|dba|aka|trading as)\b[\s.:,-]*"
)
_ID_TAG_RE = re.compile(r"\(\s*id\s*[:#]?\s*\d+\s*\)|#\s*\d+")
_DOMAIN_RE = re.compile(
    r"(?:https?://)?(?:www\.)?([a-z0-9][a-z0-9-]*)\.(?:co\.in|com|net|org|in|co|biz|info|fr|us|io|de)\b"
)
_HANDLE_RE = re.compile(r"@([a-z0-9_]+)")
_MS_PREFIX_RE = re.compile(r"^\s*m\s*/\s*s\b\.?")


_DIGIT_AS_LETTER = str.maketrans({"0": "o", "1": "l", "3": "e", "4": "a", "5": "s", "6": "g", "7": "t", "8": "b"})


def _fix_digits(token):
    """Undo OCR-style digit-for-letter typos inside words: 'israe1'->'israel',
    '6lobal'->'global', 'a1l'->'all'. Tokens that are mostly digits
    ('24x7', '3m', 'b2b') are left alone."""
    if token.isalpha() or token.isdigit():
        return token
    letters = sum(c.isalpha() for c in token)
    if letters >= 2 and letters >= len(token) - 2:
        fixed = token.translate(_DIGIT_AS_LETTER)
        if fixed.isalpha():
            return fixed
    return token


def _name_core(tokens):
    """Split tokens into (core tokens, canonical legal forms)."""
    core, legal = [], []
    prev = ""
    for t in tokens:
        t = NAME_ABBREV.get(_fix_digits(t), _fix_digits(t))
        if t in LEGAL_FORMS:
            legal.append(LEGAL_FORMS[t])
        elif t == "li" and prev in ("pra", "pvt"):  # "प्रा. लि." = Pvt. Ltd.
            legal.append("ltd")
        elif t == "pra":
            legal.append("pvt")
        elif t not in NAME_STOPWORDS:
            core.append(t)
        prev = t
    return _dedupe_consecutive(core), legal


def normalize_name(raw):
    """Return a dict describing a business name.

    keys: tokens (core words), full (core as one string), nospace, skel
    (phonetic skeleton), legal (sorted legal forms), parts (alternative core
    strings: rename parts, website labels, handles), web (1 if the name is a
    website/handle).
    """
    text = to_ascii_lower(raw)
    text = _MS_PREFIX_RE.sub(" ", text)
    text = _ID_TAG_RE.sub(" ", text)
    text = text.replace("&", " and ")
    text = _APOSTROPHE_RE.sub("", text)

    parts_raw = []
    web = 0
    # website labels and @handles become alternative parts (and stay in the text)
    for m in _DOMAIN_RE.finditer(text):
        parts_raw.append(m.group(1).replace("-", ""))
        web = 1
    text = _DOMAIN_RE.sub(lambda m: " " + m.group(1).replace("-", "") + " ", text)
    for m in _HANDLE_RE.finditer(text):
        parts_raw.append(m.group(1).replace("_", ""))
        web = 1
    # "... | extra" -> the text after the pipe is only an alternative part
    if "|" in text:
        head, _, tail = text.partition("|")
        text = head
        parts_raw.append(tail)
    text = _ACRONYM_DOTS_RE.sub("", text)
    pieces = [p for p in _RENAME_RE.split(text) if p.strip()]
    if len(pieces) > 1:
        parts_raw.extend(pieces)
        text = " ".join(pieces)

    all_legal = []
    core, legal = _name_core(_tokens(text))
    all_legal.extend(legal)
    parts = []
    for p in parts_raw:
        pc, pl = _name_core(_tokens(p))
        all_legal.extend(pl)
        if pc:
            parts.append(" ".join(pc))
    full = " ".join(core)
    return {
        "tokens": core,
        "full": full,
        "nospace": full.replace(" ", ""),
        "skel": " ".join(skeleton(t) for t in core),
        "legal": " ".join(sorted(set(all_legal))),
        "parts": parts,
        "web": web,
    }


# --------------------------------------------------------------------------
# Addresses
# --------------------------------------------------------------------------

ADDR_CANON = {
    # street types (US / India / generic)
    "street": "st", "str": "st", "saint": "st", "avenue": "ave", "av": "ave", "avn": "ave",
    "aven": "ave", "road": "rd", "drive": "dr", "drv": "dr", "lane": "ln", "court": "ct",
    "crt": "ct", "circle": "cir", "circ": "cir", "boulevard": "blvd", "boul": "blvd", "bd": "blvd",
    "place": "pl", "parkway": "pkwy", "pky": "pkwy", "highway": "hwy", "hiway": "hwy",
    "terrace": "ter", "terr": "ter", "trail": "trl", "square": "sq", "sqr": "sq", "point": "pt",
    "mount": "mt", "mountain": "mtn", "heights": "hts", "expressway": "expy", "freeway": "fwy",
    "turnpike": "tpke", "crossing": "xing", "junction": "jct", "center": "ctr", "centre": "ctr",
    "plaza": "plz", "ridge": "rdg", "creek": "crk", "valley": "vly", "alley": "aly", "route": "rte",
    "rt": "rte", "sector": "sec", "near": "nr", "opposite": "opp", "behind": "bh",
    "society": "soc", "township": "twp", "taluka": "tal", "taluk": "tal", "district": "dist",
    "apartment": "apt", "apartments": "apt", "suite": "ste", "sainte": "ste", "building": "bldg",
    "floor": "fl", "flr": "fl", "room": "rm", "ground": "gr", "cross": "crs",
    # directions
    "north": "n", "south": "s", "east": "e", "west": "w", "northeast": "ne", "northwest": "nw",
    "southeast": "se", "southwest": "sw",
    # French
    "r": "rue", "chemin": "chem", "che": "chem", "impasse": "imp", "allee": "all",
    "faubourg": "fbg", "residence": "res", "batiment": "bldg", "bat": "bldg", "etage": "fl",
}
_DIGITS_RE = re.compile(r"\d+")
ORDINAL_WORDS = {
    "first": "1", "second": "2", "third": "3", "fourth": "4", "fifth": "5", "sixth": "6",
    "seventh": "7", "eighth": "8", "ninth": "9", "tenth": "10", "eleventh": "11",
    "twelfth": "12", "thirteenth": "13", "fourteenth": "14", "fifteenth": "15",
    "sixteenth": "16", "seventeenth": "17", "eighteenth": "18", "nineteenth": "19",
    "twentieth": "20", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
}
ADDR_STOPWORDS = {
    "no", "nos", "null", "none", "of", "the", "and", "city", "cedex", "nan",
    # Indian address labels that precede a number ("Door No 183", "Plot No. 10")
    "door", "plot", "flat", "shop", "hno",
}

US_STATES = {
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca",
    "colorado": "co", "connecticut": "ct", "delaware": "de", "district of columbia": "dc",
    "florida": "fl", "georgia": "ga", "hawaii": "hi", "idaho": "id", "illinois": "il",
    "indiana": "in", "iowa": "ia", "kansas": "ks", "kentucky": "ky", "louisiana": "la",
    "maine": "me", "maryland": "md", "massachusetts": "ma", "michigan": "mi", "minnesota": "mn",
    "mississippi": "ms", "missouri": "mo", "montana": "mt", "nebraska": "ne", "nevada": "nv",
    "new hampshire": "nh", "new jersey": "nj", "new mexico": "nm", "new york": "ny",
    "north carolina": "nc", "north dakota": "nd", "ohio": "oh", "oklahoma": "ok",
    "oregon": "or", "pennsylvania": "pa", "rhode island": "ri", "south carolina": "sc",
    "south dakota": "sd", "tennessee": "tn", "texas": "tx", "utah": "ut", "vermont": "vt",
    "virginia": "va", "washington": "wa", "west virginia": "wv", "wisconsin": "wi",
    "wyoming": "wy", "puerto rico": "pr", "guam": "gu",
}
INDIA_STATES = {
    "andhra pradesh": "ap", "arunachal pradesh": "ar", "assam": "as", "bihar": "br",
    "chhattisgarh": "cg", "chattisgarh": "cg", "goa": "ga", "gujarat": "gj", "haryana": "hr",
    "himachal pradesh": "hp", "jharkhand": "jh", "karnataka": "ka", "kerala": "kl",
    "madhya pradesh": "mp", "maharashtra": "mh", "manipur": "mn", "meghalaya": "ml",
    "mizoram": "mz", "nagaland": "nl", "odisha": "od", "orissa": "od", "punjab": "pb",
    "rajasthan": "rj", "sikkim": "sk", "tamil nadu": "tn", "telangana": "ts", "tripura": "tr",
    "uttar pradesh": "up", "uttarakhand": "uk", "uttaranchal": "uk", "west bengal": "wb",
    "delhi": "dl", "jammu and kashmir": "jk", "ladakh": "la", "puducherry": "py",
    "pondicherry": "py", "chandigarh": "ch", "andaman and nicobar islands": "an",
    "lakshadweep": "ld", "dadra and nagar haveli and daman and diu": "dn",
    # native-script names as produced by er.indic (not close enough for fuzzy matching)
    "pashchimabanga": "wb", "pashchimbanga": "wb", "paschimbanga": "wb", "pashchimabang": "wb",
    "orisha": "od", "odisa": "od", "panjab": "pb", "dilli": "dl", "tamilanadu": "tn",
    "tamilnadu": "tn",
}
# Codes that appear as whole address components, e.g. ", MH" or ", WB".
INDIA_CODE_ALIASES = {"ori": "od", "or": "od", "tg": "ts", "ct": "cg", "uk": "uk", "ut": "uk"}


_STATE_TABLES = {
    "us": (US_STATES, set(US_STATES.values()), {}),
    "india": (INDIA_STATES, set(INDIA_STATES.values()), INDIA_CODE_ALIASES),
}
_INDIA_STATE_NAMES = list(INDIA_STATES)

_PMB_RE = re.compile(r"\b(?:pmb|p\s*o\s*box|po\s*box|post\s*box)\s*#?\s*[a-z0-9-]+")
_ORDINAL_NUM_RE = re.compile(r"\b(\d+)(?:st|nd|rd|th)\b")
_ALNUM_SPLIT_RE = re.compile(r"^(\d+)([a-z]{1,2})$")  # "15921a" -> "15921" "a"; "5f" -> "5" "f"


def _canon_component(comp, country, from_indic):
    """Normalise one comma-separated address component; returns tokens.

    A state name is replaced by its code only when it forms the whole
    component (optionally with a postcode), so "Washington Street" stays a
    street while ", Washington," becomes "wa".
    """
    toks = _tokens(comp)
    table = _STATE_TABLES.get(country)
    if table is not None and toks:
        names, codes, aliases = table
        key = " ".join(t for t in toks if not t.isdigit())
        code = names.get(key) or (aliases.get(key) if " " not in key else None)
        if code is None and country == "india" and from_indic and key and key not in codes:
            # transliterated state names are close but not exact ("karnatak")
            hit = process.extractOne(key, _INDIA_STATE_NAMES, scorer=fuzz.ratio, score_cutoff=85)
            if hit:
                code = INDIA_STATES[hit[0]]
        if code is not None:
            toks = [code] + [t for t in toks if t.isdigit()]
    out = []
    for tok in toks:
        m = _ALNUM_SPLIT_RE.match(tok)
        for t in (m.groups() if m else (tok,)):
            t = ORDINAL_WORDS.get(t, t)
            if t.isdigit():
                t = t.lstrip("0") or "0"
            else:
                t = ADDR_CANON.get(t, t)
            if t not in ADDR_STOPWORDS:
                out.append(t)
    return out


def normalize_address(raw, country):
    """Return a dict describing an address.

    keys: tokens, full, nums (numbers in order), house (first number or ''),
    comps (list of component strings).
    """
    from_indic = has_indic(raw)
    text = to_ascii_lower(raw)
    text = _PMB_RE.sub(" ", text)
    text = _ORDINAL_NUM_RE.sub(r"\1", text)
    text = _APOSTROPHE_RE.sub("", text)
    ckey = (country or "").strip().lower()
    comps, tokens = [], []
    for comp in re.split(r"[,;\n]", text):
        ct = _canon_component(comp, ckey, from_indic)
        if ct:
            comps.append(" ".join(ct))
            tokens.extend(ct)
    tokens = _dedupe_consecutive(tokens)
    nums = [d.lstrip("0") or "0" for t in tokens for d in _DIGITS_RE.findall(t)]
    return {
        "tokens": tokens,
        "full": " ".join(tokens),
        "nums": nums,
        "house": nums[0] if nums else "",
        "comps": comps,
    }
