"""Phonetic transliteration of Indian (Brahmic) scripts into Latin letters.

All nine major Indian scripts in Unicode (Devanagari, Bengali, Gurmukhi,
Gujarati, Oriya, Tamil, Telugu, Kannada, Malayalam) share the same layout
inside their 128-code-point blocks, so one offset table covers all of them.

Unlike generic transliterators, this keeps the inherent vowel "a" that follows
every consonant unless a vowel sign or virama cancels it, and drops it at the
end of a word (Hindi-style schwa deletion). So "भारत" becomes "bharat" rather
than "bhrt", which is much closer to how the same name is written in English.
"""

_BLOCKS = (0x0900, 0x0980, 0x0A00, 0x0A80, 0x0B00, 0x0B80, 0x0C00, 0x0C80, 0x0D00)

_INDEPENDENT_VOWELS = {
    0x05: "a", 0x06: "a", 0x07: "i", 0x08: "i", 0x09: "u", 0x0A: "u", 0x0B: "ri",
    0x0C: "li", 0x0D: "e", 0x0E: "e", 0x0F: "e", 0x10: "ai", 0x11: "o", 0x12: "o",
    0x13: "o", 0x14: "au", 0x60: "ri", 0x61: "li",
}
_CONSONANTS = {
    0x15: "k", 0x16: "kh", 0x17: "g", 0x18: "gh", 0x19: "n", 0x1A: "ch", 0x1B: "chh",
    0x1C: "j", 0x1D: "jh", 0x1E: "n", 0x1F: "t", 0x20: "th", 0x21: "d", 0x22: "dh",
    0x23: "n", 0x24: "t", 0x25: "th", 0x26: "d", 0x27: "dh", 0x28: "n", 0x29: "n",
    0x2A: "p", 0x2B: "ph", 0x2C: "b", 0x2D: "bh", 0x2E: "m", 0x2F: "y", 0x30: "r",
    0x31: "r", 0x32: "l", 0x33: "l", 0x34: "l", 0x35: "v", 0x36: "sh", 0x37: "sh",
    0x38: "s", 0x39: "h", 0x58: "q", 0x59: "kh", 0x5A: "g", 0x5B: "z", 0x5C: "r",
    0x5D: "rh", 0x5E: "f", 0x5F: "y",
}
_VOWEL_SIGNS = {
    0x3E: "a", 0x3F: "i", 0x40: "i", 0x41: "u", 0x42: "u", 0x43: "ri", 0x44: "ri",
    0x45: "e", 0x46: "e", 0x47: "e", 0x48: "ai", 0x49: "o", 0x4A: "o", 0x4B: "o",
    0x4C: "au", 0x62: "li", 0x63: "li", 0x57: "au",
}
_NASAL = {0x01: "n", 0x02: "n", 0x70: "n"}  # candrabindu, anusvara, Gurmukhi tippi
_NUKTA_MAP = {"ph": "f", "j": "z", "d": "r", "dh": "rh", "k": "q", "g": "g", "kh": "kh"}
_VIRAMA = 0x4D
_NUKTA = 0x3C
# Script-specific extra letters (outside the shared layout).
_EXTRA = {
    0x09CE: "t",  # Bengali khanda ta
    0x0D7A: "n", 0x0D7B: "n", 0x0D7C: "r", 0x0D7D: "l", 0x0D7E: "l", 0x0D7F: "k",  # Malayalam chillu
}


def _offset(cp):
    """Return (offset within the shared layout) or None if cp is not Indic."""
    if 0x0900 <= cp <= 0x0D7F:
        base = _BLOCKS[(cp - 0x0900) // 0x80]
        return cp - base
    return None


def has_indic(text):
    """True if the text contains any character from the nine Indic blocks."""
    return any(0x0900 <= ord(c) <= 0x0D7F for c in text)


def transliterate(text):
    """Transliterate every Indic character in ``text``; other characters pass through."""
    out = []
    pending_a = False  # a consonant was emitted and may still take the inherent "a"
    cluster = False  # that consonant closes a cluster (e.g. "ष्ट्र"), which keeps its final "a"
    after_virama = False

    def flush_a(next_is_word_char):
        nonlocal pending_a
        if pending_a and (next_is_word_char or cluster):
            out.append("a")
        pending_a = False

    for ch in text:
        cp = ord(ch)
        if cp in _EXTRA:
            flush_a(True)
            out.append(_EXTRA[cp])
            continue
        off = _offset(cp)
        if off is None:
            # Non-Indic character: a word boundary if it is not a letter/digit.
            flush_a(ch.isalnum())
            out.append(ch)
            continue
        if off in _CONSONANTS:
            flush_a(True)
            out.append(_CONSONANTS[off])
            pending_a = True
            # clusters ending in y/r/v keep their final "a" (aditya, maharashtra);
            # others such as "rt" in "smart" do not
            cluster = after_virama and _CONSONANTS[off] in ("y", "r", "v")
            after_virama = False
            continue
        after_virama = False
        if off in _VOWEL_SIGNS:
            pending_a = False
            out.append(_VOWEL_SIGNS[off])
        elif off == _VIRAMA:
            pending_a = False
            after_virama = True
        elif off == _NUKTA:
            if out and out[-1] in _NUKTA_MAP:
                out[-1] = _NUKTA_MAP[out[-1]]
        elif off in _NASAL:
            flush_a(True)
            out.append(_NASAL[off])
        elif off == 0x03:  # visarga
            flush_a(True)
            out.append("h")
        elif off in _INDEPENDENT_VOWELS:
            flush_a(True)
            out.append(_INDEPENDENT_VOWELS[off])
        elif 0x66 <= off <= 0x6F:  # native digits
            flush_a(False)
            out.append(str(off - 0x66))
        elif off in (0x64, 0x65):  # danda / double danda
            flush_a(False)
            out.append(" ")
        # everything else (avagraha, length marks, addak, ...) is dropped
    flush_a(False)
    return "".join(out)
