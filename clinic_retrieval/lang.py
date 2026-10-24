"""Devanagari, English, and Hindi typed in Latin script.

Transliteration keys first, then the tokeniser that uses them. The
Devanagari block is listed explicitly in _WORD: \\w alone drops matras and
the virama, which breaks every Hindi word into fragments."""
from __future__ import annotations

import re

_CONSONANTS = {
    "क": "k", "ख": "kh", "ग": "g", "घ": "gh", "ङ": "ng",
    "च": "ch", "छ": "chh", "ज": "j", "झ": "jh", "ञ": "n",
    "ट": "t", "ठ": "th", "ड": "d", "ढ": "dh", "ण": "n",
    "त": "t", "थ": "th", "द": "d", "ध": "dh", "न": "n",
    "प": "p", "फ": "ph", "ब": "b", "भ": "bh", "म": "m",
    "य": "y", "र": "r", "ल": "l", "व": "v", "ळ": "l",
    "श": "sh", "ष": "sh", "स": "s", "ह": "h",
    "क़": "k", "ख़": "kh", "ग़": "g", "ज़": "z", "ड़": "r", "ढ़": "rh", "फ़": "f",
}

_VOWELS = {
    "अ": "a", "आ": "aa", "इ": "i", "ई": "ii", "उ": "u", "ऊ": "uu",
    "ऋ": "ri", "ए": "e", "ऐ": "ai", "ओ": "o", "औ": "au", "ऑ": "o",
}

_MATRAS = {
    "ा": "aa", "ि": "i", "ी": "ii", "ु": "u", "ू": "uu", "ृ": "ri",
    "े": "e", "ै": "ai", "ो": "o", "ौ": "au", "ॉ": "o", "ॅ": "e",
}

_SIGNS = {"ं": "n", "ः": "h", "ँ": "n"}
_VIRAMA = "्"
_NUKTA = "़"

_VOWELS_LATIN = set("aeiou")

# Spelling variants that turn up in romanised Hindi.
_FOLD = (("ck", "k"), ("ph", "f"), ("w", "v"), ("c", "k"), ("q", "k"), ("x", "ks"))


def is_devanagari(text: str) -> bool:
    return any("ऀ" <= ch <= "ॿ" for ch in text)


def to_latin(word: str) -> str:
    """Transliterate a Devanagari word. Latin input comes back lowercased."""
    if not is_devanagari(word):
        return word.lower()

    out: list[str] = []
    i = 0
    while i < len(word):
        ch = word[i]
        if i + 1 < len(word) and word[i + 1] == _NUKTA and ch + _NUKTA in _CONSONANTS:
            ch, i = ch + _NUKTA, i + 1

        if ch in _CONSONANTS:
            out.append(_CONSONANTS[ch])
            nxt = word[i + 1] if i + 1 < len(word) else ""
            if nxt == _VIRAMA:
                i += 2
                continue
            if nxt in _MATRAS:
                out.append(_MATRAS[nxt])
                i += 2
                continue
            out.append("a")      # inherent vowel
        elif ch in _VOWELS:
            out.append(_VOWELS[ch])
        elif ch in _MATRAS:
            out.append(_MATRAS[ch])
        elif ch in _SIGNS:
            out.append(_SIGNS[ch])
        i += 1
    return "".join(out)


def _fold(latin: str) -> str:
    for old, new in _FOLD:
        latin = latin.replace(old, new)
    out: list[str] = []
    for ch in latin:
        if not out or out[-1] != ch:
            out.append(ch)
    return "".join(out)


def latin_key(word: str) -> str:
    """Normalised Latin form. आयुष्मान and ayushman land on the same key."""
    out = _fold(to_latin(word))
    return out[:-1] if len(out) > 2 and out.endswith("a") else out


def skeleton(word: str) -> str:
    """First letter plus the consonant run. yojana and yojna collide here."""
    out = _fold(to_latin(word))
    if not out:
        return ""
    return out[0] + "".join(c for c in out[1:] if c not in _VOWELS_LATIN)


_WORD = re.compile(r"[\wऀ-ॿ']+", re.UNICODE)

_STOP_EN = {
    "the", "a", "an", "is", "are", "do", "does", "did", "what", "when", "where",
    "how", "why", "which", "who", "can", "should", "would", "will", "to", "of",
    "for", "and", "or", "in", "on", "at", "it", "this", "that", "be", "with",
    "my", "our", "we", "i", "you", "from", "has", "have", "if", "there", "about",
}

_STOP_HI = {
    "क्या", "कैसे", "कितना", "कितनी", "कितने", "कहाँ", "कहां", "कौन", "क्यों",
    "है", "हैं", "होता", "होती", "होते", "किया", "करने", "करना", "करें", "लिये",
    "लिए", "में", "से", "को", "पर", "और", "या", "यह", "वह", "कि", "ही", "का",
    "की", "के", "तो", "नहीं", "हो", "था", "थी",
}

_STOP_HINGLISH = {
    "kya", "kaise", "kitna", "kitni", "kitne", "kahan", "kaun", "kyon", "kyun",
    "hai", "hain", "hota", "hoti", "kiya", "karne", "karna", "karen", "liye",
    "mein", "me", "se", "ko", "par", "aur", "ya", "yeh", "ye", "woh", "wo",
    "ki", "ka", "ke", "hi", "to", "nahi", "nahin", "ho", "tha", "thi",
}

_STOP = _STOP_EN | _STOP_HI | _STOP_HINGLISH

# Words that mark romanised Hindi without also being English words, so "I want
# to speak to a human" is not read as Hinglish because of "to".
_HINGLISH_MARKERS = {
    "kya", "kaise", "kitna", "kitni", "kitne", "kahan", "kyun", "kyon",
    "hai", "hain", "nahi", "nahin", "karna", "karein", "karen", "chahiye",
    "mein", "milta", "milti", "bacha", "dena", "deni", "dawa", "yojna",
}


def words(text: str) -> list[str]:
    out = []
    for raw in _WORD.findall(text):
        w = raw if is_devanagari(raw) else raw.lower()
        if w in _STOP or to_latin(w) in _STOP:
            continue
        if len(w) < 2:
            continue
        out.append(w)
    return out


def keys(word: str) -> list[tuple[str, float]]:
    """Surface form, Latin form and consonant skeleton, in descending weight."""
    out = [(f"s:{word}", 1.0)]
    latin = latin_key(word)
    if latin:
        out.append((f"l:{latin}", 0.6))
    skel = skeleton(word)
    if skel and skel != latin:
        out.append((f"k:{skel}", 0.5))
    return out


def detect_form(text: str) -> str:
    """Devanagari, English, or Hindi typed on an English keyboard."""
    if is_devanagari(text):
        return "hi"
    found = {w.lower() for w in _WORD.findall(text)}
    return "hinglish" if found & _HINGLISH_MARKERS else "en"
