"""Deterministic display -> spoken text normalisation for TTS (Kokoro), plus word counting.

The Scriptwriter asks the LLM for both forms, but the spoken form that reaches TTS is always
`to_spoken(display)`, so pronunciation never depends on the LLM. Rules (PLAN §1.3, §6.4):

- Phone / helpline numbers: read **digit by digit**, groups kept as written and separated by a comma
  pause: "1800-180-5522" -> "one eight zero zero, one eight zero, five five two two", "1930" -> "one nine
  three zero". A bare number is a phone number when it is in the contacts whitelist or near a phone word
  (call, dial, helpline...). Digit-by-digit is unambiguous for listeners and for the ASR/WER check.
- 4-digit numbers become years ("twenty twenty-six") only when the context says year (in/since/by/month
  names/"Act"/"Rules"/"Regulations"...); otherwise a bare 4-digit number is read digit by digit, unless a
  counting noun follows ("2000 students" -> "two thousand students").
- Money: "₹2,000" / "Rs. 2000" / "INR 2000" -> "two thousand rupees" (Indian lakh/crore system).
- ALL-CAPS acronyms are spelled with spaces ("KJSCE" -> "K J S C E") unless listed in PRONOUNCEABLE
  ("POSH" -> "posh", "APAAR" -> "apaar") or a common English word written in caps ("NOT" -> "not").
- URLs are spoken host-only: "https://www.somaiya.edu/en/..." -> "somaiya dot E D U"; emails:
  "helpline@antiragging.in" -> "helpline at antiragging dot in". The caption keeps the full URL.
- Spelled letters are upper case ("E D U", "A C") because Kokoro reads a lower-case "a" as the article.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Iterable

# ---------------------------------------------------------------- vocabularies
ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
        "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
ORDINAL_IRREGULAR = {"one": "first", "two": "second", "three": "third", "five": "fifth", "eight": "eighth",
                     "nine": "ninth", "twelve": "twelfth"}
MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august", "september",
          "october", "november", "december"]
MONTH_ABBR = {m[:3] for m in MONTHS} | {"sept"}

# ALL-CAPS words read as words, not letters (extend in one place).
PRONOUNCEABLE: dict[str, str] = {
    "POSH": "posh", "APAAR": "apaar", "NASA": "nasa", "UNESCO": "unesco", "UNICEF": "unicef",
    "COVID": "covid", "MANAS": "manas", "SWAYAM": "swayam", "NEET": "neet", "GATE": "gate",
    "SOMAIYA": "somaiya", "OK": "okay", "PIN": "pin", "OTP": "O T P", "WHO": "W H O",
    "LASER": "laser", "RADAR": "radar", "SIM": "sim", "ASAP": "A S A P",
}
# Common English words that appear in caps for emphasis ("NOT allowed"): read as words.
CAPS_WORDS = {
    "NOT", "NO", "DO", "DONT", "DON'T", "AND", "OR", "THE", "YOU", "YOUR", "ALL", "NEW", "FREE", "STOP",
    "NOW", "YES", "HELP", "ASK", "READ", "NEVER", "ALWAYS", "ONLY", "BE", "IS", "ARE", "CAN", "CAN'T",
    "WILL", "WON'T", "MUST", "SAFE", "CALL", "CHECK", "WHAT", "WHY", "HOW", "WHEN", "WHERE", "THIS",
    "THAT", "FOR", "TO", "IN", "ON", "AT", "OF", "WITH", "BEFORE", "AFTER", "EXAM", "EXAMS", "RULES",
    "ALLOWED", "BANNED", "REPORT", "SPEAK", "UP", "OUT", "GET", "KNOW", "IMPORTANT", "NOTE", "TIP",
    "FACT", "MYTH", "TRUE", "FALSE", "WARNING", "ALERT", "FIRST", "LAST", "DAY", "LAB",
}
ABBREVIATIONS: list[tuple[str, str]] = [
    (r"\bPh\.?D\.?(?=\W|$)", "P H D"), (r"\bB\.\s?Tech\b", "B tech"), (r"\bM\.\s?Tech\b", "M tech"),
    (r"\bB\.\s?E\.(?=\W|$)", "B E"), (r"\bM\.\s?E\.(?=\W|$)", "M E"), (r"\bB\.\s?Sc\b\.?", "B S C"),
    (r"\bM\.\s?Sc\b\.?", "M S C"), (r"\bCoE\b", "C O E"), (r"\bIoT\b", "I O T"), (r"\bMeitY\b", "my tee"),
    (r"\bDr\.", "Doctor"), (r"\bProf\.", "Professor"), (r"\bGovt\.?(?=\W|$)", "government"),
    (r"\bDept\.?(?=\W|$)", "department"), (r"\be\.g\.", "for example"), (r"\bi\.e\.", "that is"),
    (r"\betc\.", "et cetera"), (r"\bvs\.?(?=\W|$)", "versus"), (r"\bapprox\.", "approximately"),
    (r"\bNo\.\s?(?=\d)", "number "), (r"\bSr\.", "senior"), (r"\bSt\.", "saint"),
]
TLD_SPEAK = {"edu": "E D U", "in": "in", "gov": "gov", "com": "com", "org": "org", "net": "net",
             "ac": "A C", "co": "co", "res": "res", "nic": "nic", "info": "info", "io": "I O"}
SPELL_LABELS = {"kjsce", "ugc", "pib", "aicte", "ncw", "nirf", "naac", "mhrd", "coe", "svu", "kjssc",
                "nta", "cbse", "ncert", "uidai", "dpdp", "nep", "kjsim", "kjssm", "mu", "iitb"}
PHONE_WORDS = {"call", "calling", "dial", "helpline", "helplines", "hotline", "number", "numbers", "toll",
               "toll-free", "tollfree", "phone", "sms", "whatsapp", "contact", "ring", "text", "line",
               "emergency", "tele-manas", "telemanas"}
YEAR_PREV = {"in", "since", "year", "years", "from", "by", "until", "till", "before", "after", "during",
             "batch", "academic", "session", "regulations", "regulation", "act", "rules", "rule", "policy",
             "circular", "dated", "class", "of", "fy", "ay", "between", "around", "circa", "guidelines",
             "amendment", "notification", "budget", "summer", "winter", "monsoon", "spring", "autumn"}
YEAR_NEXT = {"regulations", "regulation", "rules", "act", "batch", "edition", "onwards", "session",
             "guidelines", "policy", "amendment", "notification", "circular", "admissions", "cycle"}
COUNT_NOUNS = {"students", "people", "rupees", "marks", "seats", "hours", "minutes", "seconds", "days",
               "weeks", "months", "years", "words", "pages", "colleges", "teams", "members", "participants",
               "applications", "cases", "complaints", "calls", "points", "credits", "steps", "times",
               "lakh", "crore", "thousand", "hundred", "million", "billion", "per", "percent", "km", "kg"}

_URL = re.compile(r"(?i)\b(?:https?://)?(?:www\.)?(?:[a-z0-9-]+\.)+(?:edu|in|gov|com|org|net|ac|co|info|io|res)"
                  r"(?:\.[a-z]{2})?(?![a-z0-9-])(?:/[^\s,;)\]]*)?")
_EMAIL = re.compile(r"(?i)\b[a-z0-9._%+-]+@(?:[a-z0-9-]+\.)+[a-z]{2,}\b")
_CAPS = re.compile(r"\b([A-Z][A-Z0-9&]*[A-Z0-9])(['’]?s)?\b")


# ---------------------------------------------------------------- numbers
def _under_1000(n: int) -> str:
    parts: list[str] = []
    if n >= 100:
        parts.append(f"{ONES[n // 100]} hundred")
        n %= 100
    if n >= 20:
        parts.append(TENS[n // 10] + (f"-{ONES[n % 10]}" if n % 10 else ""))
    elif n > 0 or not parts:
        parts.append(ONES[n])
    return " ".join(parts)


def cardinal(n: int, indian: bool = False) -> str:
    """Integer -> words. `indian=True` uses lakh/crore grouping (₹ amounts)."""
    if n < 0:
        return "minus " + cardinal(-n, indian)
    if n < 1000:
        return _under_1000(n)
    scales = [(10**7, "crore"), (10**5, "lakh"), (1000, "thousand")] if indian else \
        [(10**12, "trillion"), (10**9, "billion"), (10**6, "million"), (1000, "thousand")]
    parts: list[str] = []
    for value, name in scales:
        if n >= value:
            head = n // value
            parts.append(f"{cardinal(head, indian) if head >= 1000 else _under_1000(head)} {name}")
            n %= value
    if n:
        parts.append(_under_1000(n))
    return " ".join(parts)


def ordinal(n: int) -> str:
    words = cardinal(n)
    head, _, last = words.rpartition(" ")
    sep = " " if head else ""
    if "-" in last:
        a, b = last.rsplit("-", 1)
        return f"{head}{sep}{a}-{ORDINAL_IRREGULAR.get(b, b + 'th')}"
    if last in ORDINAL_IRREGULAR:
        return f"{head}{sep}{ORDINAL_IRREGULAR[last]}"
    if last.endswith("y"):
        return f"{head}{sep}{last[:-1]}ieth"
    return f"{head}{sep}{last}th"


def year_words(y: int) -> str:
    """2026 -> 'twenty twenty-six', 2005 -> 'two thousand five', 1905 -> 'nineteen oh five'."""
    if 2000 <= y <= 2009:
        return "two thousand" + (f" {ONES[y % 10]}" if y % 10 else "")
    hi, lo = divmod(y, 100)
    if lo == 0:
        return f"{_under_1000(hi)} hundred"
    if lo < 10:
        return f"{_under_1000(hi)} oh {ONES[lo]}"
    return f"{_under_1000(hi)} {_under_1000(lo)}"


def digits(s: str) -> str:
    """'1930' -> 'one nine three zero' (non-digits ignored)."""
    return " ".join(ONES[int(c)] for c in s if c.isdigit())


def phone_words(s: str) -> str:
    """Digit by digit, keeping the written grouping as comma pauses; leading + -> 'plus'."""
    s = s.strip()
    plus = "plus " if s.startswith("+") else ""
    groups = [g for g in re.split(r"[\s\-–.()]+", s.lstrip("+")) if g]
    return plus + ", ".join(digits(g) for g in groups)


def _is_indian_grouping(num: str) -> bool:
    return bool(re.fullmatch(r"\d{1,2}(,\d\d)+,\d{3}", num))


def _num(num: str, indian: bool = False) -> str:
    """'2,000' / '2000' / '2.5' -> words."""
    clean = num.replace(",", "")
    if "." in clean:
        whole, frac = clean.split(".", 1)
        return f"{cardinal(int(whole or 0), indian)} point {digits(frac)}"
    return cardinal(int(clean), indian or _is_indian_grouping(num))


# ---------------------------------------------------------------- URL / email / acronyms
def _spell(s: str) -> str:
    return " ".join(ONES[int(c)] if c.isdigit() else c.upper() for c in s if c.isalnum())


def _speak_label(label: str) -> str:
    low = label.lower()
    if low in TLD_SPEAK:
        return TLD_SPEAK[low]
    if low in SPELL_LABELS or len(low) <= 2 or not re.search(r"[aeiouy]", low):
        return _spell(low)
    parts = [p for p in re.split(r"[-_]", low) if p]
    return " ".join(_spell(p) if p.isdigit() or not re.search(r"[aeiouy]", p) else p for p in parts)


def speak_url(url: str) -> str:
    """Host only: 'https://www.somaiya.edu/en/x' -> 'somaiya dot E D U'."""
    host = re.sub(r"(?i)^https?://", "", url.strip()).split("/")[0]
    host = re.sub(r"(?i)^www\.", "", host).rstrip(".")
    return " dot ".join(_speak_label(lab) for lab in host.split(".") if lab)


def speak_email(email: str) -> str:
    local, _, host = email.partition("@")
    local_words = " dot ".join(" ".join(_speak_label(p) for p in re.split(r"[_\-+]", seg) if p)
                               for seg in local.split(".") if seg)
    return f"{local_words} at {speak_url(host)}"


def speak_acronym(word: str) -> str:
    """'KJSCE' -> 'K J S C E'; 'POSH' -> 'posh'; 'NOT' -> 'not'."""
    up = word.upper()
    if up in PRONOUNCEABLE:
        return PRONOUNCEABLE[up]
    if up in CAPS_WORDS:
        return word.lower()
    return " ".join(_spell(ch) if ch.isalnum() else "and" for ch in word if ch.isalnum() or ch == "&")


# ---------------------------------------------------------------- contacts
def contact_digits(contacts: Iterable[dict] | None) -> set[str]:
    """Digit strings of whitelisted phones (with and without +91 / leading 0)."""
    out: set[str] = set()
    for c in contacts or []:
        if c.get("kind") != "phone":
            continue
        for v in (c.get("value", ""), c.get("display", "")):
            d = re.sub(r"\D", "", v)
            if d:
                out.add(d)
                if d.startswith("91") and len(d) > 10:
                    out.add(d[2:])
                out.add(d.lstrip("0"))
    return out


# ---------------------------------------------------------------- main entry
def _words_around(text: str, start: int, end: int, n: int) -> tuple[list[str], list[str]]:
    prev = re.findall(r"[A-Za-z0-9][A-Za-z0-9\-']*", text[:start])[-n:]
    nxt = re.findall(r"[A-Za-z0-9][A-Za-z0-9\-']*", text[end:])[:n]
    return prev, nxt


def _bare_number(text: str, m: re.Match, phones: set[str]) -> str:
    """Order: whitelist phone > counting noun > year word adjacent > phone word nearby > acronym before /
    year word in the next 3 words > default (4 or 6+ digits: digit by digit, else cardinal)."""
    d = m.group(0)
    raw_prev, raw_next = _words_around(text, m.start(), m.end(), 5)
    prev5, next5 = [w.lower() for w in raw_prev], [w.lower() for w in raw_next]
    prev1, next1 = prev5[-1:], next5[:1]
    acronym_before = bool(raw_prev) and bool(re.fullmatch(r"[A-Z]{2,}", raw_prev[-1]))   # "NEP 2020", "UGC 2018"
    if len(d) <= 2 and not d.startswith("0"):
        return cardinal(int(d))
    if d in phones or d.lstrip("0") in phones:
        return digits(d)
    if next1 and next1[0] in COUNT_NOUNS and not d.startswith("0"):
        return cardinal(int(d))
    is_yearish = len(d) == 4 and 1000 <= int(d) <= 2099
    month_before = bool(prev1) and (prev1[0] in MONTHS or prev1[0] in MONTH_ABBR)
    if is_yearish and ((prev1 and prev1[0] in YEAR_PREV) or month_before or (next1 and next1[0] in YEAR_NEXT)):
        return year_words(int(d))
    if PHONE_WORDS & (set(prev5) | set(next5)) or d.startswith("0"):
        return digits(d)
    if is_yearish and (acronym_before or YEAR_NEXT & set(next5[:3])):
        return year_words(int(d))
    if len(d) == 4 or len(d) >= 6:
        return digits(d)
    return cardinal(int(d))


def to_spoken(display: str, contacts: Iterable[dict] | None = None) -> str:
    """Deterministic TTS-safe form of a caption sentence (see module docstring)."""
    phones = contact_digits(contacts)
    t = unicodedata.normalize("NFKC", display)
    t = t.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    t = re.sub(r"[\U0001F000-\U0001FFFF☀-➿️]", " ", t)          # emoji / dingbats

    t = _EMAIL.sub(lambda m: f" {speak_email(m.group(0))} ", t)
    t = _URL.sub(lambda m: f" {speak_url(m.group(0))} ", t)

    # dates dd/mm/yyyy or dd-mm-yyyy (Indian order)
    def _date(m: re.Match) -> str:
        d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if 1 <= mo <= 12 and 1 <= d <= 31:
            return f"{ordinal(d)} {MONTHS[mo - 1].capitalize()} {year_words(y)}"
        return m.group(0)
    t = re.sub(r"\b(\d{1,2})[/\-.](\d{1,2})[/\-.]((?:19|20)\d\d)\b", _date, t)

    # identifiers like COE/134/2022 -> parts
    def _ident(m: re.Match) -> str:
        out = []
        for p in m.group(0).split("/"):
            if p.isdigit():
                out.append(year_words(int(p)) if len(p) == 4 and p[:2] in ("19", "20") else cardinal(int(p)))
            else:
                out.append(p)
        return " ".join(out)
    t = re.sub(r"\b[A-Za-z]+(?:/[A-Za-z0-9]+)+\b", _ident, t)

    for pat, rep in ABBREVIATIONS:
        t = re.sub(pat, rep, t)
    t = re.sub(r"\b24\s?[x×/]\s?7\b", "twenty-four seven", t, flags=re.I)
    t = re.sub(r"\b(\d+)\s?[x×]\s?(\d+)\b", lambda m: f"{m.group(1)} by {m.group(2)}", t)

    # money
    def _money(m: re.Match) -> str:
        amount, scale = m.group(1), (m.group(2) or "").lower()
        words = _num(amount, indian=True)
        return f" {words}{' ' + scale if scale else ''} rupees"
    t = re.sub(r"(?:₹|\bRs\.?|\bINR)\s?(\d[\d,]*(?:\.\d+)?)(?:\s?/-)?(?:\s?(lakh|lakhs|crore|crores)\b)?",
               _money, t, flags=re.I)   # "Rs.500/-" is the common Indian form; "/-" is not spoken
    t = re.sub(r"(\d[\d,]*(?:\.\d+)?)\s?(%|percent\b)", lambda m: f"{_num(m.group(1))} percent", t)

    # year ranges 2024-25 / 2023–2024 (before phone detection)
    def _yrange(m: re.Match) -> str:
        a, b = m.group(1), m.group(2)
        b_full = int(a[:2] + b) if len(b) == 2 else int(b)
        return f"{year_words(int(a))} to {year_words(b_full) if len(b) == 4 else _under_1000(int(b))}"
    t = re.sub(r"\b((?:19|20)\d\d)\s?[-–]\s?(\d{4}|\d\d)\b", _yrange, t)

    # phone numbers with separators (>= 7 digits in total)
    def _phone(m: re.Match) -> str:
        s = m.group(0)
        return f" {phone_words(s)} " if len(re.sub(r"\D", "", s)) >= 7 else s
    t = re.sub(r"(?<![\w.])\+?\d{1,5}(?:[ \-–]\d{2,5}){1,4}(?![\w])", _phone, t)
    t = re.sub(r"(?<![\w.])\+\d{7,13}\b", lambda m: f" {phone_words(m.group(0))} ", t)

    # numeric ranges 10-15 -> ten to fifteen
    t = re.sub(r"\b(\d+)\s?[–-]\s?(\d+)\b", lambda m: f"{m.group(1)} to {m.group(2)}", t)
    # ordinals 1st 2nd 3rd 21st
    t = re.sub(r"\b(\d+)(st|nd|rd|th)\b", lambda m: ordinal(int(m.group(1))), t, flags=re.I)
    # times 10:30 am
    def _time(m: re.Match) -> str:
        h, mi, ap = int(m.group(1)), int(m.group(2)), (m.group(3) or "").lower().replace(".", "")
        mins = "" if mi == 0 else (f" oh {ONES[mi]}" if mi < 10 else f" {_under_1000(mi)}")
        tail = f" {_spell(ap)}" if ap else ("" if mi else " o'clock")
        return f"{cardinal(h)}{mins}{tail}" + ("." if m.group(0).endswith(".") else "")
    t = re.sub(r"\b(\d{1,2}):(\d{2})\s?([ap]\.?m\.?)?(?=\W|$)", _time, t, flags=re.I)
    t = re.sub(r"\b(\d{1,2})\s?([ap])\.?m\.?(?=\W|$)", lambda m: f"{cardinal(int(m.group(1)))} {m.group(2).upper()} M",
               t, flags=re.I)
    # grouped / decimal numbers
    t = re.sub(r"\b\d{1,3}(?:,\d{2,3})+(?:\.\d+)?\b", lambda m: _num(m.group(0)), t)
    t = re.sub(r"\b\d+\.\d+\b", lambda m: _num(m.group(0)), t)
    # bare integers (context-sensitive)
    t = re.sub(r"\b\d+\b", lambda m: _bare_number(t, m, phones), t)

    # acronyms (after URLs / numbers so their outputs are untouched)
    def _caps(m: re.Match) -> str:
        word, plural = m.group(1), m.group(2) or ""
        if len(re.sub(r"[^A-Z]", "", word)) < 2:
            return m.group(0)
        spoken = speak_acronym(word)
        return spoken + ("s" if plural and spoken.islower() else ("'s" if plural else ""))
    t = _CAPS.sub(_caps, t)

    # symbols
    t = t.replace("&", " and ").replace("@", " at ").replace("+", " plus ").replace("#", " number ")
    t = re.sub(r"\s*/\s*", " or ", t)
    t = re.sub(r"\s*[—–]\s*", ", ", t)
    t = re.sub(r"(?<=\w)-(?=\s)|(?<=\s)-(?=\w)", " ", t)
    t = re.sub(r"[^A-Za-z0-9\s,.?!'’;:\-]", " ", t)
    t = re.sub(r"\d+", lambda m: digits(m.group(0)), t)        # safety net: no digit reaches TTS
    t = re.sub(r"\s+([,.?!;:])", r"\1", t)
    t = re.sub(r"([,;:])(?:\s*[,;:])+", r"\1", t)
    t = re.sub(r"\s{2,}", " ", t).strip(" ,;:")
    return t


def spoken_issues(spoken: str) -> list[str]:
    """Things TTS would mis-read: digits, symbols, unspelled ALL-CAPS words."""
    issues = []
    if re.search(r"\d", spoken):
        issues.append("digits")
    if re.search(r"[₹$%&@/#+=<>|_*~^]", spoken):
        issues.append("symbols")
    for w in re.findall(r"\b[A-Z]{2,}\b", spoken):
        if w not in CAPS_WORDS and w.upper() not in PRONOUNCEABLE:
            issues.append(f"caps:{w}")
    return issues


def count_words(text: str) -> int:
    """Whitespace tokens that contain a letter or digit ("1800-180-5522" counts as one)."""
    return sum(1 for tok in text.split() if re.search(r"[A-Za-z0-9₹]", tok))


def shorten(text: str, max_words: int = 7) -> str:
    """Trim a sentence to <= max_words for on-screen text (drops leading fillers, trailing stopwords)."""
    words = re.sub(r"[\"“”]", "", text).split()
    while words and words[0].lower().strip(",") in {"and", "so", "but", "also", "now", "here's", "remember,", "remember"}:
        words = words[1:]
    if len(words) > max_words:      # drop articles / fillers before cutting
        slim = [w for w in words if w.lower() not in {"the", "a", "an", "really", "just", "very", "actually"}]
        words = slim if len(slim) >= 2 else words
    if len(words) > max_words:      # prefer a natural break (comma / colon) that keeps >= 3 words
        cut = next((i + 1 for i in range(max_words - 1, 1, -1) if words[i].endswith((",", ":", ";"))), max_words)
        words = words[:cut]
    while len(words) > 1 and words[-1].lower().strip(",.;:") in {"a", "an", "the", "and", "or", "of", "to", "for",
                                                                  "in", "on", "with", "by", "your", "is", "are"}:
        words = words[:-1]
    out = " ".join(words).rstrip(",;:—-")
    out = out[:1].upper() + out[1:]
    return out.rstrip(".") if not out.endswith("?") else out


# ---------------------------------------------------------------- contacts in display text
_SEP_PHONE = re.compile(r"(?<![\w.])\+?\d{1,5}(?:[ \-–]\d{2,5}){1,4}(?![\w])")
_BARE_NUM = re.compile(r"(?<![\w.,/:-])\d{3,13}(?![\w/:]|[.,]\d)")


def _norm_url(u: str) -> str:
    u = re.sub(r"(?i)^https?://", "", u.strip()).lower()
    u = re.sub(r"^www\.", "", u)
    return u.rstrip("/.")


def find_contacts(text: str, contacts: Iterable[dict] | None = None) -> list[dict]:
    """Phone numbers, emails and URLs mentioned in a display sentence: [{kind, text, span}]."""
    phones = contact_digits(contacts)
    found: list[dict] = []
    taken: list[tuple[int, int]] = []

    def add(kind: str, m: re.Match) -> None:
        if any(a < m.end() and m.start() < b for a, b in taken):
            return
        taken.append((m.start(), m.end()))
        found.append({"kind": kind, "text": m.group(0).rstrip("."), "span": (m.start(), m.end())})

    for m in _EMAIL.finditer(text):
        add("email", m)
    for m in _URL.finditer(text):
        add("url", m)
    for m in re.finditer(r"\b(19|20)\d\d\s?[-–]\s?(\d{4}|\d\d)\b", text):   # year ranges are not phones
        taken.append((m.start(), m.end()))
    for m in _SEP_PHONE.finditer(text):
        if len(re.sub(r"\D", "", m.group(0))) >= 7:
            add("phone", m)
    for m in _BARE_NUM.finditer(text):
        d = m.group(0)
        if d in phones or len(d) >= 7:
            add("phone", m)
            continue
        raw_prev, raw_next = _words_around(text, m.start(), m.end(), 5)
        prev5, next5 = {w.lower() for w in raw_prev}, {w.lower() for w in raw_next}
        prev1 = raw_prev[-1].lower() if raw_prev else ""
        yearish = len(d) == 4 and d[:2] in ("19", "20") and (prev1 in YEAR_PREV or prev1 in MONTHS)
        if PHONE_WORDS & (prev5 | next5) and not yearish and not (raw_next and raw_next[0].lower() in COUNT_NOUNS):
            add("phone", m)
    return sorted(found, key=lambda f: f["span"][0])


def contact_allowed(item: dict, contacts: Iterable[dict] | None, extra_urls: Iterable[str] = ()) -> bool:
    """Exact whitelist match: phone digits, lower-case email, URL equal to (or a shorter prefix of) a
    whitelisted URL or a fact source URL."""
    contacts = list(contacts or [])
    if item["kind"] == "phone":
        d = re.sub(r"\D", "", item["text"])
        allowed = contact_digits(contacts)
        return d in allowed or d.lstrip("0") in allowed or (d.startswith("91") and d[2:] in allowed)
    if item["kind"] == "email":
        return item["text"].lower() in {c.get("value", "").lower() for c in contacts if c.get("kind") == "email"} \
            | {c.get("display", "").lower() for c in contacts if c.get("kind") == "email"}
    u = _norm_url(item["text"])
    pool = [_norm_url(c.get("value", "")) for c in contacts if c.get("kind") == "url"]
    pool += [_norm_url(c.get("display", "")) for c in contacts if c.get("kind") == "url"]
    pool += [_norm_url(x) for x in extra_urls]
    return any(u and (p == u or p.startswith(u + "/")) for p in pool if p) or \
        any(u == p.split("/")[0] for p in pool if p)


SAFE_CONTACT_TEXT = {"phone": "the official helpline", "email": "the official email address",
                     "url": "the official website"}


def strip_unlisted_contacts(text: str, contacts: Iterable[dict] | None,
                            extra_urls: Iterable[str] = ()) -> tuple[str, list[str]]:
    """Replace contacts not in the whitelist with a neutral phrase. Returns (text, removed)."""
    contacts = list(contacts or [])
    removed: list[str] = []
    for item in reversed(find_contacts(text, contacts)):
        if contact_allowed(item, contacts, extra_urls):
            continue
        a, b = item["span"]
        tail = "." if text[a:b].endswith(".") else ""
        text = text[:a] + SAFE_CONTACT_TEXT[item["kind"]] + tail + text[b:]
        removed.append(item["text"])
    return text, removed[::-1]


_URL_PREFIX = re.compile(r"\bhttps?://(?:www\.)?", re.I)
_URL_TRAILING_SLASH = re.compile(r"(\b[\w.-]+\.[a-z]{2,}(?:/[\w%.~-]+)*)/(?=[\s.,;!?)]|$)", re.I)


def short_urls(text: str) -> str:
    """Display form of URLs: 'https://www.somaiya.edu/en/x/' -> 'somaiya.edu/en/x'. Shorter captions and cards;
    the critic's whitelist check normalises URLs the same way, so verification is unaffected."""
    return _URL_TRAILING_SLASH.sub(r"\1", _URL_PREFIX.sub("", text or ""))
