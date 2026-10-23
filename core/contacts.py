"""Contact whitelist: phones / emails / URLs found in SOURCE TEXT only, with authority precedence.

Precedence: gov > somaiya > other. Near-duplicate values (one or two characters apart, e.g. the Somaiya
anti-ragging page's "helplin@antiragqins.in" vs antiragging.in's "helpline@antiragging.in") are treated as
the same contact: the higher-authority value is kept and the variant is recorded in `conflict_note`.
Look-alike letters inside phone numbers ("l800-180-5522", "O9818044577") are normalised to digits and
the original spelling is recorded.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from core.schemas import Contact

RANK = {"gov": 0, "somaiya": 1, "other": 2}
_LOOKALIKE = str.maketrans({"l": "1", "I": "1", "O": "0", "o": "0"})
_D = r"[0-9lIO]"   # digit or a look-alike letter, only accepted inside a digit-dense match

PHONE_PATTERNS = [
    re.compile(r"(?<![\w@.])(?:\+91[\s-]?)?[1lI][\s-]?800[\s-]?\d{3}[\s-]?\d{4}(?!\d)"),                  # 1800 / 1-800 toll free
    re.compile(r"(?<![\w@.])\+91[\s-]?\(?0?\d{2,4}\)?[\s-]?\d{3,4}[\s-]?\d{4}(?!\d)"),                  # +91 landline
    re.compile(r"(?<![\w@.])(?:\+91[\s-]?)?[6-9]\d{4}[\s-]?\d{5}(?!\d)"),                                 # mobile
    re.compile(rf"(?<![\w@.]){_D}[6-9]\d{{9}}(?!\d)"),                                                      # 0-prefixed mobile
    re.compile(r"(?<![\w@./])\(?0\d{2,4}\)?(?:[\s-]?\d{2,5}){1,3}(?!\d)"),                            # 022-2102 3000
]
SHORT_CODE = re.compile(r"(?<!\d[\s-])(?<![\w@./,:-])(\d{3,5})(?![\w@/:]|[.,]\d|[\s-]\d)")
SHORT_CONTEXT = re.compile(r"\b(helpline|help line|call|dial|number|toll[- ]free|phone|tel|sms|whatsapp|hotline)\b", re.I)
EMAIL = re.compile(r"(?<![\w.+-])[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}(?![\w-])")
URL = re.compile(r"\bhttps?://[^\s<>()\"'\]\[{}|]+|\bwww\.[a-z0-9-]+(?:\.[a-z0-9-]+)+(?:/[^\s<>()\"']*)?", re.I)
BARE_DOMAIN = re.compile(r"(?<![@\w./-])[a-z0-9-]+(?:\.[a-z0-9-]+)*\.(?:gov\.in|nic\.in|edu\.in|org\.in|ac\.in)(?![\w-])", re.I)


@dataclass
class Mention:
    kind: str
    value: str
    display: str
    label: str
    source_id: str
    authority: str
    note: str = ""


_FILLER = {"or", "and", "on", "at", "the", "to", "via", "no", "number", "email", "e-mail", "us", "visit",
           "call", "dial", "contact", "write"}
_LEAD = re.compile(r"^((can|please|or|and)\s+)*((call|dial|contact|email|write to|visit)(\s+(the|us|at|on))?\s*)?", re.I)
_TAIL = re.compile(r"(\s*[:\-]?\s*\b(call|dial|visit|email|e-mail|at|on|or|no|number)\b\W*)+$", re.I)


def _label(text: str, start: int, end: int) -> str:
    """Short description from the words just before a mention on the same line (numbers excluded)."""
    line_start = text.rfind("\n", 0, start) + 1
    before = re.split(r"[.;!?](?:\s|$)", text[max(line_start, start - 90):start])[-1]
    segments = re.split(r"[,(){}\[\]|]|\S*\d\S*|\S+\.[a-z]{2,}\S*", before)
    for seg in reversed(segments):
        seg = re.sub(r"\s+", " ", seg).strip(" *#>-:'\"")
        words = [w for w in re.findall(r"[A-Za-z][\w'-]*", seg) if w.lower() not in _FILLER]
        if words:
            stripped = _TAIL.sub("", _LEAD.sub("", seg)).strip(" :-")
            return (stripped if re.search(r"[A-Za-z]{3}", stripped) else seg)[-70:]
    return ""


def normalise_phone(raw: str) -> tuple[str, bool]:
    """-> (value, had_lookalike_letters). E.164 via phonenumbers (region IN) when valid, else digits only."""
    fixed = raw.translate(_LOOKALIKE)
    typo = fixed != raw
    digits = re.sub(r"\D", "", fixed)
    if len(digits) <= 5:
        return digits, typo
    i = digits.find("1800")
    if i >= 0 and len(digits) - i == 11:
        return digits[i:], typo
    try:
        import phonenumbers

        num = phonenumbers.parse(fixed, "IN")
        if phonenumbers.is_valid_number(num):
            return phonenumbers.format_number(num, phonenumbers.PhoneNumberFormat.E164), typo
    except Exception:  # noqa: BLE001
        pass
    return digits, typo


def _pretty(value: str) -> str:
    if value.startswith("1800") and len(value) == 11:
        return f"{value[:4]}-{value[4:7]}-{value[7:]}"
    return value


def find_mentions(text: str, source_id: str, authority: str) -> list[Mention]:
    out: list[Mention] = []
    taken: list[tuple[int, int]] = []

    def free(a: int, b: int) -> bool:
        return all(b <= s or a >= e for s, e in taken)

    for m in EMAIL.finditer(text):
        taken.append(m.span())
        out.append(Mention("email", m.group(0).lower().rstrip("."), m.group(0), _label(text, *m.span()), source_id, authority))
    for m in URL.finditer(text):
        if not free(*m.span()):
            continue
        raw = m.group(0).rstrip(".,;:)`'\"")
        taken.append((m.start(), m.start() + len(raw)))
        if re.match(r"(?i)(https?://)?(localhost|127\.|\d+\.\d+\.\d+\.\d+)", raw):
            continue
        out.append(Mention("url", _norm_url(raw), raw, _label(text, *m.span()), source_id, authority))
    for m in BARE_DOMAIN.finditer(text):
        if free(*m.span()):
            taken.append(m.span())
            out.append(Mention("url", _norm_url(m.group(0)), m.group(0), _label(text, *m.span()), source_id, authority))
    for pat in PHONE_PATTERNS:
        for m in pat.finditer(text):
            raw = m.group(0).strip()
            n_digits = sum(ch.isdigit() for ch in raw)
            if not free(*m.span()) or n_digits < 8 or (raw.lstrip("(").startswith("0") and n_digits not in (10, 11, 12)):
                continue
            taken.append(m.span())
            value, typo = normalise_phone(raw)
            note = f"{source_id} spells it '{raw}'" if typo else ""
            display = _pretty(value) if typo else raw
            out.append(Mention("phone", value, display, _label(text, *m.span()), source_id, authority, note))
    for m in SHORT_CODE.finditer(text):
        if not free(*m.span(1)):
            continue
        lead = text[max(0, m.start() - 40):m.start()]
        if not SHORT_CONTEXT.search(lead) or re.search(r"(?i)\b(room|floor|building|plot|sector|page|section|circular|notice)(\s*no\.?)?\s*[a-z]?\s*$", lead):
            continue
        year_like = re.fullmatch(r"(19|20)\d\d", m.group(1))
        lead = text[max(0, m.start() - 30):m.start()]
        if year_like and not re.search(r"(?i)(helpline|dial|call|number|no\.?)\W{0,12}$", lead):
            continue
        taken.append(m.span(1))
        out.append(Mention("phone", m.group(1), m.group(1), _label(text, *m.span(1)), source_id, authority))
    return out


def _norm_url(u: str) -> str:
    u = u.strip().rstrip("/.,;").lower()
    u = re.sub(r"^https?://", "", u)
    return re.sub(r"^www\.", "", u)


def _similar(kind: str, a: str, b: str) -> bool:
    if a == b:
        return True
    if kind == "phone":
        da, db = re.sub(r"\D", "", a)[-10:], re.sub(r"\D", "", b)[-10:]
        return len(da) == len(db) >= 8 and sum(x != y for x, y in zip(da, db)) <= 2
    from rapidfuzz.distance import Levenshtein

    if kind == "email":
        (la, _, da), (lb, _, db) = a.partition("@"), b.partition("@")
        return Levenshtein.distance(la, lb) <= 1 and Levenshtein.distance(da, db) <= 2 and min(len(la), len(lb)) >= 4
    return min(len(a), len(b)) >= 8 and Levenshtein.distance(a, b) <= 1


def build_whitelist(mentions: list[Mention]) -> list[Contact]:
    """Group mentions by value, apply precedence, fold near-duplicate variants into the higher authority."""
    groups: dict[tuple[str, str], list[Mention]] = {}
    for m in mentions:
        groups.setdefault((m.kind, m.value), []).append(m)

    def best(ms: list[Mention]) -> Mention:
        return sorted(ms, key=lambda m: (RANK[m.authority], m.note != "", int(m.source_id.lstrip("S") or 0)
                                         if m.source_id.lstrip("S").isdigit() else 0))[0]

    keys = sorted(groups, key=lambda k: (RANK[best(groups[k]).authority], k))
    contacts: dict[tuple[str, str], Contact] = {}
    for key in keys:
        ms = groups[key]
        top = best(ms)
        notes = sorted({m.note for m in ms if m.note})
        winner = next((c for k2, c in contacts.items() if k2[0] == key[0] and _similar(key[0], k2[1], key[1])), None)
        if winner is not None:
            srcs = ",".join(sorted({m.source_id for m in ms}))
            if RANK[top.authority] > RANK[winner.authority]:
                note = f"conflict: {top.authority} source {srcs} gives '{top.display}'; kept {winner.authority} value"
            else:
                note = f"conflict: same-authority variant '{top.display}' in {srcs}; verify at review"
                contacts[key] = Contact(kind=key[0], value=key[1], display=top.display, label=top.label,
                                        source_ids=_ordered_ids(ms), authority=top.authority,
                                        conflict_note=f"variant of '{winner.display}'; verify at review")
            winner.conflict_note = "; ".join(x for x in [winner.conflict_note, note] if x)
            continue
        label = top.label or next((m.label for m in ms if m.label), "")
        contacts[key] = Contact(kind=key[0], value=key[1], display=top.display, label=label,
                                source_ids=_ordered_ids(ms), authority=top.authority,
                                conflict_note="; ".join(notes))
    return list(contacts.values())


def _ordered_ids(ms: list[Mention]) -> list[str]:
    ids = sorted({(RANK[m.authority], m.source_id) for m in ms}, key=lambda t: (t[0], _sid(t[1])))
    return list(dict.fromkeys(s for _, s in ids))


def _sid(s: str) -> int:
    return int(s[1:]) if s[1:].isdigit() else 0


def extract_contacts(docs: list[tuple[str, str, str]], allowlist=None) -> list[Contact]:
    """docs: [(source_id, authority, text)] -> whitelist.

    With an `allowlist` (core.sources.Allowlist), a URL is judged by its OWN host: URLs on non-official hosts are
    dropped and an official URL takes its host's authority. Without this, the bibliography of one government PDF
    whitelisted 88 arbitrary URLs as "gov" contacts (first Kaggle run, 2026-10-22)."""
    mentions: list[Mention] = []
    for sid, authority, text in docs:
        for m in find_mentions(text, sid, authority):
            if m.kind == "url" and allowlist is not None:
                full = m.value if "://" in m.value else f"https://{m.value}"   # values are stored scheme-less
                if not allowlist.is_allowed(full):
                    continue
                m = Mention(m.kind, m.value, m.display, m.label, m.source_id, allowlist.authority(full), m.note)
            mentions.append(m)
    return build_whitelist(mentions)
