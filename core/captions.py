"""Captions: spoken-token timings -> display-word timings -> caption pages -> .srt, plus VO WER.

Rules (PLAN §6.6 / §9): pages of <= 2 lines, <= 32 chars/line, <= 17 CPS where possible, >= 0.83 s,
and a page never contains words from two sentences.
"""
from __future__ import annotations

import re
from difflib import SequenceMatcher
from pathlib import Path

from core.schemas import CaptionPage, WordTiming

MAX_LINES = 2
MAX_CHARS = 32
MAX_CPS = 17.0
MIN_DUR = 0.83
GAP_CLOSE_S = 0.3          # close gaps shorter than this between pages (avoids caption flicker)
WORDS_PER_S = 2.4          # fallback speech rate when no timings exist


# ---------------------------------------------------------------- word alignment
def norm_word(w: str) -> str:
    """Lowercase alphanumerics only (used to match display words to spoken tokens)."""
    return "".join(ch for ch in w.lower() if ch.isalnum())


def _distribute(words: list[str], start: float, end: float, si: int) -> list[WordTiming]:
    """Spread words over [start, end] proportionally to their length."""
    if not words:
        return []
    end = max(end, start)
    weights = [len(w) + 1 for w in words]
    total = float(sum(weights))
    out, t = [], start
    for w, wt in zip(words, weights):
        d = (end - start) * wt / total
        out.append(WordTiming(word=w, start=round(t, 3), end=round(t + d, 3), sentence_index=si))
        t += d
    return out


def _split_long(wt: WordTiming, max_chars: int) -> list[WordTiming]:
    """Break an over-long token (URLs) at / - . boundaries so it can fit on a caption line."""
    if len(wt.word) <= max_chars:
        return [wt]
    parts = [p for p in re.split(r"(?<=[/\-.])", wt.word) if p]
    pieces: list[str] = []
    for p in parts:
        if pieces and len(pieces[-1]) + len(p) <= max_chars:
            pieces[-1] += p
        else:
            while len(p) > max_chars:  # no separator to break on: hard cut
                pieces.append(p[:max_chars])
                p = p[max_chars:]
            pieces.append(p)
    return _distribute(pieces, wt.start, wt.end, wt.sentence_index)


def _align_sentence(dwords: list[str], toks: list[dict], span: tuple[float, float], si: int) -> list[WordTiming]:
    if not toks:
        return _distribute(dwords, span[0], span[1], si)
    a = [norm_word(d) for d in dwords]
    b = [norm_word(str(t["text"])) for t in toks]
    out: list[WordTiming] = []
    for tag, i1, i2, j1, j2 in SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if tag == "equal":
            for k in range(i2 - i1):
                t = toks[j1 + k]
                out.append(WordTiming(word=dwords[i1 + k], start=round(float(t["start"]), 3),
                                      end=round(float(t["end"]), 3), sentence_index=si))
        elif tag == "replace":   # e.g. "₹2,000" <-> "two thousand rupees": share the spoken run's span
            out.extend(_distribute(dwords[i1:i2], float(toks[j1]["start"]), float(toks[j2 - 1]["end"]), si))
        elif tag == "delete":    # display-only words (symbols): squeeze into the gap between neighbours
            prev_end = float(toks[j1 - 1]["end"]) if j1 > 0 else span[0]
            next_start = float(toks[j1]["start"]) if j1 < len(toks) else max(span[1], prev_end)
            out.extend(_distribute(dwords[i1:i2], prev_end, max(prev_end, next_start), si))
        # "insert": spoken-only tokens have no display word; their time is simply not shown
    # enforce monotonic, non-negative durations
    last = span[0]
    for w in out:
        w.start = max(w.start, last)
        w.end = max(w.end, w.start)
        last = w.start
    return out


def align_display_words(display_sentences: list[str], tokens: list[dict],
                        sentence_spans: list[tuple[float, float]] | list[list[float]],
                        fallback_words: list[dict] | None = None,
                        max_chars: int = MAX_CHARS) -> list[WordTiming]:
    """Map spoken-token timings (TTS tokens: {text,start,end,sentence_index}) onto display words.

    Per sentence, display words are matched to spoken tokens by normalised text; unmatched runs share the
    time span of the corresponding spoken run proportionally (by length). Falls back to `fallback_words`
    (ASR words inside the sentence span) and then to proportional distribution over the sentence span.
    """
    out: list[WordTiming] = []
    cursor = 0.0
    for si, display in enumerate(display_sentences):
        dwords = display.split()
        if not dwords:
            continue
        if si < len(sentence_spans):
            span = (float(sentence_spans[si][0]), float(sentence_spans[si][1]))
        else:
            span = (cursor, cursor + len(dwords) / WORDS_PER_S)
        toks = [t for t in tokens
                if int(t.get("sentence_index", -1)) == si and norm_word(str(t.get("text", "")))
                and t.get("start") is not None and t.get("end") is not None]
        if not toks and fallback_words:
            toks = [{"text": w["word"], "start": w["start"], "end": w["end"]} for w in fallback_words
                    if span[0] - 0.05 <= float(w["start"]) < span[1] and norm_word(str(w["word"]))]
        for wt in _align_sentence(dwords, toks, span, si):
            out.extend(_split_long(wt, max_chars))
        cursor = span[1]
    return out


# ---------------------------------------------------------------- paging
def wrap_lines(words: list[str], max_chars: int = MAX_CHARS, max_lines: int = MAX_LINES) -> list[str] | None:
    """Fit words into <= max_lines lines of <= max_chars (balanced). None if impossible."""
    if not words:
        return []
    text = " ".join(words)
    if len(text) <= max_chars:
        return [text]
    if max_lines < 2:
        return None
    best: tuple[int, list[str]] | None = None
    for i in range(1, len(words)):
        first = " ".join(words[:i])
        rest = wrap_lines(words[i:], max_chars, max_lines - 1)
        if len(first) > max_chars or rest is None:
            continue
        score = max(len(first), *(len(r) for r in rest))
        if best is None or score < best[0]:
            best = (score, [first, *rest])
    return best[1] if best else None


def _page_chars(words: list[WordTiming]) -> int:
    return len(" ".join(w.word for w in words))


def _min_pages(words: list[WordTiming], max_chars: int, max_lines: int) -> int:
    k, cur = 0, []
    for w in words:
        if cur and wrap_lines([x.word for x in cur + [w]], max_chars, max_lines) is None:
            k, cur = k + 1, []
        cur.append(w)
    return k + (1 if cur else 0)


def _partition(words: list[WordTiming], k: int, max_chars: int, max_lines: int,
               min_dur: float, max_cps: float) -> list[list[WordTiming]] | None:
    """Split one sentence into exactly k pages: balanced length, prefer breaks after punctuation."""
    n = len(words)
    target = _page_chars(words) / k
    INF = float("inf")

    memo: dict[tuple[int, int], float] = {}

    def cost(i: int, j: int) -> float:
        if (i, j) not in memo:
            memo[(i, j)] = _cost(i, j)
        return memo[(i, j)]

    def _cost(i: int, j: int) -> float:
        seg = words[i:j]
        if sum(len(w.word) + 1 for w in seg) - 1 > max_chars * max_lines:
            return INF
        if wrap_lines([w.word for w in seg], max_chars, max_lines) is None:
            return INF
        chars = _page_chars(seg)
        dur = max(seg[-1].end - seg[0].start, 1e-3)
        c = (chars - target) ** 2
        if dur < min_dur:
            c += 400 * (min_dur - dur)
        if chars / dur > max_cps:
            c += 10 * (chars / dur - max_cps)
        if j < n and seg[-1].word[-1:] in ",;:—–":
            c -= 60
        if j - i == 1 and n > 1:
            c += 80                                   # avoid orphan single-word pages
        return c

    best = [[INF] * (n + 1) for _ in range(k + 1)]
    back = [[-1] * (n + 1) for _ in range(k + 1)]
    best[0][0] = 0.0
    for p in range(1, k + 1):
        for j in range(1, n + 1):
            for i in range(p - 1, j):
                if best[p - 1][i] == INF:
                    continue
                c = cost(i, j)
                if c == INF:
                    continue
                if best[p - 1][i] + c < best[p][j]:
                    best[p][j], back[p][j] = best[p - 1][i] + c, i
    if best[k][n] == INF:
        return None
    cuts, j = [], n
    for p in range(k, 0, -1):
        i = back[p][j]
        cuts.append((i, j))
        j = i
    return [words[i:j] for i, j in reversed(cuts)]


def _greedy(words: list[WordTiming], max_chars: int, max_lines: int) -> list[list[WordTiming]]:
    pages, cur = [], []
    for w in words:
        if cur and wrap_lines([x.word for x in cur + [w]], max_chars, max_lines) is None:
            pages.append(cur)
            cur = []
        cur.append(w)
    if cur:
        pages.append(cur)
    return pages


def paginate(words: list[WordTiming], *, max_lines: int = MAX_LINES, max_chars: int = MAX_CHARS,
             max_cps: float = MAX_CPS, min_dur: float = MIN_DUR, end_time: float | None = None) -> list[CaptionPage]:
    """Build caption pages. Text never crosses a sentence boundary; display time may extend into pauses."""
    groups: list[list[WordTiming]] = []
    for w in words:
        if groups and groups[-1][-1].sentence_index == w.sentence_index:
            groups[-1].append(w)
        else:
            groups.append([w])
    chunks: list[list[WordTiming]] = []
    for g in groups:
        k = _min_pages(g, max_chars, max_lines)
        chunks.extend(_partition(g, k, max_chars, max_lines, min_dur, max_cps) or _greedy(g, max_chars, max_lines))

    pages: list[CaptionPage] = []
    for idx, seg in enumerate(chunks):
        lines = wrap_lines([w.word for w in seg], max_chars, max_lines) or [" ".join(w.word for w in seg)]
        pages.append(CaptionPage(index=idx + 1, start=seg[0].start, end=seg[-1].end, lines=lines, words=seg))
    # timing: extend for min duration / CPS / gap closing, never overlapping the next page
    for i, p in enumerate(pages):
        chars = len(" ".join(p.lines))
        want = max(p.end, p.start + min_dur, p.start + chars / max_cps)
        nxt = pages[i + 1].start if i + 1 < len(pages) else (end_time if end_time is not None else want)
        if nxt - want < GAP_CLOSE_S and i + 1 < len(pages):
            want = nxt
        p.end = round(max(p.end, min(want, nxt)), 3)
        p.start = round(p.start, 3)
    return pages


def check_pages(pages: list[CaptionPage], *, max_lines: int = MAX_LINES, max_chars: int = MAX_CHARS,
                max_cps: float = MAX_CPS, min_dur: float = MIN_DUR) -> list[str]:
    """Readability violations (for the Critic): empty list = all pages pass."""
    issues = []
    for p in pages:
        dur = p.end - p.start
        chars = len(" ".join(p.lines))
        if len(p.lines) > max_lines:
            issues.append(f"page {p.index}: {len(p.lines)} lines")
        for ln in p.lines:
            if len(ln) > max_chars:
                issues.append(f"page {p.index}: line of {len(ln)} chars")
        if dur < min_dur - 1e-6:
            issues.append(f"page {p.index}: {dur:.2f}s < {min_dur}s")
        if dur > 0 and chars / dur > max_cps + 1e-6:
            issues.append(f"page {p.index}: {chars / dur:.1f} CPS > {max_cps}")
        if len({w.sentence_index for w in p.words}) > 1:
            issues.append(f"page {p.index}: crosses a sentence boundary")
    for a, b in zip(pages, pages[1:]):
        if b.start < a.end - 1e-6:
            issues.append(f"pages {a.index}/{b.index} overlap")
    return issues


# ---------------------------------------------------------------- SRT
def srt_timestamp(t: float) -> str:
    ms = max(0, int(round(t * 1000)))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def to_srt(pages: list[CaptionPage]) -> str:
    blocks = [f"{i}\n{srt_timestamp(p.start)} --> {srt_timestamp(p.end)}\n" + "\n".join(p.lines)
              for i, p in enumerate(pages, start=1)]
    return "\n\n".join(blocks) + "\n"


def write_srt(pages: list[CaptionPage], path: str | Path) -> Path:
    p = Path(path)
    p.write_text(to_srt(pages), encoding="utf-8")
    return p


# ---------------------------------------------------------------- WER
_ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven",
         "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]


def int_to_words(n: int) -> str:
    """Cardinal English words (no 'and'), up to the trillions."""
    if n < 20:
        return _ONES[n]
    if n < 100:
        return _TENS[n // 10] + ("" if n % 10 == 0 else " " + _ONES[n % 10])
    if n < 1000:
        return _ONES[n // 100] + " hundred" + ("" if n % 100 == 0 else " " + int_to_words(n % 100))
    for value, name in ((10**12, "trillion"), (10**9, "billion"), (10**6, "million"), (1000, "thousand")):
        if n >= value:
            rest = n % value
            return int_to_words(n // value) + f" {name}" + ("" if rest == 0 else " " + int_to_words(rest))
    return str(n)


_NUM_RE = re.compile(r"\d+(?:\.\d+)?")


def _num_words(tok: str, digitwise: bool) -> str:
    whole, _, frac = tok.partition(".")
    if digitwise or len(whole) > 13:
        w = " ".join(_ONES[int(d)] for d in whole)
    else:
        w = int_to_words(int(whole))
    if frac:
        w += " point " + " ".join(_ONES[int(d)] for d in frac)
    return w


def _prep(text: str) -> str:
    t = text.lower()
    t = re.sub(r"(?<=\d),(?=\d{3}\b)", "", t)
    for a, b in (("%", " percent "), ("&", " and "), ("+", " plus "), ("@", " at ")):
        t = t.replace(a, b)
    return t


def _tokens(text: str) -> list[str]:
    t = text.replace("'", "").replace("’", "")
    t = re.sub(r"[^\w]+|_", " ", t)
    toks = t.split()
    out: list[str] = []
    run: list[str] = []
    for w in toks + [""]:                       # collapse spelled letters: "k j s c e" == "kjsce"
        if len(w) == 1 and w.isalpha():
            run.append(w)
            continue
        if run:
            out.extend(["".join(run)] if len(run) > 1 else run)
            run = []
        if w:
            out.append(w)
    return out


def normalize_for_wer(text: str, digitwise: bool | list[bool] = False) -> list[str]:
    """Lowercase, numbers -> words (cardinal or digit-by-digit per number), punctuation stripped."""
    t = _prep(text)
    nums = _NUM_RE.findall(t)
    flags = digitwise if isinstance(digitwise, list) else [digitwise] * len(nums)
    it = iter(flags)
    t = _NUM_RE.sub(lambda m: " " + _num_words(m.group(0), next(it, False)) + " ", t)
    return _tokens(t)


def wer(reference: str, hypothesis: str) -> float:
    """WER of an ASR hypothesis vs the spoken script. Each number in the hypothesis is read either as a
    cardinal or digit-by-digit, whichever matches the reference better (greedy per number)."""
    import jiwer

    ref = " ".join(normalize_for_wer(reference))
    if not ref:
        return 0.0
    n = len(_NUM_RE.findall(_prep(hypothesis)))
    flags = [False] * n

    def score(fl: list[bool]) -> float:
        hyp = " ".join(normalize_for_wer(hypothesis, fl))
        return 1.0 if not hyp else float(jiwer.wer(ref, hyp))

    best = score(flags)
    for i in range(n):
        trial = flags.copy()
        trial[i] = True
        s = score(trial)
        if s < best:
            best, flags = s, trial
    return round(best, 4)
