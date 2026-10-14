"""
v4_stageV_verify.py  (v4 axes, Stage V: verification and merging)

Checks every quote extracted in Stage A against its PDF, merges the two
models' extractions into one pool of verified phrases, and reports how
well the two models agree (v4_protocol.md, version 1.2, section 5).
Never edits the Stage A files.

What it does
------------
1. Reads the 9 PDFs (PyMuPDF) and reports pages with no text layer.
2. Reads the Stage A files stageA_<model>_<DOC_ID>_pass<n>.json (one JSON
   array each; several arrays pasted one after the other, or a ```json
   fence, are also accepted). Checks the fields, the allowed label values,
   the id format and duplicate ids. Problems are listed, not fixed.
3. Verifies every quote (section 5.2 of the protocol):
     comparison key = Unicode-normalised, lower case, letters and digits
     only, so spacing, hyphens, line breaks and punctuation never cause a
     mismatch. The context sentence is located first; the quote is then
     looked for inside it, else anywhere in the document; the occurrence
     nearest the stated page is used.
       VERIFIED             on the stated pdf_page or within PAGE_WINDOW
       VERIFIED_WRONG_PAGE  elsewhere (page corrected)
       NEAR_MATCH           not exact, but a span with similarity >=
                            NEAR_MATCH_MIN within NEAR_PAGE_WINDOW pages of
                            the stated page: listed for you to confirm
       NOT_FOUND            dropped (listed)
     Flags (information only): the strict comparison (punctuation kept)
     fails; the match starts or ends inside a word; the context is not
     found; the quote is not inside its context.
4. Near matches: stageV_near_matches.csv has a "keep" column. Write yes
   or no and run the script again: your answers are kept across runs;
   confirmed items enter the pool with the PDF's own text as the quote.
5. Review (stageV_review.csv; section 5.4 of the protocol): one row per
   decision you need to make, with a "decision" column (and "comment").
   Write your decisions and run the script again: they are kept across
   runs (by review_id) and applied.
     not_found      a quote not found in the PDF, shown with the nearest
                    PDF passage: confirm (the words are there, broken up
                    by the PDF) or drop
     first_person   a phrase whose speaker labels disagree on whether it
                    is first-person (participant, patient, historical
                    case): yes or no
     invalid_label  a label outside the allowed values: one of the allowed
                    values
     overlap        two models' spans that overlap without one containing
                    the other: merge or separate
6. Merges (section 5.3): within a model, pass-2 items that repeat a pass-1
   item at the same place are removed; across models, two items match when
   one quote's place in the PDF lies inside the other's (the same words
   at the same place, one span containing the other). Matches are joined
   transitively into phrases, which keep the longest span and record every
   item, model and pass.
7. Labels: each model's speaker, target, sense, state, quoted_from and
   note are kept per phrase; Cohen's kappa per label on the phrases both
   models found with one label each.
8. Reports per document: items, verification, merging, found by both /
   one model, second-pass gains, page coverage and the longest runs of
   pages without items (for the single-pass book, the check that no part
   was skipped).

Outputs (OUTPUT_DIR)
--------------------
  stageA_verified.csv       the phrase pool (one row per phrase), with
                            first_person (yes / no / undecided) and where
                            it came from (agreed / review)
  stageV_items.csv          every Stage A item with its verification
  stageV_near_matches.csv   near matches to confirm (keep = yes / no)
  stageV_review.csv         decisions to make (decision column; see 5.)
  stageV_problems.csv       format problems and flags
  stageA_run_log.csv        the run log rebuilt from the files, with empty
                            columns to fill in (model_name, date, ...)
  stageV_report.txt         summary (start here)
  run_meta.json

Usage
-----
    conda activate drl2drv
    pip install pymupdf          # if not installed
    python v4_stageV_verify.py

Run from drl2drv_updated/ (or set the paths below). A few seconds per
document; the book takes longest.
"""

import bisect
import difflib
import glob
import json
import os
import re
import time
import unicodedata
from collections import Counter, defaultdict

import pandas as pd

try:
    import pymupdf
except ImportError:
    try:
        import fitz as pymupdf
    except ImportError:
        raise SystemExit("PyMuPDF is needed: pip install pymupdf")


# ============================================================
# SETTINGS
# ============================================================

PDF_DIR     = "dpd_selected_scientific_literature"
STAGEA_DIR  = "stageA_v4" # "axes/v4/stageA"          # the 34 JSON files
OUTPUT_DIR  = "axes/v4/stageV"

# Document id (as in the Stage A file names and item ids) -> the start of
# its PDF file name. Each prefix must match exactly one PDF in PDF_DIR.
DOCUMENTS = {
    "CIAUNICA2022": "Ciaunica et al (2022)",
    "DILKES2024":   "Dilkes (2024)",
    "FINO2024":     "Fino et al (2024)",
    "MIYAZONO2021": "Miyazono (2021)",
    "PIENKOS2022":  "Pienkos et al (2022)",
    "RADOVIC2002":  "Radovic et al (2002)",
    "SIERRA2001":   "Sierra et al (2001)",
    "SIMEON2023":   "Simeon et al (2023)",
    "TANAKA2018":   "Tanaka (2018)",
}
SINGLE_PASS = ["SIMEON2023"]            # extracted without Prompt A2
MODELS      = ["claude", "chatgpt"]

PAGE_WINDOW       = 1      # pages either side counted as the stated page
NEAR_PAGE_WINDOW  = 2      # where near matches are looked for
NEAR_MATCH_MIN    = 0.95   # similarity for a near match
NEAR_MIN_KEY      = 6      # quotes shorter than this (letters + digits)
                           # are never near-matched (too ambiguous)
MIN_PAGE_CHARS    = 40     # a page with fewer letters / digits has no
                           # usable text layer
GAP_REPORT_MIN    = 5      # report runs of at least this many pages
                           # without items
REVIEW_MARGIN     = 150    # characters of PDF text shown either side of
                           # the nearest passage of a quote not found
FIRST_PERSON = {"participant_or_patient", "historical_case"}

ALLOWED = {
    "speaker": {"participant_or_patient", "historical_case",
                "scale_or_questionnaire_item", "diagnostic_criteria",
                "author_description", "other"},
    "target": {"surroundings", "objects", "other_people",
               "perceptual_situation", "mixed", "unclear"},
    "sense": {"vision", "hearing", "touch", "other_sense",
              "not_sense_specific"},
    "state": {"altered", "ordinary", "unclear"},
}
LABELS = ["speaker", "target", "sense", "state", "quoted_from", "note"]
REQUIRED = ["id", "quote", "context", "pdf_page", "printed_page", "speaker",
            "quoted_from", "target", "sense", "state", "note"]


# ============================================================
# Text normalisation
# ============================================================

def keyed(text):
    """Comparison key: NFKC, case-folded, letters and digits only. Returns
    (key, positions) where positions[i] is the index in `text` of the
    character that produced key[i]."""
    out, pos = [], []
    for i, ch in enumerate(text):
        for c in unicodedata.normalize("NFKC", ch).casefold():
            if c.isalnum():
                out.append(c)
                pos.append(i)
    return "".join(out), pos


def key(text):
    return keyed(text or "")[0]


_QUOTES = str.maketrans({"‘": "'", "’": "'", "‚": "'",
                         "‛": "'", "′": "'", "“": '"',
                         "”": '"', "„": '"', "″": '"',
                         "‐": "-", "‑": "-", "‒": "-",
                         "–": "-", "—": "-", "―": "-",
                         "−": "-", "­": ""})


def strict(text):
    """Punctuation kept: NFKC, case-folded, straight quotes and dashes,
    line-break hyphens rejoined, all white space removed."""
    t = unicodedata.normalize("NFKC", text or "").casefold().translate(_QUOTES)
    t = re.sub(r"-\s*\n\s*", "", t)
    return re.sub(r"\s+", "", t)


def wordkey(text):
    """Letters and digits with single spaces between words (line-break
    hyphens rejoined first)."""
    t = unicodedata.normalize("NFKC", text or "").casefold().translate(_QUOTES)
    t = re.sub(r"-\s*\n\s*", "", t)
    return " " + " ".join(re.findall(r"[^\W_]+", t)) + " "


# ============================================================
# PDFs
# ============================================================

class Doc:
    def __init__(self, doc_id, path):
        self.id, self.path = doc_id, path
        with pymupdf.open(path) as pdf:
            self.pages = [p.get_text("text") for p in pdf]
        self.n = len(self.pages)
        keys, self.start = [], []
        self.pos = []                       # (page, index in page text)
        off = 0
        for pi, t in enumerate(self.pages):
            k, p = keyed(t)
            keys.append(k)
            self.start.append(off)
            self.pos.extend((pi + 1, j) for j in p)
            off += len(k)
        self.key = "".join(keys)
        self.page_len = [len(k) for k in keys]
        self.no_text = [i + 1 for i, L in enumerate(self.page_len)
                        if L < MIN_PAGE_CHARS]
        self.strict_pages = [strict(t) for t in self.pages]
        self.word_pages = [wordkey(t) for t in self.pages]

    def page_of(self, k):
        """1-based page of key offset k."""
        return bisect.bisect_right(self.start, k)

    def span(self, p0, p1):
        """Key offsets covering pages p0..p1 (1-based, clipped)."""
        p0, p1 = max(1, p0), min(self.n, p1)
        a = self.start[p0 - 1]
        b = self.start[p1] if p1 < self.n else len(self.key)
        return a, b

    def text_of(self, a, b):
        """Original PDF text for key offsets [a, b)."""
        if a >= b:
            return ""
        (pa, ia), (pb, ib) = self.pos[a], self.pos[b - 1]
        if pa == pb:
            return self.pages[pa - 1][ia:ib + 1]
        parts = [self.pages[pa - 1][ia:]]
        parts += [self.pages[p - 1] for p in range(pa + 1, pb)]
        parts.append(self.pages[pb - 1][:ib + 1])
        return "".join(parts)

    def window_text(self, p0, p1, fn):
        p0, p1 = max(1, p0), min(self.n, p1)
        return "".join(fn[p - 1] for p in range(p0, p1 + 1))


def find_all(hay, needle, lo=0, hi=None):
    hi = len(hay) if hi is None else hi
    out, i = [], hay.find(needle, lo, hi)
    while i != -1 and i + len(needle) <= hi:
        out.append(i)
        i = hay.find(needle, i + 1, hi)
    return out


def nearest(doc, offsets, length, page):
    """Occurrence nearest the stated page: (start, page, distance)."""
    best = None
    for o in offsets:
        p = doc.page_of(o)
        d = abs(p - page) if page else 0
        if best is None or d < best[2]:
            best = (o, p, d)
    return best


def fuzzy(q, hay, base):
    """Best approximate occurrence of q in hay. Returns (ratio, start,
    end) in key offsets (base added), or (0, -1, -1)."""
    if not q or not hay:
        return 0.0, -1, -1
    sm = difflib.SequenceMatcher(None, q, hay, autojunk=False)
    need = max(3, len(q) // 5)
    starts = set()
    for blk in sm.get_matching_blocks():
        if blk.size >= need:
            starts.add(blk.b - blk.a)
    best = (0.0, -1, -1)
    L = len(q)
    for s0 in starts:
        for ds in (-2, -1, 0, 1, 2):
            for dl in (-2, -1, 0, 1, 2):
                a, b = s0 + ds, s0 + ds + L + dl
                if a < 0 or b > len(hay) or b <= a:
                    continue
                r = difflib.SequenceMatcher(None, q, hay[a:b],
                                            autojunk=False).ratio()
                if r > best[0]:
                    best = (r, base + a, base + b)
    return best


# ============================================================
# Stage A files
# ============================================================

FILE_RE = re.compile(r"^stageA_([a-z]+)_([A-Z0-9]+)_pass(\d)\.json$")


def parse_json(path):
    raw = open(path, encoding="utf-8").read().strip()
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.M).strip()
    if not raw:
        return []
    dec, items, i = json.JSONDecoder(), [], 0
    while i < len(raw):
        while i < len(raw) and raw[i] in " \t\r\n,":
            i += 1
        if i >= len(raw):
            break
        obj, i = dec.raw_decode(raw, i)
        items.extend(obj if isinstance(obj, list) else [obj])
    return items


def kappa(a, b):
    """Cohen's kappa for two lists of labels."""
    n = len(a)
    if n == 0:
        return float("nan"), 0, float("nan")
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb.get(k, 0) for k in ca) / (n * n)
    k = (po - pe) / (1 - pe) if pe < 1 else float("nan")
    return k, n, po


# ============================================================
# Main
# ============================================================

def P(*n):
    return os.path.join(OUTPUT_DIR, *n)


def main():
    t0 = time.time()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=" * 70)
    print("v4_stageV_verify.py")
    print("=" * 70)
    problems = []

    def problem(kind, doc="", model="", item="", detail=""):
        problems.append({"kind": kind, "doc_id": doc, "model": model,
                         "item_id": item, "detail": detail})

    # --- PDFs ---
    pdfs = sorted(os.listdir(PDF_DIR))
    docs = {}
    for d, prefix in DOCUMENTS.items():
        m = [f for f in pdfs if f.startswith(prefix) and f.lower().endswith(".pdf")]
        if len(m) != 1:
            raise SystemExit(f"{d}: {len(m)} PDFs start with '{prefix}' in "
                             f"{PDF_DIR} (need exactly one): {m}")
        docs[d] = Doc(d, os.path.join(PDF_DIR, m[0]))
        print(f"  {d:<13} {docs[d].n:>4} pages  {m[0][:60]}")
        if docs[d].no_text:
            problem("pdf_pages_without_text", d,
                    detail=f"{len(docs[d].no_text)} pages: {docs[d].no_text[:30]}")

    # --- Stage A files ---
    files = sorted(os.path.basename(f) for f in glob.glob(os.path.join(STAGEA_DIR, "*")))
    expected, ignored = {}, []
    for f in files:
        m = FILE_RE.match(f)
        if not m or m.group(1) not in MODELS or m.group(2) not in DOCUMENTS:
            ignored.append(f)
            continue
        expected[(m.group(1), m.group(2), int(m.group(3)))] = f
    missing = []
    for d in DOCUMENTS:
        for mo in MODELS:
            for ps in ([1] if d in SINGLE_PASS else [1, 2]):
                if (mo, d, ps) not in expected:
                    missing.append(f"stageA_{mo}_{d}_pass{ps}.json")
    extra2 = [k for k in expected if k[1] in SINGLE_PASS and k[2] == 2]
    if missing:
        raise SystemExit("Missing Stage A files:\n  " + "\n  ".join(missing))
    print(f"Stage A files: {len(expected)} used, {len(ignored)} ignored")
    for f in ignored:
        problem("file_ignored", detail=f)
    for k in extra2:
        problem("unexpected_second_pass", k[1], k[0], detail=expected[k])

    # --- items ---
    items, invalid = [], []
    for (mo, d, ps), f in sorted(expected.items()):
        try:
            arr = parse_json(os.path.join(STAGEA_DIR, f))
        except Exception as e:
            raise SystemExit(f"{f}: cannot parse JSON ({e})")
        for j, it in enumerate(arr):
            if not isinstance(it, dict):
                problem("not_an_object", d, mo, detail=f"{f} #{j}")
                continue
            miss = [k for k in REQUIRED if k not in it]
            if miss:
                problem("missing_fields", d, mo, it.get("id", f"#{j}"),
                        ", ".join(miss))
            iid = str(it.get("id", f"{d}-?{j}"))
            if not re.match(rf"^{d}-\d{{3,}}$", iid):
                problem("bad_id", d, mo, iid)
            for lab, ok in ALLOWED.items():
                v = it.get(lab)
                if v not in ok:
                    problem("invalid_label", d, mo, iid, f"{lab} = {v!r}")
                    invalid.append((mo, d, iid, lab, v))
            try:
                page = int(it.get("pdf_page"))
            except (TypeError, ValueError):
                page = 0
                problem("bad_pdf_page", d, mo, iid, repr(it.get("pdf_page")))
            items.append({"model": mo, "doc_id": d, "pass": ps, "file": f,
                          "item_id": iid, "quote": str(it.get("quote") or ""),
                          "context": str(it.get("context") or ""),
                          "pdf_page": page,
                          "printed_page": it.get("printed_page"),
                          **{lab: it.get(lab) for lab in LABELS}})
    ids = Counter((x["model"], x["doc_id"], x["item_id"]) for x in items)
    for (mo, d, iid), c in ids.items():
        if c > 1:
            problem("duplicate_id", d, mo, iid, f"{c} times")
    print(f"Items: {len(items):,}")

    # --- review decisions (stageV_review.csv) ---
    review_old, review_rows = {}, []
    if os.path.exists(P("stageV_review.csv")):
        old = pd.read_csv(P("stageV_review.csv"), dtype=str).fillna("")
        for r in old.itertuples():
            review_old[r.review_id] = (r.decision.strip().lower(), r.comment)
        n_dec = sum(1 for v in review_old.values() if v[0])
        print(f"Review decisions read: {n_dec}")

    def review(rid, kind, options, **fields):
        """Register a review row; return the valid decision or None."""
        dec, com = review_old.get(rid, ("", ""))
        valid = dec in options
        review_rows.append({"review_id": rid, "kind": kind,
                            "options": " / ".join(options),
                            "decision": dec, "comment": com,
                            "decision_valid": valid or not dec, **fields})
        return dec if valid else None

    by_key = {(x["model"], x["doc_id"], x["item_id"]): x for x in items}
    for mo, d, iid, lab, v in invalid:
        it = by_key.get((mo, d, iid))
        dec = review(f"IL|{mo}|{iid}|{lab}", "invalid_label",
                     sorted(ALLOWED[lab]), doc_id=d, page=it["pdf_page"] if it else "",
                     item_ids=f"{mo}:{iid}", quote=it["quote"] if it else "",
                     detail=f"{lab} = {v!r}", context=it["context"] if it else "")
        if dec and it is not None:
            it[lab] = dec
            it["label_fixed"] = f"{lab}: {v} -> {dec}"

    # --- earlier near-match decisions ---
    decisions = {}
    if os.path.exists(P("stageV_near_matches.csv")):
        old = pd.read_csv(P("stageV_near_matches.csv"), dtype=str).fillna("")
        for r in old.itertuples():
            v = r.keep.strip().lower()
            if v in ("yes", "no", "y", "n"):
                decisions[(r.model, r.item_id, r.quote)] = v[0] == "y"
        print(f"Near-match decisions read: {len(decisions)}")

    # --- verification ---
    print("Verifying quotes...")
    near_rows = []
    for it in items:
        doc = docs[it["doc_id"]]
        qk, ck = key(it["quote"]), key(it["context"])
        page = it["pdf_page"]
        it.update({"status": "NOT_FOUND", "verified_page": None, "start": -1,
                   "end": -1, "similarity": None, "pdf_text": "",
                   "context_status": "not_found", "quote_in_context": False,
                   "strict_ok": None, "inside_word": None})
        if not qk:
            problem("empty_quote", it["doc_id"], it["model"], it["item_id"])
            continue
        it["quote_in_context"] = bool(ck) and qk in ck
        # 1) context, then the quote inside it
        ctx = None
        if ck:
            occ = find_all(doc.key, ck)
            if occ:
                ctx = nearest(doc, occ, len(ck), page)
                it["context_status"] = "found"
        hit = None
        if ctx is not None:
            inside = find_all(doc.key, qk, ctx[0], ctx[0] + len(ck))
            if inside:
                hit = (inside[0], doc.page_of(inside[0]))
        # 2) anywhere in the document
        if hit is None:
            occ = find_all(doc.key, qk)
            if occ:
                o, p, _ = nearest(doc, occ, len(qk), page)
                hit = (o, p)
        if hit is not None:
            o, p = hit
            it["start"], it["end"], it["verified_page"] = o, o + len(qk), p
            it["status"] = ("VERIFIED" if not page or abs(p - page) <= PAGE_WINDOW
                            else "VERIFIED_WRONG_PAGE")
            it["similarity"] = 1.0
            it["pdf_text"] = doc.text_of(o, o + len(qk))
        elif len(qk) >= NEAR_MIN_KEY:
            p0 = (page or 1) - NEAR_PAGE_WINDOW
            p1 = (page or doc.n) + NEAR_PAGE_WINDOW
            a, b = doc.span(p0, p1)
            r, s0, e0 = fuzzy(qk, doc.key[a:b], a)
            if r >= NEAR_MATCH_MIN:
                it.update({"similarity": round(r, 4), "start": s0, "end": e0,
                           "verified_page": doc.page_of(s0),
                           "pdf_text": doc.text_of(s0, e0)})
                dk = (it["model"], it["item_id"], it["quote"])
                if dk in decisions:
                    it["status"] = ("NEAR_MATCH_CONFIRMED" if decisions[dk]
                                    else "NEAR_MATCH_REJECTED")
                else:
                    it["status"] = "NEAR_MATCH"
                near_rows.append({"doc_id": it["doc_id"], "model": it["model"],
                                  "item_id": it["item_id"],
                                  "pdf_page": page,
                                  "found_page": it["verified_page"],
                                  "similarity": round(r, 4),
                                  "quote": it["quote"],
                                  "pdf_text": " ".join(it["pdf_text"].split()),
                                  "context": it["context"],
                                  "keep": {True: "yes", False: "no"}.get(
                                      decisions.get(dk), "")})
        # context near match (information)
        if it["context_status"] == "not_found" and ck and len(ck) >= NEAR_MIN_KEY:
            p = it["verified_page"] or page or 1
            a, b = doc.span(p - 1, p + 1)
            r, _, _ = fuzzy(ck, doc.key[a:b], a)
            if r >= NEAR_MATCH_MIN:
                it["context_status"] = "near_match"
        # flags
        if it["start"] >= 0:
            p = it["verified_page"]
            sw = doc.window_text(p - 1, p + 1, doc.strict_pages)
            it["strict_ok"] = strict(it["quote"]) in sw
            ww = doc.window_text(p - 1, p + 1, doc.word_pages)
            it["inside_word"] = wordkey(it["quote"]) not in ww
        if it["status"] == "NOT_FOUND":
            problem("quote_not_found", it["doc_id"], it["model"], it["item_id"],
                    it["quote"][:80])
        if it["context_status"] == "not_found" and it["start"] >= 0:
            problem("context_not_found", it["doc_id"], it["model"],
                    it["item_id"], it["context"][:80])
        if not it["quote_in_context"]:
            problem("quote_not_in_context", it["doc_id"], it["model"],
                    it["item_id"], it["quote"][:80])
        if it["inside_word"]:
            problem("match_inside_word_or_spacing", it["doc_id"], it["model"],
                    it["item_id"], it["quote"][:80])

    # --- quotes not found: the nearest PDF passage, for review ---
    for it in items:
        if it["status"] != "NOT_FOUND":
            continue
        doc = docs[it["doc_id"]]
        qk = key(it["quote"])
        if not qk:
            continue
        page = it["pdf_page"] or 1
        a, b = doc.span(page - NEAR_PAGE_WINDOW, page + NEAR_PAGE_WINDOW)
        r, s0, e0 = fuzzy(qk, doc.key[a:b], a)
        if r < 0.8:                                   # then the whole document
            for p0 in range(1, doc.n + 1, 5):
                a2, b2 = doc.span(p0, p0 + 5)
                r2, s2, e2 = fuzzy(qk, doc.key[a2:b2], a2)
                if r2 > r:
                    r, s0, e0 = r2, s2, e2
        passage, found_page = "", ""
        if s0 >= 0:
            m0, m1 = max(0, s0 - REVIEW_MARGIN), min(len(doc.key), e0 + REVIEW_MARGIN)
            passage = " ".join(doc.text_of(m0, m1).split())
            found_page = doc.page_of(s0)
        dec = review(f"NF|{it['model']}|{it['item_id']}", "not_found",
                     ["confirm", "drop"], doc_id=it["doc_id"], page=page,
                     item_ids=f"{it['model']}:{it['item_id']}", quote=it["quote"],
                     detail=(f"nearest passage on page {found_page}, similarity "
                             f"{r:.2f}" if s0 >= 0 else "no similar passage"),
                     pdf_passage=passage, context=it["context"])
        if dec == "confirm" and s0 >= 0:
            it.update({"status": "CONFIRMED_BY_REVIEW", "start": s0, "end": e0,
                       "verified_page": found_page, "similarity": round(r, 4),
                       "pdf_text": doc.text_of(s0, e0)})
        elif dec == "drop":
            it["status"] = "DROPPED_BY_REVIEW"

    ok_status = {"VERIFIED", "VERIFIED_WRONG_PAGE", "NEAR_MATCH_CONFIRMED",
                 "CONFIRMED_BY_REVIEW"}

    # --- within-model duplicates (same model, same place) ---
    seen = {}
    for it in sorted(items, key=lambda x: (x["pass"], x["item_id"])):
        it["duplicate_of"] = ""
        if it["status"] not in ok_status:
            continue
        k = (it["model"], it["doc_id"], it["start"], it["end"])
        if k in seen:
            it["duplicate_of"] = seen[k]["item_id"]
        else:
            seen[k] = it
    pool = [it for it in items if it["status"] in ok_status and not it["duplicate_of"]]

    # --- merging across models (one span inside the other) ---
    parent = list(range(len(pool)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    by_doc = defaultdict(list)
    for i, it in enumerate(pool):
        by_doc[it["doc_id"]].append(i)
    partial = 0
    for d, idx in by_doc.items():
        idx.sort(key=lambda i: pool[i]["start"])
        for a_ in range(len(idx)):
            i = idx[a_]
            for b_ in range(a_ + 1, len(idx)):
                j = idx[b_]
                if pool[j]["start"] >= pool[i]["end"]:
                    break
                if pool[i]["model"] == pool[j]["model"]:
                    continue
                si, ei = pool[i]["start"], pool[i]["end"]
                sj, ej = pool[j]["start"], pool[j]["end"]
                if (si <= sj and ej <= ei) or (sj <= si and ei <= ej):
                    parent[root(j)] = root(i)
                else:
                    partial += 1
                    x, y = sorted((pool[i], pool[j]),
                                  key=lambda z: MODELS.index(z["model"]))
                    dec = review(f"OV|{x['model']}:{x['item_id']}|{y['model']}:{y['item_id']}",
                                 "overlap", ["merge", "separate"], doc_id=d,
                                 page=x["verified_page"],
                                 item_ids=f"{x['model']}:{x['item_id']};{y['model']}:{y['item_id']}",
                                 quote=f"{x['quote']}  ||  {y['quote']}",
                                 detail="the two spans overlap; neither contains the other",
                                 context=x["context"])
                    if dec == "merge":
                        parent[root(j)] = root(i)
    groups = defaultdict(list)
    for i in range(len(pool)):
        groups[root(i)].append(pool[i])

    phrases = []
    for g in groups.values():
        rep_ = max(g, key=lambda x: (x["end"] - x["start"], -x["pass"],
                                     x["model"] == MODELS[0]))
        models = sorted({x["model"] for x in g}, key=MODELS.index)
        row = {"doc_id": rep_["doc_id"], "start": min(x["start"] for x in g),
               "quote": (" ".join(rep_["pdf_text"].split())
                         if rep_["status"] == "NEAR_MATCH_CONFIRMED"
                         else rep_["quote"]),
               "context": rep_["context"], "pdf_page": rep_["verified_page"],
               "printed_page": rep_["printed_page"],
               "found_by": "both" if len(models) == len(MODELS) else models[0],
               "n_items": len(g),
               "item_ids": ";".join(f"{x['model']}:{x['item_id']}" for x in g),
               "pass": ";".join(f"{m}:" + ",".join(sorted({str(x['pass']) for x in g
                                                          if x['model'] == m}))
                                for m in models),
               "verification": ("near_match_confirmed"
                                if any(x["status"] == "NEAR_MATCH_CONFIRMED" for x in g)
                                else "confirmed_by_review"
                                if all(x["status"] == "CONFIRMED_BY_REVIEW" for x in g)
                                else "verified"),
               "label_fixed": ";".join(x.get("label_fixed", "") for x in g
                                       if x.get("label_fixed")),
               "page_corrected": any(x["status"] == "VERIFIED_WRONG_PAGE" for x in g),
               "flags": ";".join(sorted(
                   {"punct_or_spacing_differs" for x in g if x["strict_ok"] is False} |
                   {"inside_word" for x in g if x["inside_word"]} |
                   {"context_not_found" for x in g if x["context_status"] == "not_found"} |
                   {"quote_not_in_context" for x in g if not x["quote_in_context"]}))}
        for m in MODELS:
            mine = [x for x in g if x["model"] == m]
            row[f"pass2_only_{m}"] = bool(mine) and all(x["pass"] == 2 for x in mine)
            for lab in LABELS:
                vals = sorted({str(x[lab]) for x in mine if x[lab] is not None})
                row[f"{lab}_{m}"] = "|".join(vals)
        # first person: agreed by every label, else your decision
        fp = {x["speaker"] in FIRST_PERSON for x in g if x["speaker"] is not None}
        if len(fp) == 1:
            row["first_person"] = "yes" if fp.pop() else "no"
            row["first_person_source"] = "agreed"
        else:
            ids_ = ";".join(sorted(f"{x['model']}:{x['item_id']}" for x in g))
            dec = review(f"FP|{ids_}", "first_person", ["yes", "no"],
                         doc_id=rep_["doc_id"], page=rep_["verified_page"],
                         item_ids=ids_, quote=row["quote"],
                         detail="speaker: " + "; ".join(
                             f"{m} {row[f'speaker_{m}'] or '-'}" for m in MODELS),
                         context=rep_["context"])
            row["first_person"] = dec or "undecided"
            row["first_person_source"] = "review" if dec else "undecided"
        phrases.append(row)
    ph = pd.DataFrame(phrases).sort_values(["doc_id", "start"]).reset_index(drop=True)
    ph.insert(0, "phrase_id", [f"{d}-P{i + 1:04d}" for d, i in
                               zip(ph["doc_id"], ph.groupby("doc_id").cumcount())])
    ph = ph.drop(columns="start")
    ph.to_csv(P("stageA_verified.csv"), index=False)

    # --- outputs: items, near matches, problems ---
    it_df = pd.DataFrame(items)
    it_df.drop(columns=["start", "end"]).to_csv(P("stageV_items.csv"), index=False)
    nm = pd.DataFrame(near_rows, columns=["doc_id", "model", "item_id", "pdf_page",
                                          "found_page", "similarity", "quote",
                                          "pdf_text", "context", "keep"])
    nm.to_csv(P("stageV_near_matches.csv"), index=False)
    pr = pd.DataFrame(problems, columns=["kind", "doc_id", "model", "item_id",
                                         "detail"])
    pr.to_csv(P("stageV_problems.csv"), index=False)
    rv_cols = ["review_id", "kind", "doc_id", "page", "item_ids", "quote",
               "detail", "pdf_passage", "context", "options", "decision",
               "comment"]
    kind_order = {"not_found": 0, "first_person": 1, "invalid_label": 2,
                  "overlap": 3}
    rv = pd.DataFrame(review_rows)
    for c in rv_cols + ["decision_valid"]:
        if c not in rv:
            rv[c] = ""
    rv = rv.drop_duplicates("review_id")
    rv["_o"] = rv["kind"].map(kind_order)
    rv = rv.sort_values(["_o", "doc_id", "review_id"]).drop(columns="_o")
    rv[rv_cols].to_csv(P("stageV_review.csv"), index=False)
    obsolete = [k for k, v in review_old.items() if v[0] and k not in set(rv.review_id)]

    # --- coverage and run log ---
    def gaps(pages_with, n, skip):
        runs, run = [], []
        for p in range(1, n + 1):
            if p in pages_with or p in skip:
                if len(run) >= GAP_REPORT_MIN:
                    runs.append((run[0], run[-1]))
                run = []
            else:
                run.append(p)
        if len(run) >= GAP_REPORT_MIN:
            runs.append((run[0], run[-1]))
        return sorted(runs, key=lambda r: r[0] - r[1])

    log_rows, cover = [], {}
    for d in DOCUMENTS:
        for m in MODELS:
            mine = it_df[(it_df.doc_id == d) & (it_df.model == m)]
            okp = mine[mine.status.isin(ok_status)]
            pages = set(int(x) for x in okp.verified_page.dropna())
            cover[(d, m)] = pages
            g_ = gaps(pages, docs[d].n, set(docs[d].no_text))
            log_rows.append({
                "doc_id": d, "model_app": m, "model_name": "", "date": "",
                "chat_type": "", "pdf_readable": "", "continues_needed": "",
                "problems": "",
                "n_items_pass1": int((mine["pass"] == 1).sum()),
                "n_items_pass2": (int((mine["pass"] == 2).sum())
                                  if d not in SINGLE_PASS else "single pass"),
                "n_verified": int(len(okp)),
                "pages_with_items": len(pages), "pdf_pages": docs[d].n,
                "first_page": min(pages) if pages else "",
                "last_page": max(pages) if pages else "",
                "longest_gaps": "; ".join(f"{a}-{b}" for a, b in g_[:5])})
    pd.DataFrame(log_rows).to_csv(P("stageA_run_log.csv"), index=False)

    # --- agreement ---
    both = ph[ph.found_by == "both"]
    kap = {}
    for lab in ["speaker", "target", "sense", "state"]:
        a_, b_ = [], []
        for r in both.itertuples():
            x, y = getattr(r, f"{lab}_{MODELS[0]}"), getattr(r, f"{lab}_{MODELS[1]}")
            if x and y and "|" not in x and "|" not in y:
                a_.append(x)
                b_.append(y)
        kap[lab] = kappa(a_, b_)

    # ========================================================
    # Report
    # ========================================================
    L = []
    W = L.append
    W("STAGE V: VERIFICATION AND MERGING (v4_protocol.md 1.2, section 5)")
    W("=" * 78)
    W(f"Date: {time.strftime('%Y-%m-%d %H:%M')}")
    W(f"Stage A files: {len(expected)} ({len(ignored)} other files ignored"
      + (f": {', '.join(ignored[:5])}" if ignored else "") + ")")
    W(f"Items: {len(items):,}; phrases in the verified pool: {len(ph):,}")
    W("")
    W("1. VERIFICATION OF THE QUOTES (all items)")
    W("-" * 78)
    st = it_df.groupby(["model", "status"]).size().unstack(fill_value=0)
    W(st.to_string())
    W(f"  repeats removed (same model, same place, e.g. pass 2 repeating pass 1): "
      f"{int((it_df.duplicate_of != '').sum())}")
    W(f"  near matches to confirm: {int((it_df.status == 'NEAR_MATCH').sum())} "
      f"(stageV_near_matches.csv, column keep)")
    W(f"  contexts not found: {int(((it_df.context_status == 'not_found') & (it_df.start >= 0)).sum())}; "
      f"quotes not inside their context: {int((~it_df.quote_in_context).sum())}; "
      f"punctuation / spacing differs: {int((it_df.strict_ok == False).sum())}; "  # noqa: E712
      f"match inside a word: {int((it_df.inside_word == True).sum())}")  # noqa: E712
    W("")
    W("2. PER DOCUMENT")
    W("-" * 78)
    W(f"  {'document':<13}{'pages':>6}  {'items c/g':>11}  {'verified c/g':>13}"
      f"  {'not found c/g':>14}  {'phrases':>8}{'both':>7}{'c only':>8}"
      f"{'g only':>8}  {'pass-2 only c/g':>16}")
    for d in DOCUMENTS:
        m0, m1 = MODELS
        x = it_df[it_df.doc_id == d]
        n_i = [int((x.model == m).sum()) for m in MODELS]
        n_v = [int(((x.model == m) & x.status.isin(ok_status)).sum()) for m in MODELS]
        n_nf = [int(((x.model == m) & (x.status == "NOT_FOUND")).sum()) for m in MODELS]
        y = ph[ph.doc_id == d]
        p2 = ("single pass" if d in SINGLE_PASS else
              f"{int(y[f'pass2_only_{m0}'].sum())}/{int(y[f'pass2_only_{m1}'].sum())}")
        W(f"  {d:<13}{docs[d].n:>6}  {n_i[0]:>5}/{n_i[1]:<5}  {n_v[0]:>6}/{n_v[1]:<6}"
          f"  {n_nf[0]:>7}/{n_nf[1]:<6}  {len(y):>8}{int((y.found_by == 'both').sum()):>7}"
          f"{int((y.found_by == m0).sum()):>8}{int((y.found_by == m1).sum()):>8}"
          f"  {p2:>16}")
    tot = len(ph)
    W(f"  total: {tot:,} phrases; found by both {int((ph.found_by == 'both').sum()):,} "
      f"({100 * (ph.found_by == 'both').mean():.0f}%), {MODELS[0]} only "
      f"{int((ph.found_by == MODELS[0]).sum()):,}, {MODELS[1]} only "
      f"{int((ph.found_by == MODELS[1]).sum()):,}")
    W(f"  (c = {MODELS[0]}, g = {MODELS[1]}; pass-2 only = phrases a model found only "
      "in its second pass)")
    W(f"  spans of the two models that overlap without one containing the other "
      f"(not merged): {partial}")
    W("")
    W("3. PAGE COVERAGE (pages with at least one verified item; longest runs of "
      f">= {GAP_REPORT_MIN} pages without items)")
    W("-" * 78)
    for r in log_rows:
        W(f"  {r['doc_id']:<13}{r['model_app']:<9}{r['pages_with_items']:>4} of "
          f"{r['pdf_pages']:<4} pages, {r['first_page']}-{r['last_page']}   gaps: "
          f"{r['longest_gaps'] or 'none'}")
    for d in SINGLE_PASS:
        a_, b_ = cover[(d, MODELS[0])], cover[(d, MODELS[1])]
        W(f"  {d}: pages with items from {MODELS[0]} only {len(a_ - b_)}, from "
          f"{MODELS[1]} only {len(b_ - a_)}, from both {len(a_ & b_)}")
    for d in DOCUMENTS:
        if docs[d].no_text:
            W(f"  {d}: {len(docs[d].no_text)} pages without a text layer: "
              f"{docs[d].no_text[:20]}")
    W("")
    W("4. LABEL AGREEMENT (phrases found by both models, one label each)")
    W("-" * 78)
    for lab, (k, n, po) in kap.items():
        W(f"  {lab:<8} kappa {k:.2f}  (agreement {po:.0%}, n = {n})")
    W("  Labels per model are in stageA_verified.csv (<label>_<model>).")
    sp = both[(both[f"speaker_{MODELS[0]}"] != both[f"speaker_{MODELS[1]}"])]
    first = {"participant_or_patient", "historical_case"}
    fp = sp[sp.apply(lambda r: (bool(set(r[f"speaker_{MODELS[0]}"].split("|")) & first)
                                != bool(set(r[f"speaker_{MODELS[1]}"].split("|")) & first)),
                     axis=1)] if len(sp) else sp
    W(f"  speaker disagreements that change whether the phrase is first-person: "
      f"{len(fp)} (to review before Stage B)")
    W("")
    W("5. REVIEW (stageV_review.csv: write decision, run again)")
    W("-" * 78)
    if len(rv):
        for kd in kind_order:
            x = rv[rv.kind == kd]
            if not len(x):
                continue
            dec_ = x[(x.decision != "") & x.decision_valid.astype(bool)]
            bad = x[(x.decision != "") & ~x.decision_valid.astype(bool)]
            W(f"  {kd:<15}{len(x):>5} rows; decided {len(dec_):>4}"
              f" ({', '.join(f'{k} {v}' for k, v in dec_.decision.value_counts().items()) or '-'})"
              + (f"; INVALID decisions {len(bad)} (options: {x.options.iloc[0]})"
                 if len(bad) else ""))
    else:
        W("  nothing to review")
    if obsolete:
        W(f"  {len(obsolete)} earlier decisions no longer apply (their rows changed, "
          "e.g. after merging): decide those rows again")
    open_ = int(((rv.decision == "") | ~rv.decision_valid.astype(bool)).sum()) if len(rv) else 0
    W("  first_person in the pool: "
      + ", ".join(f"{k} {v}" for k, v in ph.first_person.value_counts().items()))
    W("  STAGE V COMPLETE: every review row decided." if open_ == 0 else
      f"  {open_} review rows still open.")
    W("")
    W("6. PROBLEMS (stageV_problems.csv)")
    W("-" * 78)
    for k, c in Counter(x["kind"] for x in problems).most_common():
        W(f"  {k:<32}{c:>6}")
    W("")
    W("Next: decide the open rows of stageV_review.csv (and any near matches in")
    W("stageV_near_matches.csv), run again until Stage V is complete; then Stage B.")
    W(f"Runtime {time.time() - t0:.0f} s")
    open(P("stageV_report.txt"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))

    json.dump({"script": "v4_stageV_verify.py", "protocol": "1.2",
               "created": time.strftime("%Y-%m-%d %H:%M:%S"),
               "pdfs": {d: os.path.basename(docs[d].path) for d in docs},
               "files_used": sorted(expected.values()), "files_ignored": ignored,
               "settings": {"page_window": PAGE_WINDOW,
                            "near_page_window": NEAR_PAGE_WINDOW,
                            "near_match_min": NEAR_MATCH_MIN,
                            "near_min_key": NEAR_MIN_KEY,
                            "single_pass": SINGLE_PASS},
               "n_items": len(items), "n_phrases": int(len(ph)),
               "review": {"rows": int(len(rv)), "open": open_},
               "kappa": {k: v[0] for k, v in kap.items()}},
              open(P("run_meta.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
