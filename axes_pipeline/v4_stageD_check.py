"""
v4_stageD_check.py  (v4 axes, Stage D: filters, flags, suggestions, review)

Reads the two models' counterparts (candidate negatives), filters them,
flags the doubtful ones, suggests one negative per anchor, and, once you
have checked the suggestions, writes the final positive-negative pairs and
axes_v4.yaml (v4_protocol.md, version 1.8, section 6, Stage D; section 8,
decisions 4-6 and 14-16). Never edits the model outputs.

Filters (a candidate that fails one is rejected)
------------------------------------------------
  same        identical to the positive
  negation    contains not, no, never, nothing, none, without, n't, ... or an
              un- / non- / in- form of a word of the positive (D: rule 4)
  form        does not complete "a scene that looks ___" at its start (the
              Stage C form check), or changes the caption's form ("like",
              "as if", "as ... as", a single word; rule 2 of the prompt)
  edit        more than MAX_CONTENT_EDIT content words changed, or a length
              more than MAX_LEN_DIFF words away from the positive (decision 4);
              [1.9] or no content word of the positive replaced (words only
              added: the quality word is still there)
  band        [1.9: information only, SIM_BAND_FILTER = False] cosine between
              positive and candidate captions in a general sentence model
              (SIM_MODEL, not CLIP) outside SIM_BAND. Single-word antonyms
              fall below the band (unreal / real 0.46) and one-word changes
              in long captions above it, so the band measured caption
              length more than a change of topic, which the edit and form
              filters already prevent

Flags (a candidate that passes but is doubtful; you check it)
-------------------------------------------------------------
  grammar     a grammaticality classifier trained on CoLA gives the
              negative's sentence P(acceptable) < COLA_MIN and at least
              COLA_DROP below its positive's (decision 5)
  generic     the same counterpart is proposed for anchors of two or more
              axes, or for GENERIC_MIN_ANCHORS or more anchors (decision D6)
  shared word [1.9] a replacement word of the candidate (a content word it adds)
              is used mostly by another axis's candidates: e.g. "reality" in
              A04 when "real" / "reality" belong to A01's counterparts. The
              word belongs to the axis whose passing candidates use it most
  clip_close  CLIP separates the pair poorly: the CLIP cosine of the two
              sentences is in the top CLIP_TAIL of candidate pairs whose
              positive has a similar length [1.10: within LENGTH_STRATA;
              long pairs share many words and have high cosines whatever
              the change] (decision D10)
  crowded     [1.10] the candidate's crowding (mean CLIP cosine with its
              CROWD_TOP nearest candidates of other axes, minus its cosine
              with its own positive) is in the top CLIP_TAIL (decision D10).
              (1.9 flagged any candidate closer to one other-axis candidate
              than to its positive: 921 of 1,229)

Suggestion (decision D7)
------------------------
Among the candidates that pass the filters: fewest flags first, then
proposed by both models, then consistency within the axis, then the models'
own ranking [1.9]. Consistency: each replacement (a content word of the
positive replaced by a content word of the candidate, e.g. unreal -> real)
is counted over the axis's passing candidates (a candidate proposed by both
models counts twice); a candidate scores the sum of its replacements'
counts, so the axis uses the same counterpart for the same word. (1.8 used
the cosine closest to the middle of the band, which pushed away the most
natural counterparts.) The 3 ordinary phrases from the sources
closest in meaning to the caption are shown next to it (decision D8).

Your review (stageD_review.csv, one row per anchor)
---------------------------------------------------
Write in "decision":
    K1, K2, ...   that candidate (listed in "candidates"; a flagged one may
                  be chosen, a rejected one may not). [1.10] The candidate's
                  text works too, and is safer: K numbers change when flags
                  change
    O1, O2, O3    that ordinary phrase from the sources (it must pass the
                  negation and form filters)
    drop          no pair: the anchor is dropped (say why in "comment")
    other text    your own negative; it must pass the same filters (the
                  similarity band is reported, not applied)
Leave it empty to take the suggestion once ACCEPT_SUGGESTIONS = True.
Decisions are kept across runs (by anchor_id).

Negation in the positive (decision D9)
--------------------------------------
For anchors whose positive caption contains a negation word, the final
pair is dropped when CLIP separates it as poorly as the CLIP_TAIL least
separated final pairs of similar length [1.10: within LENGTH_STRATA]
("negation not read by CLIP").

Inputs
------
  STAGED_DIR: stageD_anchors.csv (v4_stageD_prepare.py),
              outputs/stageD_<model>_batchNN.json
  STAGEB_DIR: stageB_ordinary.csv, stageB_phrase_map.csv,
              stageB_clip_selection.csv

Outputs (STAGED_DIR)
--------------------
  stageD_check_report.txt   summary (start here)
  stageD_candidates.csv     every candidate: filters, scores, flags
  stageD_review.csv         one row per anchor: suggestion, candidates,
                            ordinary phrases, your decision
  stageD_pairs.csv          final pairs, with how each was decided
  stageD_axes.csv           pairs per axis and support (keeping rules)
  axes_v4.yaml              the v4 axes, with provenance (when complete)
  stageD_problems.csv       format problems in the model outputs
  stageD_cache_*.npz/.json  model scores already computed

Models (downloaded once from Hugging Face on first run)
-------------------------------------------------------
  SIM_MODEL   sentence-transformers/all-mpnet-base-v2  (pip install
              sentence-transformers)
  COLA_MODEL  a RoBERTa fine-tuned on CoLA (transformers)
  CLIP_MODEL  the CLIP of step 05 (transformers)

Usage
-----
    conda activate drl2drv
    python v4_stageD_check.py
"""

import difflib
import glob
import json
import os
import re
import time
import unicodedata
from collections import Counter

import numpy as np
import pandas as pd


# ============================================================
# SETTINGS
# ============================================================

STAGEB_DIR   = "axes/v4/stageB"
STAGED_DIR   = "axes/v4/stageD"
MODELS       = ["claude", "chatgpt"]
TEMPLATE     = "a scene that looks {}"
N_CANDIDATES = 3

MAX_CONTENT_EDIT = 2            # decision 4
MAX_LEN_DIFF     = 2
SIM_MODEL    = "sentence-transformers/all-mpnet-base-v2"   # decision 6
SIM_BAND     = (0.50, 0.92)
SIM_BAND_FILTER = False         # 1.9: the band is reported, not applied
COLA_MODEL   = "textattack/roberta-base-CoLA"              # decision 5
COLA_ACCEPT_LABEL = 1           # index of the "acceptable" class (checked at run)
COLA_MIN     = 0.5
COLA_DROP    = 0.2
CLIP_MODEL   = "openai/clip-vit-large-patch14-336"         # as in step 05
CLIP_TAIL    = 0.05             # decisions D9, D10
GENERIC_MIN_ANCHORS = 5         # decision D6
N_ORDINARY   = 3                # decision D8
ACCEPT_SUGGESTIONS = False
USE_SIM, USE_COLA, USE_CLIP = True, True, True

MIN_TEXTS, MIN_DOCS, MIN_FIRST_PERSON = 5, 2, 1           # keeping rules
SCENARIOS = {"all": {"vision", "not_specific", "disputed"},
             "vision": {"vision"}, "not_specific": {"not_specific"}}

NEGATION = {"not", "no", "never", "nothing", "none", "nobody", "nowhere", "neither",
            "nor", "without", "cannot"}
SMALL = {"a", "an", "the", "of", "as", "if", "though", "like", "is", "are", "was",
         "were", "be", "been", "being", "it", "its", "to", "in", "on", "at", "by",
         "with", "from", "through", "behind", "into", "and", "or", "that", "this",
         "there", "seen", "seem", "seems", "seemed"}
PREPOSITIONS = {"with", "of", "in", "on", "at", "to", "from", "by", "through",
                "behind", "into"}
BAD_START = ({"a", "an", "the", "it", "is", "are", "was", "were", "be", "been",
              "being", "and", "or", "that", "this", "i", "you", "he", "she", "we",
              "they"} | PREPOSITIONS) - {"like"}


# ============================================================
# Helpers
# ============================================================

def P(*n):
    return os.path.join(STAGED_DIR, *n)


def clean(t):
    t = unicodedata.normalize("NFKC", str(t))
    t = t.replace("’", "'").replace("‘", "'")
    t = re.sub(r'["“”]', "", t)
    t = re.sub(r"(?<![A-Za-z])'|'(?![A-Za-z])", "", t)
    return " ".join(t.split()).strip().rstrip(".")


def words(t):
    return re.findall(r"[a-z0-9]+(?:'[a-z]+)?", clean(t).lower())


def key(t):
    return " ".join(words(t))


def form_check(c):
    w = words(c)
    if not w:
        return False, "empty"
    if w[0] == "like":
        return True, ""
    if w[0] == "as":
        ok = len(w) > 1 and (w[1] in {"if", "though"} or (w[1] not in BAD_START
                                                          and "as" in w[2:]))
        return ok, "" if ok else "'as' not followed by 'if' / 'though' / '... as'"
    if w[0] in BAD_START:
        return False, f"starts with '{w[0]}'"
    return True, ""


def opener(c):
    w = words(c)
    if w[:1] == ["like"]:
        return "like"
    if w[:2] in (["as", "if"], ["as", "though"]):
        return "as if"
    if w[:1] == ["as"]:
        return "as ... as"
    return "word"


def negation_check(cand, pos):
    cw, pw = words(cand), set(words(pos))
    bad = [w for w in cw if w in NEGATION or w.endswith("n't")]
    for w in cw:
        for pre in ("un", "non", "in", "im", "dis"):
            if w.startswith(pre) and len(w) > len(pre) + 2 and w[len(pre):] in pw:
                bad.append(w)
    return sorted(set(bad))


def edit_size(pos, cand):
    """Content words changed (the larger of removed and added, small words
    not counted) and the difference in length, in words."""
    a, b = words(pos), words(cand)
    rem, add = [], []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(a=a, b=b,
                                                      autojunk=False).get_opcodes():
        if op != "equal":
            rem += a[i1:i2]
            add += b[j1:j2]
    rc = [w for w in rem if w not in SMALL]
    ac = [w for w in add if w not in SMALL]
    return max(len(rc), len(ac)), abs(len(a) - len(b)), rem, add


def parse_json(path):
    raw = open(path, encoding="utf-8").read().strip()
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.M).strip()
    dec, items, i = json.JSONDecoder(), [], 0
    while i < len(raw):
        while i < len(raw) and raw[i] in " \t\r\n,":
            i += 1
        if i >= len(raw):
            break
        if raw[i] not in "[{":
            j = raw.find("[", i)
            if j < 0:
                break
            i = j
        obj, i = dec.raw_decode(raw, i)
        items.extend(obj if isinstance(obj, list) else [obj])
    return items


def norm_aid(t):
    m = re.match(r"^\s*(A\d+)\s*-\s*0*(\d+)\s*$", str(t), flags=re.I)
    return f"{m.group(1).upper()}-{int(m.group(2)):02d}" if m else str(t).strip()


# --- model scores, cached on disk ---

def _cached_vectors(kind, texts, encode):
    path = P(f"stageD_cache_{kind}.npz")
    have = {}
    if os.path.exists(path):
        z = np.load(path, allow_pickle=False)
        have = dict(zip(z["keys"].tolist(), z["vecs"]))
    todo = [t for t in dict.fromkeys(texts) if t not in have]
    if todo:
        v = encode(todo)
        for t, x in zip(todo, v):
            have[t] = x
        ks = list(have)
        np.savez(path, keys=np.array(ks), vecs=np.stack([have[k] for k in ks]))
    return have


def sim_vectors(texts):
    def enc(todo):
        from sentence_transformers import SentenceTransformer
        m = SentenceTransformer(SIM_MODEL)
        return m.encode(todo, normalize_embeddings=True, batch_size=64)
    return _cached_vectors("sim", texts, enc)


def clip_vectors(texts):
    """CLIP text embeddings, computed exactly as in step 05 (encode_hf):
    padding to 77 tokens, the text tower's pooled output, text_projection,
    L2-normalised."""
    def enc(todo):
        import torch
        from transformers import CLIPModel, CLIPTokenizer
        tok = CLIPTokenizer.from_pretrained(CLIP_MODEL)
        m = CLIPModel.from_pretrained(CLIP_MODEL, use_safetensors=True)
        m.vision_model = None           # text tower only
        m.visual_projection = None
        m = m.eval()
        out = []
        with torch.inference_mode():
            for i in range(0, len(todo), 64):
                b = tok(todo[i:i + 64], padding="max_length", max_length=77,
                        truncation=True, return_tensors="pt")
                pooled = m.text_model(input_ids=b["input_ids"],
                                      attention_mask=b["attention_mask"]).pooler_output
                e = m.text_projection(pooled)
                e = e / e.norm(dim=-1, keepdim=True).clamp(min=1e-12)
                out.append(e.float().numpy())
        return np.concatenate(out)
    return _cached_vectors("clip", texts, enc)


def cola_scores(sentences):
    path = P("stageD_cache_cola.json")
    have = json.load(open(path)) if os.path.exists(path) else {}
    probe = ["a scene that looks foggy.", "scene a that foggy looks a."]
    todo = [s for s in dict.fromkeys(list(sentences) + probe) if s not in have]
    if todo:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(COLA_MODEL)
        m = AutoModelForSequenceClassification.from_pretrained(COLA_MODEL).eval()
        with torch.no_grad():
            for i in range(0, len(todo), 32):
                inp = tok(todo[i:i + 32], padding=True, truncation=True,
                          return_tensors="pt")
                p = torch.softmax(m(**inp).logits, dim=-1)[:, COLA_ACCEPT_LABEL]
                for s, x in zip(todo[i:i + 32], p.tolist()):
                    have[s] = x
        json.dump(have, open(path, "w"), indent=0)
    if have[probe[0]] <= have[probe[1]]:
        raise RuntimeError("the CoLA classifier ranks a scrambled sentence above a "
                           "grammatical one: check COLA_ACCEPT_LABEL")
    return have


LENGTH_STRATA = [(1, 1), (2, 3), (4, 6), (7, 10), (11, 999)]   # words of the positive
CROWD_TOP = 5                   # nearest other-axis candidates averaged


def stratum(c):
    n = len(words(c))
    for lo_, hi_ in LENGTH_STRATA:
        if lo_ <= n <= hi_:
            return f"{lo_}-{hi_}"
    return "other"


def tail_by_stratum(values, strata, tail, min_n=20):
    """The (1 - tail) quantile of values within each length stratum (the
    whole set when a stratum has fewer than min_n values): CLIP cosines of
    pairs that share many words are high whatever the change, so pairs are
    compared with pairs of similar length."""
    v = pd.Series(values, dtype=float)
    st = pd.Series(strata)
    glob_ = float(np.nanquantile(v, 1 - tail)) if v.notna().any() else np.nan
    cut = {}
    for k_ in st.unique():
        x = v[st == k_].dropna()
        cut[k_] = float(np.quantile(x, 1 - tail)) if len(x) >= min_n else glob_
    return st.map(cut).to_numpy()


def S(c):
    return TEMPLATE.format(c) + "."


# ============================================================
# Main
# ============================================================

def main():
    t0 = time.time()
    print("=" * 70)
    print("v4_stageD_check.py")
    print("=" * 70)
    problems = []

    def problem(kind, model="", detail=""):
        problems.append({"kind": kind, "model": model, "detail": detail})

    an = pd.read_csv(P("stageD_anchors.csv"), dtype=str).fillna("")
    aids = list(an.anchor_id)
    A = an.set_index("anchor_id")
    ordinary = pd.read_csv(os.path.join(STAGEB_DIR, "stageB_ordinary.csv"),
                           dtype=str).fillna("")
    ordinary["clean"] = ordinary.quote.map(clean)
    ordinary = ordinary[ordinary.clean != ""].drop_duplicates("clean")
    pm = pd.read_csv(os.path.join(STAGEB_DIR, "stageB_phrase_map.csv"), dtype=str).fillna("")
    pm = pm[pm.included.str.lower() == "true"]
    sel = pd.read_csv(os.path.join(STAGEB_DIR, "stageB_clip_selection.csv"),
                      dtype=str).fillna("").set_index("axis_id")
    old = {}
    if os.path.exists(P("stageD_review.csv")):
        o = pd.read_csv(P("stageD_review.csv"), dtype=str).fillna("")
        old = {r.anchor_id: (r.decision.strip(), r.comment) for r in o.itertuples()}

    # --- model outputs ---
    cand = []
    for m in MODELS:
        files = sorted(glob.glob(P("outputs", f"stageD_{m}_batch*.json")))
        if not files:
            raise SystemExit(f"No Stage D files for {m} in {P('outputs')}")
        seen = Counter()
        for f in files:
            try:
                items = parse_json(f)
            except json.JSONDecodeError as e:
                problem("unreadable_json", m, f"{os.path.basename(f)}: {e}")
                continue
            for x in items:
                if not isinstance(x, dict):
                    problem("not_an_object", m, f"{os.path.basename(f)}: {x!r}"[:200])
                    continue
                a = norm_aid(x.get("id", ""))
                if a not in A.index:
                    problem("unknown_anchor", m, f"{os.path.basename(f)}: {x.get('id')!r}")
                    continue
                seen[a] += 1
                if seen[a] > 1:
                    problem("answered_more_than_once", m, f"{a} (first kept)")
                    continue
                cs = x.get("counterparts") or x.get("candidates") or []
                cs = [clean(c) for c in cs if isinstance(c, str) and clean(c)]
                if len(cs) != N_CANDIDATES:
                    problem("not_3_counterparts", m, f"{a}: {len(cs)}")
                for r, c in enumerate(cs[:N_CANDIDATES], 1):
                    cand.append({"anchor_id": a, "model": m, "rank": r, "text": c})
        for a in aids:
            if not seen[a]:
                problem("missing_anchor", m, a)
    C = pd.DataFrame(cand)

    # merge identical candidates within an anchor
    C["key"] = C.text.map(key)
    rows = []
    for (a, k), g in C.groupby(["anchor_id", "key"], sort=False):
        ms = sorted(set(g.model), key=MODELS.index)
        rows.append({"anchor_id": a, "axis_id": A.at[a, "axis_id"],
                     "positive": A.at[a, "caption"], "candidate": g.text.iloc[0],
                     "key": k, "models": "+".join(ms), "both": len(ms) == len(MODELS),
                     "best_rank": int(g["rank"].min()),
                     "proposed_as": "; ".join(f"{x.model}#{x.rank}" for x in g.itertuples())})
    K = pd.DataFrame(rows)

    # --- your decisions that are new texts (scored like candidates) ---
    own = {}
    for a, (dec, _) in old.items():
        if a in A.index and dec and not re.fullmatch(r"(?i)[KO]\d+|drop", dec):
            own[a] = clean(dec)

    def hard(a, c):
        pos = A.at[a, "caption"]
        why = []
        if key(c) == key(pos):
            why.append("same as positive")
        neg = negation_check(c, pos)
        if neg:
            why.append("negation (" + " ".join(neg) + ")")
        ok, note = form_check(c)
        if not ok:
            why.append(f"form ({note})")
        elif opener(c) != opener(pos):
            why.append(f"form changed ({opener(pos)} -> {opener(c)})")
        ce, ld, rem, _ = edit_size(pos, c)
        if not [w for w in rem if w not in SMALL]:
            why.append("no word of the positive replaced")
        if ce > MAX_CONTENT_EDIT:
            why.append(f"edit ({ce} content words)")
        if ld > MAX_LEN_DIFF:
            why.append(f"length (±{ld} words)")
        return why, ce, ld

    hk = [hard(r.anchor_id, r.candidate) for r in K.itertuples()]
    K["reject"] = ["; ".join(h[0]) for h in hk]
    K["content_changed"] = [h[1] for h in hk]
    K["length_diff"] = [h[2] for h in hk]

    # --- similarity band (general sentence model) ---
    notes = {}
    sims = {}
    if USE_SIM:
        try:
            texts = (list(K.candidate) + list(A.caption) + list(own.values())
                     + list(ordinary.clean))
            V = sim_vectors(texts)
            sims = V
        except Exception as e:  # missing package or model
            notes["sim"] = f"skipped ({type(e).__name__}: {e})"
            print(f"  WARNING: similarity band {notes['sim']}")

    def sim(a_text, b_text):
        if not sims:
            return np.nan
        return float(np.dot(sims[a_text], sims[b_text]))

    K["sim"] = [sim(r.positive, r.candidate) for r in K.itertuples()]
    lo, hi = SIM_BAND
    out_band = K.sim.notna() & ((K.sim < lo) | (K.sim > hi))
    K["outside_band"] = out_band
    if SIM_BAND_FILTER:
        K.loc[out_band, "reject"] = K.loc[out_band].apply(
            lambda r: (r.reject + "; " if r.reject else "") +
            f"band (cosine {r.sim:.2f})", axis=1)
    K["passes"] = K.reject == ""
    # replacements: content words of the positive replaced by content words
    # of the candidate
    reps = []
    for r in K.itertuples():
        _, _, rem, add = edit_size(r.positive, r.candidate)
        rc = [w for w in rem if w not in SMALL]
        ac = [w for w in add if w not in SMALL]
        reps.append((rc, ac))
    K["removed_words"] = [" ".join(x[0]) for x in reps]
    K["added_words"] = [" ".join(x[1]) for x in reps]

    # --- flags ---
    K["flag_list"] = ""

    def add_flag(mask, label):
        K.loc[mask, "flag_list"] = K.loc[mask, "flag_list"].map(
            lambda f: (f + "; " if f else "") + label)

    # generic (D6), counted over the candidates that pass
    pk = K[K.passes]
    axes_per = pk.groupby("key").axis_id.nunique()
    anch_per = pk.groupby("key").anchor_id.nunique()
    gen = K.passes & K.key.map(lambda k: axes_per.get(k, 0) >= 2 or
                               anch_per.get(k, 0) >= GENERIC_MIN_ANCHORS)
    add_flag(gen, "generic")

    # shared replacement word (3a): a word belongs to the axis whose passing
    # candidates add it most often; candidates of other axes adding it are
    # flagged
    pk = K[K.passes]
    use = Counter((w, r.axis_id) for r in pk.itertuples() for w in set(r.added_words.split()))
    owner = {}
    for (w, ax_), n in sorted(use.items(), key=lambda kv: (-kv[1], kv[0][1])):
        owner.setdefault(w, ax_)
    shared_label = []
    for r in K.itertuples():
        ws = [w for w in dict.fromkeys(r.added_words.split())
              if r.passes and owner.get(w, r.axis_id) != r.axis_id]
        shared_label.append(", ".join(f"{w} ({owner[w]})" for w in ws))
    K["shared_words"] = shared_label
    for ix in K.index[K.shared_words != ""]:
        K.at[ix, "flag_list"] = (K.at[ix, "flag_list"] + "; " if K.at[ix, "flag_list"]
                                 else "") + f"shared word: {K.at[ix, 'shared_words']}"

    # grammar (decision 5)
    cola = {}
    if USE_COLA:
        try:
            sents = [S(x) for x in list(K.candidate) + list(A.caption) + list(own.values())]
            cola = cola_scores(sents)
        except Exception as e:
            notes["cola"] = f"skipped ({type(e).__name__}: {e})"
            print(f"  WARNING: grammar flag {notes['cola']}")
    K["cola_pos"] = [cola.get(S(r.positive), np.nan) for r in K.itertuples()]
    K["cola"] = [cola.get(S(r.candidate), np.nan) for r in K.itertuples()]
    gram = K.passes & (K.cola < COLA_MIN) & (K.cola < K.cola_pos - COLA_DROP)
    add_flag(gram, "grammar")

    # CLIP (D10)
    clipv = {}
    if USE_CLIP:
        try:
            sents = [S(x) for x in list(K.candidate) + list(A.caption) + list(own.values())
                     + list(ordinary.clean)]
            clipv = clip_vectors(sents)
        except Exception as e:
            notes["clip"] = f"skipped ({type(e).__name__}: {e})"
            print(f"  WARNING: CLIP flags {notes['clip']}")

    def ccos(a_text, b_text):
        if not clipv:
            return np.nan
        return float(np.dot(clipv[S(a_text)], clipv[S(b_text)]))

    K["clip_pair"] = [ccos(r.positive, r.candidate) for r in K.itertuples()]
    K["stratum"] = K.positive.map(stratum)
    K["clip_cut"] = np.nan
    clip_cut = np.nan
    if clipv and K.passes.any():
        pk = K[K.passes]
        K.loc[pk.index, "clip_cut"] = tail_by_stratum(pk.clip_pair, pk.stratum, CLIP_TAIL)
        clip_cut = float(np.nanquantile(pk.clip_pair, 1 - CLIP_TAIL))
    add_flag(K.passes & (K.clip_pair > K.clip_cut), "clip_close")
    # crowded (1.10): how much closer the candidate is to its nearest
    # other-axis candidates (mean of CROWD_TOP) than to its own positive;
    # the top CLIP_TAIL are flagged
    K["crowding"] = np.nan
    if clipv and K.passes.any():
        pk = K[K.passes]
        M = np.stack([clipv[S(x)] for x in pk.candidate])
        ax = pk.axis_id.to_numpy()
        G = M @ M.T
        near = []
        for i in range(len(pk)):
            o = np.sort(G[i][ax != ax[i]])[::-1][:CROWD_TOP]
            near.append(float(o.mean()) if len(o) else np.nan)
        K.loc[pk.index, "clip_other_axis"] = near
        K.loc[pk.index, "crowding"] = K.loc[pk.index, "clip_other_axis"] - \
            K.loc[pk.index, "clip_pair"]
        crowd_cut = float(np.nanquantile(K.loc[pk.index, "crowding"], 1 - CLIP_TAIL))
        add_flag(K.passes & (K.crowding > crowd_cut), "crowded")
    else:
        K["clip_other_axis"] = np.nan

    # --- suggestion (D7) ---
    K["n_flags"] = K.flag_list.map(lambda f: len([x for x in f.split("; ") if x]))
    # consistency within the axis (2a)
    cnt = Counter()
    for r in K[K.passes].itertuples():
        wgt = 2 if r.both else 1
        for a_ in r.removed_words.split():
            for b_ in r.added_words.split():
                cnt[(r.axis_id, a_, b_)] += wgt
    K["consistency"] = [sum(cnt[(r.axis_id, a_, b_)] for a_ in r.removed_words.split()
                            for b_ in r.added_words.split()) if r.passes else 0
                        for r in K.itertuples()]
    K["order"] = 0
    sugg = {}
    for a, g in K[K.passes].groupby("anchor_id"):
        g = g.sort_values(["n_flags", "both", "consistency", "best_rank"],
                          ascending=[True, False, False, True])
        sugg[a] = g.index[0]
    # candidate numbers per anchor: passing first (in suggestion order), then rejected
    K["K"] = ""
    for a, g in K.groupby("anchor_id"):
        g = g.sort_values(["passes", "n_flags", "both", "consistency", "best_rank"],
                          ascending=[False, True, False, False, True])
        for i, ix in enumerate(g.index, 1):
            K.at[ix, "K"] = f"K{i}"
    K.drop(columns=["order"]).to_csv(P("stageD_candidates.csv"), index=False,
                                     float_format="%.3f")

    # --- ordinary phrases closest to each caption (D8) ---
    ordm = {}
    if sims and len(ordinary):
        OV = np.stack([sims[x] for x in ordinary.clean])
        for a in aids:
            s = OV @ sims[A.at[a, "caption"]]
            top = np.argsort(-s)[:N_ORDINARY]
            ordm[a] = [(ordinary.clean.iloc[i], ordinary.doc_id.iloc[i], float(s[i]))
                       for i in top]

    # --- review and final pairs ---
    rv, pairs = [], []
    for a in aids:
        g = K[K.anchor_id == a].sort_values("K", key=lambda s: s.str[1:].astype(int))
        sg = sugg.get(a)

        def desc(r):
            tag = []
            if r.both:
                tag.append("both models")
            else:
                tag.append(r.proposed_as)
            sc = (f"sim {r.sim:.2f}" + (" outside band" if r.outside_band else "")
                  if pd.notna(r.sim) else "")
            cc = f"clip {r.clip_pair:.3f}" if pd.notna(r.clip_pair) else ""
            info = ", ".join(x for x in tag + [sc, cc] if x)
            fl = f" [{r.flag_list}]" if r.flag_list else ""
            return f"{r.K}: {r.candidate} ({info}){fl}"
        passing = " | ".join(desc(r) for r in g[g.passes].itertuples())
        rejected = " | ".join(f"{r.candidate} ({r.reject})" for r in g[~g.passes].itertuples())
        ords = " | ".join(f"O{i}: {t} [{d}, sim {s:.2f}]"
                          for i, (t, d, s) in enumerate(ordm.get(a, []), 1))
        dec, com = old.get(a, ("", ""))
        from_sug = False
        if not dec and sg is not None and ACCEPT_SUGGESTIONS:
            dec, from_sug = K.at[sg, "K"], True
        neg, how, note, src = "", "open", "", ""
        if dec:
            d = dec.strip()
            same = g[(g.key == key(d)) & g.passes]
            if not re.fullmatch(r"(?i)K\d+|O\d+|drop", d) and len(same):
                # your decision is the text of a candidate: that candidate
                d = same.K.iloc[0]
            if re.fullmatch(r"(?i)K\d+", d):
                hit = g[g.K == d.upper()]
                if not len(hit):
                    note = f"{d} is not a candidate of this anchor"
                elif not hit.passes.iloc[0]:
                    note = f"{d} was rejected ({hit.reject.iloc[0]})"
                else:
                    r = hit.iloc[0]
                    neg, src = r.candidate, r.models
                    how = "suggestion" if from_sug else "your choice"
            elif re.fullmatch(r"(?i)O\d+", d):
                i = int(d[1:]) - 1
                if i >= len(ordm.get(a, [])):
                    note = f"{d} is not listed"
                else:
                    t = ordm[a][i][0]
                    bad = negation_check(t, A.at[a, "caption"])
                    ok, fn = form_check(t)
                    if bad or not ok:
                        note = f"{d} fails: " + " ".join(bad) + (f" form ({fn})" if not ok else "")
                    else:
                        neg, src, how = t, f"sources ({ordm[a][i][1]})", "your choice"
            elif d.lower() == "drop":
                how = "dropped by you"
            else:
                why, _, _ = hard(a, d)
                if why:
                    note = "your negative fails: " + "; ".join(why)
                else:
                    neg, src, how = clean(d), "yours", "your negative"
        rv.append({"anchor_id": a, "axis_id": A.at[a, "axis_id"],
                   "positive": A.at[a, "caption"],
                   "negation_in_positive": A.at[a, "negation_in_positive"],
                   "suggestion": K.at[sg, "K"] if sg is not None else "",
                   "suggested_negative": K.at[sg, "candidate"] if sg is not None else "",
                   "suggestion_flags": K.at[sg, "flag_list"] if sg is not None else "",
                   "candidates": passing, "rejected": rejected, "ordinary": ords,
                   "decision": "" if from_sug else dec, "comment": com, "problem": note})
        pairs.append({"anchor_id": a, "axis_id": A.at[a, "axis_id"],
                      "positive": A.at[a, "caption"], "negative": neg, "how": how,
                      "negative_source": src,
                      "negation_in_positive": A.at[a, "negation_in_positive"],
                      "text_ids": A.at[a, "text_ids"], "phrase_ids": A.at[a, "phrase_ids"],
                      "documents": A.at[a, "documents"],
                      "text_classes": A.at[a, "text_classes"]})
    RV = pd.DataFrame(rv)
    RV.to_csv(P("stageD_review.csv"), index=False)
    PR = pd.DataFrame(pairs)

    # final pair scores; D9 on the final pairs
    have = PR.negative != ""
    if clipv:
        missing = [S(x) for x in PR.loc[have, "negative"] if S(x) not in clipv]
        if missing:
            clipv.update(clip_vectors(missing))
    if sims:
        miss = [x for x in PR.loc[have, "negative"] if x not in sims]
        if miss:
            sims.update(sim_vectors(miss))
    PR["sim"] = [sim(p, n) if n else np.nan for p, n in zip(PR.positive, PR.negative)]
    PR["clip_pair"] = [ccos(p, n) if n else np.nan for p, n in zip(PR.positive, PR.negative)]
    n_open = int((PR.how == "open").sum())
    PR["stratum"] = PR.positive.map(stratum)
    PR["clip_cut"] = np.nan
    final_cut = np.nan
    if clipv and have.any():
        PR.loc[have, "clip_cut"] = tail_by_stratum(PR.loc[have, "clip_pair"],
                                                   PR.loc[have, "stratum"], CLIP_TAIL)
        final_cut = float(np.nanquantile(PR.loc[have, "clip_pair"], 1 - CLIP_TAIL))
    d9 = have & (PR.negation_in_positive != "") & (PR.clip_pair > PR.clip_cut)
    PR["d9_dropped"] = d9
    PR.loc[d9, "how"] = PR.loc[d9, "how"] + "; dropped (negation not read by CLIP)"
    PR["in_axes"] = have & ~d9
    PR.to_csv(P("stageD_pairs.csv"), index=False, float_format="%.3f")

    # --- per axis: pairs and support (keeping rules) ---
    axrows = []
    for ax_ in dict.fromkeys(an.axis_id):
        g = PR[PR.axis_id == ax_]
        k = g[g.in_axes]
        tids = {t for x in k.text_ids for t in x.split(";") if t}
        sub = pm[pm.text_id.isin(tids)]
        row = {"axis_id": ax_, "name": sel.at[ax_, "name_stageC"], "anchors": len(g),
               "pairs": int(len(k)), "open": int((g.how == "open").sum()),
               "dropped_by_you": int((g.how == "dropped by you").sum()),
               "dropped_d9": int(g.d9_dropped.sum()),
               "from_sources": int(k.negative_source.str.startswith("sources").sum()),
               "both_models": int((k.negative_source == "+".join(MODELS)).sum()),
               "median_clip_pair": float(k.clip_pair.median()) if len(k) else np.nan}
        for sc, classes in SCENARIOS.items():
            s = sub[sub.sense_class.isin(classes)]
            nt, nd = s.text_id.nunique(), s.doc_id.nunique()
            nf = int((s.first_person == "yes").sum())
            row.update({f"{sc}_texts": nt, f"{sc}_docs": nd, f"{sc}_first_person": nf,
                        f"{sc}_kept": nt >= MIN_TEXTS and nd >= MIN_DOCS and
                        nf >= MIN_FIRST_PERSON})
        axrows.append(row)
    AX = pd.DataFrame(axrows)
    AX.to_csv(P("stageD_axes.csv"), index=False, float_format="%.3f")
    pr = pd.DataFrame(problems, columns=["kind", "model", "detail"])
    pr.to_csv(P("stageD_problems.csv"), index=False)

    # --- axes_v4.yaml (when complete) ---
    complete = n_open == 0 and not (RV.problem != "").any()
    if complete:
        import yaml
        doc = {"version": "v4", "protocol": "1.10",
               "created": time.strftime("%Y-%m-%d %H:%M:%S"),
               "template": TEMPLATE,
               "templates_robustness": ["a photo that looks {}", "a place that looks {}",
                                        "the world looks {}"],
               "axes": []}
        for ax_ in dict.fromkeys(an.axis_id):
            k = PR[(PR.axis_id == ax_) & PR.in_axes]
            doc["axes"].append({
                "id": ax_, "name": sel.at[ax_, "name_stageC"],
                "definition": sel.at[ax_, "definition_stageC"],
                "pairs": [{"anchor_id": r.anchor_id, "positive": r.positive,
                           "negative": r.negative, "negative_source": r.negative_source,
                           "sense_classes": r.text_classes.split(";"),
                           "text_ids": r.text_ids.split(";"),
                           "phrase_ids": r.phrase_ids.split(";"),
                           "documents": r.documents.split(";")}
                          for r in k.itertuples()]})
        yaml.safe_dump(doc, open(P("axes_v4.yaml"), "w"), sort_keys=False,
                       allow_unicode=True, width=100)

    # ========================================================
    # Report
    # ========================================================
    L = []
    W = L.append
    W("STAGE D: COUNTERFACTUALS, FILTERS AND REVIEW (v4_protocol.md 1.10, section 6)")
    W("=" * 78)
    W(f"Date: {time.strftime('%Y-%m-%d %H:%M')}")
    W(f"Anchors: {len(aids)} in {an.axis_id.nunique()} axes; candidates: {len(C)} "
      f"({len(K)} distinct within anchors)")
    W("")
    W("1. OUTPUT CHECKS (stageD_problems.csv)")
    W("-" * 78)
    if len(pr):
        for (k_, m), n in pr.groupby(["kind", "model"]).size().items():
            W(f"  {m:<9}{k_:<32}{n:>5}")
    else:
        W("  no problems: every anchor answered once, with 3 counterparts")
    W("")
    W("2. FILTERS (a candidate can fail several)")
    W("-" * 78)
    reasons = Counter(x.split(" (")[0] for r in K.reject for x in r.split("; ") if x)
    W(f"  pass: {int(K.passes.sum())} of {len(K)}")
    for k_, v in reasons.most_common():
        W(f"    {v:>5}  {k_}")
    for k_, v in notes.items():
        W(f"  {k_}: {v}")
    if sims:
        W(f"  similarity band {lo:.2f}-{hi:.2f} ({SIM_MODEL.split('/')[-1]}): "
          + ("applied" if SIM_BAND_FILTER else "information only")
          + f"; outside it: {int(K.outside_band.sum())}; median {K.sim.median():.2f}")
    W("")
    W("3. AGREEMENT AND FLAGS (candidates that pass)")
    W("-" * 78)
    pk = K[K.passes]
    W(f"  proposed by both models: {int(pk.both.sum())} candidates; anchors with at "
      f"least one: {pk[pk.both].anchor_id.nunique()} of {len(aids)}")
    fl = Counter(x for f in pk.flag_list for x in f.split("; ") if x)
    fl["shared word"] = int((pk.shared_words != "").sum())
    for k_ in ("grammar", "generic", "shared word", "clip_close", "crowded"):
        W(f"  {k_:<12}{fl.get(k_, 0):>5}")
    own_top = Counter(w for r in pk.itertuples() for w in set(r.shared_words.split(", "))
                      if w)
    if own_top:
        W("  most shared words: " + "; ".join(f"{k_} {v}" for k_, v in own_top.most_common(8)))
    if clipv:
        W(f"  CLIP pair cosine: median {pk.clip_pair.median():.3f}; clip_close: top "
          f"{CLIP_TAIL:.0%} within length strata (overall cut {clip_cut:.3f}); crowded: "
          f"top {CLIP_TAIL:.0%} of crowding")
    gtop = pk[pk.flag_list.str.contains("generic")].groupby("candidate").anchor_id.nunique() \
        .sort_values(ascending=False).head(8)
    if len(gtop):
        W("  most shared counterparts: " + "; ".join(f"'{k_}' ({v})" for k_, v in gtop.items()))
    W(f"  anchors without any passing candidate: {len(set(aids) - set(sugg))} "
      "(write your own negative or drop)")
    W("")
    W("4. REVIEW (stageD_review.csv)")
    W("-" * 78)
    W(f"  open: {n_open}; problems in decisions: {int((RV.problem != '').sum())}")
    W("  " + "; ".join(f"{k_} {v}" for k_, v in PR.how.value_counts().items()))
    if not ACCEPT_SUGGESTIONS:
        W("  Check the suggestions, write a decision where you disagree, then set")
        W("  ACCEPT_SUGGESTIONS = True and run again.")
    W("")
    W(f"5. NEGATION IN THE POSITIVE (D9): {int((PR.negation_in_positive != '').sum())} anchors")
    W("-" * 78)
    for r in PR[PR.negation_in_positive != ""].itertuples():
        st = ("dropped" if r.d9_dropped else "kept") if r.negative else "open"
        cp = f"{r.clip_pair:.3f}" if pd.notna(r.clip_pair) else "-"
        cc = f"{r.clip_cut:.3f}" if pd.notna(r.clip_cut) else "-"
        W(f"  {r.anchor_id:<7}{st:<8}clip {cp:<6} cut {cc:<6}{r.positive[:34]:<35}-> "
          f"{r.negative[:26]}")
    if not clipv:
        W("  not applied: CLIP scores missing")
    elif not have.any():
        W("  applied once the pairs are decided")
    else:
        W(f"  cut-off: the least separated {CLIP_TAIL:.0%} of final pairs of similar length "
          f"(words of the positive: {', '.join(f'{a}-{b}' for a, b in LENGTH_STRATA)})")
    W("")
    W("6. AXES (pairs and support; keeping rules as in Stage B)")
    W("-" * 78)
    W(f"    {'axis':<5}{'name':<30}{'anch.':>6}{'pairs':>6}{'open':>5}{'D9':>4}"
      f"{'src':>4}{'docs':>5}{'1st':>5}   kept all/vis/n-s")
    for r in AX.itertuples():
        f3 = "/".join("Y" if getattr(r, f"{sc}_kept") else "-" for sc in SCENARIOS)
        W(f"    {r.axis_id:<5}{str(r.name)[:29]:<30}{r.anchors:>6}{r.pairs:>6}{r.open:>5}"
          f"{r.dropped_d9:>4}{r.from_sources:>4}{r.all_docs:>5}{r.all_first_person:>5}   {f3}")
    W("")
    W("STAGE D COMPLETE: axes_v4.yaml written." if complete else
      "Not complete: decide the open rows of stageD_review.csv and run again.")
    W(f"Runtime {time.time() - t0:.1f} s")
    open(P("stageD_check_report.txt"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    json.dump({"script": "v4_stageD_check.py", "protocol": "1.10",
               "created": time.strftime("%Y-%m-%d %H:%M:%S"), "open": n_open,
               "sim_band": SIM_BAND, "sim_band_filter": SIM_BAND_FILTER, "clip_tail": CLIP_TAIL, "clip_cut_candidates": clip_cut,
               "clip_cut_final": final_cut, "cola": [COLA_MIN, COLA_DROP],
               "max_content_edit": MAX_CONTENT_EDIT, "max_len_diff": MAX_LEN_DIFF,
               "notes": notes},
              open(P("check_meta.json"), "w"), indent=2, default=str)


if __name__ == "__main__":
    main()
