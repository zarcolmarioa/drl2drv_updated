"""
v4_stageC_check.py  (v4 axes, Stage C: checks, comparison, your review)

Reads the two models' captions, checks every caption against the rules of
Stage C (v4_protocol.md, version 1.6, section 6, Stage C and section 8,
decisions 3, 11 and 12), compares the two models, and builds the final caption table
once you have decided the texts that need you. Never edits the model
outputs.

Checks on every caption
-----------------------
  words      every word must be (a) a word of the quote, in any grammatical
             form of the same root ("see" / "seen", "dream" / "dreamy"),
             (b) a word standing within ADJACENT_WORDS of the quote in one of
             its contexts ("restored from context"), (c) an allowed added
             word (decision 3), or (d) "seen" in "as if seen" / "like seen"
             directly before a preposition. Anything else fails.
  form       "a scene that looks ___" must be grammatical at its start: the
             caption begins with "like", "as if", "as though", or a word that
             is not an article, preposition, pronoun or form of "be" (an
             adjective or participle).
  deletions  quote words missing from the caption. First-person frames,
             hedges, intensifiers and function words are only logged; any
             other deleted word is flagged "content word deleted", because
             a word carrying the quality may not be deleted (you check).
  fluency    SLOR of "a scene that looks {caption}.": the GPT-2 log
             probability minus the log probability of its words on their own
             (word frequencies, package wordfreq), per word. Unlike
             perplexity, it does not penalise short captions or rare words.
             Flagged below Q1 - SLOR_IQR x IQR of all captions (low outliers).
  meta words a caption containing a word of META_WORDS (feeling, experience,
             complaint, self, life, ...) is flagged, even when both models
             wrote it: it may not describe how a scene looks (drop reason
             not_scene, protocol 6.2 applied to single phrases).

Comparison and decisions
------------------------
  A text is ACCEPTED automatically when both models wrote the same caption
  and it passes every check without a flag, or when both dropped it. Every
  other text goes to stageC_review.csv with both candidates and their check
  results. Write in "decision":
      claude | chatgpt   take that model's caption
      not_scene          no caption: the phrase describes a feeling, a
                         judgement, a report about the experience, or a
                         person, life or event, not how a scene looks
      drop               no caption for another reason (say why in comment)
      anything else      your own caption; it must pass the word and form
                         checks, like the models' captions
  and run again. Decisions are kept across runs (by text_id).

  To change a caption that was accepted automatically, add a row to
  stageC_review.csv with its text_id and your decision (e.g. not_scene);
  the decision overrides the acceptance.

  Suggestions (protocol 6, Stage C, conventions). When the two captions
  differ only in the following ways, the "suggestion" column names the
  caption that follows the conventions, and which ones:
      1  a clause takes "as if", not "like"
      2  a barrier, fog or similar noun takes "as if seen through / in /
         with", not "like" ("as if seen through a veil", not "like a veil")
      3  first-person words removed ("the eyes", not "my eyes")
      4  hedges and intensifiers removed ("just", "so", "very", "sort of")
  Read them, write a decision wherever you disagree, then set
  ACCEPT_SUGGESTIONS = True and run again: every row with an empty decision
  and a suggestion takes the suggestion (logged as "review: suggestion").

When every text is decided, the final captions are written, support is
counted again from the phrases behind the captioned texts, and the keeping
rules are applied by sense class (as in Stage B).

Inputs
------
  STAGEC_DIR: stageC_texts.csv (from v4_stageC_prepare.py),
              outputs/stageC_<model>_batchNN.json
  STAGEB_DIR: stageB_phrase_map.csv, stageB_clip_selection.csv

Outputs (STAGEC_DIR)
--------------------
  stageC_check_report.txt   summary (start here)
  stageC_candidates.csv     every text: both models' captions and checks
  stageC_review.csv         texts that need you
  stageC_captions.csv       final: quote -> caption, with every edit, how it
                            was decided, phrase ids, sense class
  stageC_axes.csv           support after Stage C and the keeping rules
  stageC_problems.csv       format problems in the model outputs
  stageC_lm_cache.json      GPT-2 log probabilities already computed

Usage
-----
    conda activate drl2drv
    python v4_stageC_check.py
"""

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

STAGEB_DIR     = "axes/v4/stageB"
STAGEC_DIR     = "axes/v4/stageC"
MODELS         = ["claude", "chatgpt"]
TEMPLATE       = "a scene that looks {}."
ADJACENT_WORDS = 3        # context words on each side that may be restored
USE_FLUENCY    = False    # dropped 6 Oct 2026 (protocol 1.7): every flag was a
                          # false alarm; True only to reproduce the 1.6 runs
FLUENCY_MODEL  = "gpt2"
SLOR_IQR       = 1.5      # flag SLOR below Q1 - SLOR_IQR x IQR
ACCEPT_SUGGESTIONS = False  # True: empty decisions take the suggestion
MIN_TEXTS        = 5      # keeping rules (protocol 6, Stage B)
MIN_DOCS         = 2
MIN_FIRST_PERSON = 1
SCENARIOS = {"all": {"vision", "not_specific", "disputed"},
             "vision": {"vision"}, "not_specific": {"not_specific"}}

# decision 3 (protocol 8)
ALLOWED_ADD = {"a", "an", "the", "like", "as", "if", "though", "with", "of", "in",
               "on", "at", "to", "from", "by", "through", "behind", "into", "is",
               "are", "was", "were", "be", "been", "being", "seem", "seems",
               "seemed", "seeming", "it"}
PREPOSITIONS = {"with", "of", "in", "on", "at", "to", "from", "by", "through",
                "behind", "into"}
# deletions that are only logged (frames, hedges, function words)
FIRST_PERSON = {"i", "me", "my", "mine", "myself", "we", "us", "our", "ours",
                "ourselves", "i'm", "i've", "i'd", "i'll"}
FRAME_VERBS = {"see", "feel", "seem", "look", "appear", "think", "find", "perceive",
               "experience", "notice", "say", "describe"}
HEDGES = {"just", "almost", "somehow", "really", "very", "quite", "rather", "kind",
          "sort", "little", "bit", "so", "such", "too", "totally", "completely",
          "utterly", "extremely", "slightly", "somewhat", "pretty", "even", "still",
          "simply", "only", "actually", "literally", "basically", "sometimes",
          "often", "always", "suddenly", "more", "most", "less", "much"}
# captions with these words go to review even when both models agree (they
# may describe the experience, a person or an event, not a scene)
META_WORDS = {"feeling", "feelings", "experience", "experiences", "experiencing",
              "experienced", "complaint", "complaints", "sense", "awareness",
              "perception", "perceptions", "judged", "symptom", "symptoms",
              "modality", "self", "oneself", "myself", "himself", "herself",
              "themself", "themselves", "identity", "person", "people", "life",
              "lives", "memory", "memories", "mind", "consciousness", "emotional",
              "thought", "thoughts", "derealization", "depersonalization", "dpdr",
              "dp", "dr", "patient", "patients", "you", "he", "she", "him", "her",
              "his", "they", "them", "their"}
OPENERS = [("as", "if"), ("as", "though"), ("like",)]
POSSESSIVE = {"my", "our", "your", "his", "her", "their", "the", "a", "an"}
BE_FORMS = {"is", "are", "was", "were", "be", "been", "being", "it"}
FUNCTION = ALLOWED_ADD | PREPOSITIONS | {
    "and", "or", "but", "that", "this", "these", "those", "there", "its", "which",
    "who", "what", "when", "while", "you", "your", "he", "she", "they", "them",
    "his", "her", "their", "one", "am", "has", "have", "had", "do", "does", "did",
    "all", "everything", "something", "anything", "everyone", "everybody",
    "someone", "somebody", "around", "about", "for", "up", "out", "over", "under",
    "s", "t", "would", "could", "can", "will", "may", "might", "must", "should"}
FRAME_SUBJECTS = {"it", "everything", "everyone", "things", "they", "he", "she",
                  "you", "this", "that", "all", "world"}
NEGATION = {"not", "no", "never", "nothing", "none", "nobody", "nowhere", "neither",
            "nor", "without", "n't"}
BAD_START = ({"a", "an", "the", "it", "is", "are", "was", "were", "be", "been",
              "being", "and", "or", "that", "this", "i", "you", "he", "she", "we",
              "they"} | PREPOSITIONS) - {"like"}
IRREGULAR = {
    "seen": "see", "saw": "see", "felt": "feel", "made": "make", "became": "become",
    "was": "be", "were": "be", "is": "be", "are": "be", "been": "be", "being": "be",
    "am": "be", "went": "go", "gone": "go", "grew": "grow", "grown": "grow",
    "got": "get", "gotten": "get", "thought": "think", "took": "take",
    "taken": "take", "came": "come", "gave": "give", "given": "give", "had": "have",
    "has": "have", "did": "do", "done": "do", "does": "do", "left": "leave",
    "lost": "lose", "brought": "bring", "held": "hold", "drew": "draw",
    "drawn": "draw", "fell": "fall", "fallen": "fall", "hid": "hide",
    "hidden": "hide", "lay": "lie", "lain": "lie", "shone": "shine",
    "worn": "wear", "wore": "wear", "built": "build", "spun": "spin",
    "struck": "strike", "torn": "tear", "tore": "tear", "woven": "weave",
    "frozen": "freeze", "froze": "freeze", "dead": "dead", "died": "die",
    "told": "tell", "said": "say", "heard": "hear", "found": "find",
    "kept": "keep", "meant": "mean", "stood": "stand", "sat": "sit",
    "began": "begin", "begun": "begin", "broke": "break", "broken": "break",
    "chose": "choose", "chosen": "choose", "spoke": "speak", "spoken": "speak",
    "woke": "wake", "woken": "wake", "shook": "shake", "shaken": "shake",
    "ran": "run", "wrote": "write", "written": "write", "knew": "know",
    "known": "know", "flew": "fly", "flown": "fly", "blew": "blow",
    "blown": "blow", "threw": "throw", "thrown": "throw", "sank": "sink",
    "sunk": "sink", "slid": "slide", "bent": "bend", "sent": "send",
}
SUFFIXES = ["ingly", "edly", "ness", "ings", "ing", "ied", "ies", "est", "ed",
            "ly", "er", "es", "s", "y"]


# ============================================================
# Helpers
# ============================================================

def P(*n):
    return os.path.join(STAGEC_DIR, *n)


def clean(t):
    t = unicodedata.normalize("NFKC", str(t))
    return " ".join(t.replace("’", "'").replace("‘", "'").split())


def unquote(t):
    """Quotation marks removed (kept: apostrophes inside words, as in
    "don't"); they are punctuation from the source, not words, and lower
    the fluency score."""
    t = re.sub(r'["\u201c\u201d]', "", t)
    t = re.sub(r"(?<![A-Za-z])'|'(?![A-Za-z])", "", t)
    return " ".join(t.split())


def words(t):
    return re.findall(r"[a-z]+(?:'[a-z]+)?", clean(t).lower())


def norm_caption(c):
    return " ".join(words(c))


def _norm(w):
    if len(w) > 3 and w.endswith("e"):
        w = w[:-1]
    if len(w) > 3 and w[-1] == w[-2] and w[-1] not in "aeiou":
        w = w[:-1]
    return w


def variants(w):
    """Possible roots of a word: the word (irregular forms mapped) and the
    word without each suffix it ends with, each without a final "e" or a
    final doubled consonant (fog / foggy, come / comes / coming, dull /
    duller, other / others share a root)."""
    base = IRREGULAR.get(w, w)
    out = {_norm(base)}
    for s in SUFFIXES:
        if base.endswith(s) and len(base) - len(s) >= 3:
            out.add(_norm(base[:-len(s)]))
    return out


def stem(w):
    """The shortest root (used for the prefix rule)."""
    return min(variants(w), key=len)


def same_root(a, b):
    """Same word in another grammatical form: a shared root, or one root a
    prefix of the other (at least 4 letters: flat / flatten, dream /
    dreamlike, blur / blurriness)."""
    if a == b or variants(a) & variants(b):
        return True
    short, long_ = sorted((stem(a), stem(b)), key=len)
    return len(short) >= 4 and long_.startswith(short)


def is_frame_verb(w):
    roots = set().union(*(variants(v) for v in FRAME_VERBS))
    return bool(variants(w) & roots)


def adjacent(quote, context, n):
    """Words within n of the quote in the context (both sides)."""
    qw, cw = words(quote), words(context)
    if not qw:
        return []
    for i in range(len(cw) - len(qw) + 1):
        if cw[i:i + len(qw)] == qw:
            return cw[max(0, i - n):i] + cw[i + len(qw):i + len(qw) + n]
    return []


def check(caption, quotes, contexts):
    """Word, form and deletion checks of one caption. quotes / contexts:
    the text's quotes and their contexts (pairs)."""
    cw = words(caption)
    qwords = [w for q in quotes for w in words(q)]
    adj = []
    for q, c in zip(quotes, contexts):
        adj += adjacent(q, c, ADJACENT_WORDS)
    unsupported, restored, forms, added = [], [], [], []
    for i, w in enumerate(cw):
        if w in qwords:
            continue
        if w in NEGATION:          # reported separately (negation_added)
            continue
        m = next((q for q in qwords if same_root(w, q)), None)
        if m:
            forms.append(f"{m}->{w}")
            continue
        if w in ALLOWED_ADD:
            added.append(w)
            continue
        if any(same_root(w, a) or w == a for a in adj):
            restored.append(w)
            continue
        if (w == "seen" and i > 0 and cw[i - 1] in {"if", "like", "though"}
                and i + 1 < len(cw) and cw[i + 1] in PREPOSITIONS):
            added.append("seen")
            continue
        unsupported.append(w)
    neg = [w for w in cw if w in NEGATION and w not in qwords]
    # form
    if not cw:
        form_ok, form_note = False, "empty"
    elif cw[0] == "like":
        form_ok, form_note = True, ""
    elif cw[0] == "as":
        # "as if ...", "as though ...", or "as <adjective> as ..."
        form_ok = len(cw) > 1 and (cw[1] in {"if", "though"} or
                                   (cw[1] not in BAD_START and "as" in cw[2:]))
        form_note = "" if form_ok else "'as' not followed by 'if' / 'though' / '... as'"
    elif cw[0] in BAD_START:
        form_ok, form_note = False, f"starts with '{cw[0]}'"
    else:
        form_ok, form_note = True, ""
    # deletions (from the first quote; the others are the same text)
    q0 = words(quotes[0]) if quotes else []
    deleted = [w for w in q0 if not any(w == c or same_root(w, c) for c in cw)]
    # a verb like "see", "feel", "seem" counts as a frame only at the start of
    # the quote or after a pronoun ("I see", "it seems", "everything seems")
    frame = {i for i, w in enumerate(q0) if is_frame_verb(w) and
             (i == 0 or q0[i - 1] in FIRST_PERSON | FRAME_SUBJECTS)}
    content_del = [w for i, w in enumerate(q0)
                   if w in deleted and i not in frame and w not in FIRST_PERSON
                   and w not in HEDGES and w not in FUNCTION]
    return {"unsupported": unsupported, "negation_added": neg, "form_ok": form_ok,
            "form_note": form_note, "restored": restored, "forms": forms,
            "added": added, "deleted": deleted, "content_deleted": content_del,
            "words_ok": not unsupported and not neg}


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


def norm_tid(t):
    m = re.match(r"^\s*T0*(\d+)\s*$", str(t), flags=re.I)
    return f"T{int(m.group(1)):04d}" if m else str(t).strip()


def slor_scores(sentences, cache_path):
    """SLOR per sentence: (log P_GPT2(s) - sum of the words' unigram log
    probabilities) / number of words. Natural logs; the sentence is scored
    after GPT-2's start token, so every token is predicted."""
    from wordfreq import tokenize, word_frequency
    cache = json.load(open(cache_path)) if os.path.exists(cache_path) else {}
    todo = [s for s in dict.fromkeys(sentences) if s not in cache]
    if todo:
        import torch
        from transformers import GPT2LMHeadModel, GPT2TokenizerFast
        tok = GPT2TokenizerFast.from_pretrained(FLUENCY_MODEL)
        lm = GPT2LMHeadModel.from_pretrained(FLUENCY_MODEL).eval()
        with torch.no_grad():
            for s in todo:
                ids = tok(s, return_tensors="pt").input_ids
                ids = torch.cat([torch.tensor([[tok.bos_token_id]]), ids], dim=1)
                n = ids.shape[1] - 1
                cache[s] = -float(lm(ids, labels=ids).loss) * n
        json.dump(cache, open(cache_path, "w"), indent=0)
    out = {}
    for s in sentences:
        ws = tokenize(s, "en")
        uni = sum(np.log(max(word_frequency(w, "en"), 1e-9)) for w in ws)
        out[s] = (cache[s] - uni) / max(len(ws), 1)
    return out


def caption_key(c):
    """The caption without the conventions' differences: the opener ("as
    if", "as though", "like"), first-person words, hedges ("sort of", "kind
    of", "a bit"), articles and possessives, forms of "be", and then a
    leading "seen" and a leading preposition."""
    w = words(c)
    for op in OPENERS:
        if tuple(w[:len(op)]) == op:
            w = w[len(op):]
            break
    t = " ".join(w)
    for h in ("sort of", "kind of", "a bit"):
        t = re.sub(rf"\b{h}\b", " ", t)
    w = [x for x in t.split() if x not in FIRST_PERSON and x not in HEDGES
         and x not in POSSESSIVE and x not in BE_FORMS]
    if w[:1] == ["seen"]:
        w = w[1:]
    if w[:1] and w[0] in PREPOSITIONS:
        w = w[1:]
    return w


def suggest(c0, c1, m0, m1):
    """(model, conventions) when the two captions differ only by the
    conventions; None otherwise."""
    if caption_key(c0) != caption_key(c1):
        return None
    w0, w1 = words(c0), words(c1)

    def score(w):
        as_if = w[:2] in (["as", "if"], ["as", "though"])
        return (sum(x in FIRST_PERSON for x in w), sum(x in HEDGES for x in w)
                + sum(" ".join(w).count(h) for h in ("sort of", "kind of", "a bit")),
                0 if as_if else 1)
    s0, s1 = score(w0), score(w1)
    if s0 == s1:
        return (m0, "equivalent")
    # suggest only a caption at least as good on every convention
    if all(a <= b for a, b in zip(s0, s1)):
        win, ww, wl = m0, w0, w1
    elif all(b <= a for a, b in zip(s0, s1)):
        win, ww, wl = m1, w1, w0
    else:
        return None
    conv = []
    sw, sl = score(ww), score(wl)
    if sw[2] < sl[2]:
        conv.append("2" if "seen" in ww[:3] else "1")
    if sw[0] < sl[0]:
        conv.append("3")
    if sw[1] < sl[1]:
        conv.append("4")
    return (win, "convention " + ",".join(conv) if conv else "equivalent")


# ============================================================
# Main
# ============================================================

def main():
    t0 = time.time()
    print("=" * 70)
    print("v4_stageC_check.py")
    print("=" * 70)
    problems = []

    def problem(kind, model="", detail=""):
        problems.append({"kind": kind, "model": model, "detail": detail})

    st = pd.read_csv(P("stageC_texts.csv"), dtype=str).fillna("")
    ids = list(st.text_id)
    id_set = set(ids)
    info = st.set_index("text_id")
    pm = pd.read_csv(os.path.join(STAGEB_DIR, "stageB_phrase_map.csv"), dtype=str).fillna("")
    pm = pm[pm.included.str.lower() == "true"]
    sel = pd.read_csv(os.path.join(STAGEB_DIR, "stageB_clip_selection.csv"),
                      dtype=str).fillna("")
    sel = sel[sel.clip_use.str.lower() == "yes"]
    axis_names = dict(zip(sel.axis_id, sel.name_stageC))

    def qc(t):
        r = info.loc[t]
        quotes = [q for q in r.quotes.split(" || ") if q] or [r.text]
        contexts = r.contexts.split(" || ") if r.contexts else []
        # pair every quote with every context (identical texts share words)
        pq = [q for q in quotes for _ in (contexts or [""])]
        pc = [c for _ in quotes for c in (contexts or [""])]
        return quotes, pq, pc

    # --- model outputs ---
    out = {}
    for m in MODELS:
        files = sorted(glob.glob(P("outputs", f"stageC_{m}_batch*.json")))
        if not files:
            raise SystemExit(f"No Stage C files for {m} in {P('outputs')}")
        a, seen = {}, Counter()
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
                t = norm_tid(x.get("id", ""))
                if t not in id_set:
                    problem("unknown_text", m, f"{os.path.basename(f)}: {x.get('id')!r}")
                    continue
                seen[t] += 1
                cap = x.get("caption")
                cap = unquote(clean(cap)) if isinstance(cap, str) and cap.strip() else ""
                drop = str(x.get("drop") or "").strip().lower()
                if drop in ("null", "none"):
                    drop = ""
                if cap and drop:
                    problem("caption_and_drop", m, f"{t}: caption kept, drop ignored")
                    drop = ""
                if not cap and not drop:
                    problem("no_caption_no_drop", m, f"{t}: counted as drop 'cannot'")
                    drop = "cannot"
                if drop and drop not in ("cannot", "outside"):
                    problem("unknown_drop_reason", m, f"{t}: {drop!r}")
                if t in a:
                    if a[t] != (cap, drop):
                        problem("answered_twice_differently", m, f"{t} (first kept)")
                    continue
                a[t] = (cap, drop)
        missing = [t for t in ids if t not in a]
        for t in missing:
            problem("missing_text", m, t)
        for t, c in seen.items():
            if c > 1:
                problem("answered_more_than_once", m, f"{t}: {c} times")
        out[m] = a
        print(f"  {m}: {len(files)} files, {len(a)} of {len(ids)} texts answered, "
              f"{len(missing)} missing")

    # --- checks per candidate ---
    rows = []
    for t in ids:
        quotes, pq, pc = qc(t)
        r = {"text_id": t, "axis_id": info.at[t, "axis_id"], "text": info.at[t, "text"],
             "text_class": info.at[t, "text_class"], "n_phrases": info.at[t, "n_phrases"]}
        for m in MODELS:
            cap, drop = out[m].get(t, ("", "missing"))
            r[f"caption_{m}"], r[f"drop_{m}"] = cap, drop
            if cap:
                c = check(cap, pq, pc)
                r[f"words_ok_{m}"] = c["words_ok"]
                r[f"unsupported_{m}"] = " ".join(c["unsupported"] + c["negation_added"])
                r[f"form_ok_{m}"] = c["form_ok"]
                r[f"form_note_{m}"] = c["form_note"]
                r[f"content_deleted_{m}"] = " ".join(c["content_deleted"])
                r[f"restored_{m}"] = " ".join(c["restored"])
                r[f"meta_{m}"] = " ".join(dict.fromkeys(
                    w for w in words(cap) if w in META_WORDS))
            else:
                for k in ("words_ok", "form_ok"):
                    r[f"{k}_{m}"] = ""
                for k in ("unsupported", "form_note", "content_deleted", "restored",
                          "meta"):
                    r[f"{k}_{m}"] = ""
        rows.append(r)
    C = pd.DataFrame(rows)

    # --- fluency (SLOR) ---
    flu_ok, flu_note = USE_FLUENCY, ""
    sl = {}
    if USE_FLUENCY:
        sents = [TEMPLATE.format(c) for m in MODELS for c in C[f"caption_{m}"] if c]
        try:
            sl = slor_scores(sents, P("stageC_lm_cache.json"))
        except Exception as e:  # missing package or model
            flu_ok, flu_note = False, f"skipped ({type(e).__name__}: {e})"
            print(f"  WARNING: fluency check {flu_note}")
    alls = np.array(list({s_: v for s_, v in sl.items()}.values()))
    if len(alls):
        q1, q3 = np.percentile(alls, [25, 75])
        cut = q1 - SLOR_IQR * (q3 - q1)
        med = float(np.median(alls))
    else:
        cut = med = float("nan")
    for m in MODELS:
        C[f"slor_{m}"] = [sl.get(TEMPLATE.format(c), np.nan) if c else np.nan
                          for c in C[f"caption_{m}"]]
        C[f"fluency_flag_{m}"] = (C[f"slor_{m}"] < cut) if len(alls) else False

    # --- status ---
    def valid(r, m):
        return bool(r[f"caption_{m}"]) and r[f"words_ok_{m}"] is True and \
            r[f"form_ok_{m}"] is True

    def clean_pass(r, m):
        return valid(r, m) and not r[f"fluency_flag_{m}"] and \
            not r[f"content_deleted_{m}"] and not r[f"meta_{m}"]

    m0, m1 = MODELS
    status, reasons = [], []
    for r in C.to_dict("records"):
        c0, c1 = norm_caption(r[f"caption_{m0}"]), norm_caption(r[f"caption_{m1}"])
        d0, d1 = r[f"drop_{m0}"], r[f"drop_{m1}"]
        if c0 and c0 == c1 and clean_pass(r, m0):
            status.append("agreed"); reasons.append("")
        elif d0 and d1 and d0 != "missing" and d1 != "missing":
            status.append("agreed drop"); reasons.append(f"{d0} / {d1}")
        else:
            why = []
            if "missing" in (d0, d1):
                why.append("a model gave no answer")
            elif bool(d0) != bool(d1):
                why.append("one model dropped it")
            elif c0 != c1:
                why.append("captions differ")
            for m in MODELS:
                if r[f"caption_{m}"]:
                    if r[f"words_ok_{m}"] is not True:
                        why.append(f"{m}: word not from source ({r[f'unsupported_{m}']})")
                    if r[f"form_ok_{m}"] is not True:
                        why.append(f"{m}: form ({r[f'form_note_{m}']})")
                    if r[f"content_deleted_{m}"]:
                        why.append(f"{m}: content word deleted "
                                   f"({r[f'content_deleted_{m}']})")
                    if r[f"fluency_flag_{m}"]:
                        why.append(f"{m}: fluency")
                    if r[f"meta_{m}"]:
                        why.append(f"meta word ({r[f'meta_{m}']})")
            status.append("review"); reasons.append("; ".join(dict.fromkeys(why)))
    C["status"], C["reason"] = status, reasons
    # suggestions (conventions): only when both captions pass the word and
    # form checks and differ only by the conventions
    sug = []
    for r in C.to_dict("records"):
        if r["status"] == "review" and valid(r, m0) and valid(r, m1) and \
                norm_caption(r[f"caption_{m0}"]) != norm_caption(r[f"caption_{m1}"]):
            x = suggest(r[f"caption_{m0}"], r[f"caption_{m1}"], m0, m1)
            sug.append(f"{x[0]} ({x[1]})" if x else "")
        else:
            sug.append("")
    C["suggestion"] = sug
    C.to_csv(P("stageC_candidates.csv"), index=False, float_format="%.1f")

    # --- review ---
    old = {}
    if os.path.exists(P("stageC_review.csv")):
        o = pd.read_csv(P("stageC_review.csv"), dtype=str).fillna("")
        old = {r.text_id: (r.decision.strip(), r.comment) for r in o.itertuples()}
    rv, final = [], []
    for r in C.to_dict("records"):
        t = r["text_id"]
        quotes, pq, pc = qc(t)
        rec = {"text_id": t, "axis_id": r["axis_id"], "text": r["text"],
               "text_class": r["text_class"]}
        # a decision you wrote for an accepted text (a row you added to the
        # review file) overrides the automatic acceptance
        if r["status"] in ("agreed", "agreed drop") and old.get(t, ("", ""))[0]:
            r = {**r, "status": "review",
                 "reason": (r["reason"] + "; " if r["reason"] else "")
                 + f"accepted automatically ({r['status']}); your decision applies"}
        if r["status"] == "agreed":
            cap, how = r[f"caption_{m0}"], "agreed"
        elif r["status"] == "agreed drop":
            cap, how = "", f"dropped by both ({r['reason']})"
        else:
            dec, com = old.get(t, ("", ""))
            sg = r["suggestion"]
            from_sug = False
            if not dec and sg and ACCEPT_SUGGESTIONS:
                dec, from_sug = sg.split(" (")[0], True
            dl = dec.lower()
            note = ""
            if dl in MODELS:
                cap = r[f"caption_{dl}"]
                how = (f"review: suggestion, {dl} ({sg.split(' (')[1].rstrip(')')})"
                       if from_sug else f"review: {dl}")
                if not cap:
                    cap, how, note = "", "open", f"{dl} has no caption"
                elif not valid(r, dl):
                    # a flag (fluency, deleted content word) can be accepted by
                    # you; a failed word or form check cannot
                    cap, how = "", "open"
                    note = (f"{dl}'s caption fails the word or form check; "
                            "write your own caption or drop")
            elif dl == "drop":
                cap, how = "", "review: dropped"
            elif dl in ("not_scene", "not scene"):
                cap, how = "", "review: dropped (not_scene)"
            elif dec:
                c = check(dec, pq, pc)
                if c["words_ok"] and c["form_ok"]:
                    cap, how = unquote(clean(dec)), "review: your caption"
                else:
                    cap, how = "", "open"
                    note = ("your caption fails: " + " ".join(c["unsupported"] +
                            c["negation_added"]) + " " + c["form_note"]).strip()
            else:
                cap, how = "", "open"
            rv.append({"text_id": t, "axis_id": r["axis_id"], "text": r["text"],
                       "context": info.at[t, "shown"].split(" || ")[0],
                       "reason": r["reason"],
                       f"caption_{m0}": r[f"caption_{m0}"] or f"(drop: {r[f'drop_{m0}']})",
                       f"caption_{m1}": r[f"caption_{m1}"] or f"(drop: {r[f'drop_{m1}']})",
                       "suggestion": sg,
                       "decision": "" if from_sug else dec, "comment": com,
                       "problem": note})
        rec["caption"], rec["how"] = cap, how
        if cap:
            c = check(cap, pq, pc)
            rec.update({"restored_from_context": " ".join(c["restored"]),
                        "form_changes": " ".join(c["forms"]),
                        "added": " ".join(c["added"]),
                        "deleted": " ".join(c["deleted"]),
                        "content_deleted": " ".join(c["content_deleted"])})
        rec.update({"quote": quotes[0], "phrase_ids": info.at[t, "phrase_ids"],
                    "documents": info.at[t, "documents"],
                    "n_phrases": info.at[t, "n_phrases"]})
        final.append(rec)
    rvd = pd.DataFrame(rv, columns=["text_id", "axis_id", "text", "context", "reason",
                                    f"caption_{m0}", f"caption_{m1}", "suggestion",
                                    "decision", "comment", "problem"])
    rvd.to_csv(P("stageC_review.csv"), index=False)
    F = pd.DataFrame(final)
    n_open = int((F.how == "open").sum())
    F.to_csv(P("stageC_captions.csv"), index=False)

    # --- support after Stage C ---
    capd = F[F.caption != ""][["text_id", "axis_id"]]
    pmc = pm.merge(capd, on="text_id")
    axrows = []
    for a in sel.axis_id:
        sub = pmc[pmc.axis_id == a]
        caps = F[(F.axis_id == a) & (F.caption != "")]
        row = {"axis_id": a, "name": axis_names[a],
               "texts_before": int((F.axis_id == a).sum()),
               "texts_captioned": int(len(caps)),
               "distinct_captions": int(caps.caption.map(norm_caption).nunique()),
               "dropped": int(((F.axis_id == a) & F.how.str.contains("drop")).sum()),
               "open": int(((F.axis_id == a) & (F.how == "open")).sum())}
        for sc, classes in SCENARIOS.items():
            s = sub[sub.sense_class.isin(classes)]
            nt, nd = s.text_id.nunique(), s.doc_id.nunique()
            nf = int((s.first_person == "yes").sum())
            row.update({f"{sc}_texts": nt, f"{sc}_docs": nd, f"{sc}_first_person": nf,
                        f"{sc}_kept": nt >= MIN_TEXTS and nd >= MIN_DOCS and
                        nf >= MIN_FIRST_PERSON})
        axrows.append(row)
    AX = pd.DataFrame(axrows)
    AX.to_csv(P("stageC_axes.csv"), index=False)
    pr = pd.DataFrame(problems, columns=["kind", "model", "detail"])
    pr.to_csv(P("stageC_problems.csv"), index=False)

    # ========================================================
    # Report
    # ========================================================
    L = []
    W = L.append
    W("STAGE C: CAPTION FORMS, CHECKS AND REVIEW (v4_protocol.md 1.6, section 6)")
    W("=" * 78)
    W(f"Date: {time.strftime('%Y-%m-%d %H:%M')}")
    W(f"Texts: {len(ids):,} in {len(sel)} axes; template: \"{TEMPLATE.format('___')}\"")
    W("")
    W("1. OUTPUT CHECKS (stageC_problems.csv)")
    W("-" * 78)
    if len(pr):
        for (k, m), n in pr.groupby(["kind", "model"]).size().items():
            W(f"  {m:<9}{k:<32}{n:>5}")
        W("  Missing texts: ask again in the same chat and save the answer as an")
        W("  extra file (e.g. stageC_<model>_batch09.json).")
    else:
        W("  no problems: every text answered exactly once")
    W("")
    W("2. CAPTIONS AND DROPS PER MODEL")
    W("-" * 78)
    W(f"  {'':<9}{'caption':>8}{'cannot':>8}{'outside':>8}{'words ok':>10}"
      f"{'form ok':>9}{'content del':>13}{'fluency':>9}{'meta':>6}")
    for m in MODELS:
        has = C[f"caption_{m}"] != ""
        W(f"  {m:<9}{int(has.sum()):>8}{int((C[f'drop_{m}'] == 'cannot').sum()):>8}"
          f"{int((C[f'drop_{m}'] == 'outside').sum()):>8}"
          f"{int((C.loc[has, f'words_ok_{m}'] == True).sum()):>10}"  # noqa: E712
          f"{int((C.loc[has, f'form_ok_{m}'] == True).sum()):>9}"  # noqa: E712
          f"{int((C.loc[has, f'content_deleted_{m}'] != '').sum()):>13}"
          f"{int(C.loc[has, f'fluency_flag_{m}'].sum()):>9}"
          f"{int((C.loc[has, f'meta_{m}'] != '').sum()):>6}")
    W("  fluency: " + (f"SLOR, median {med:.2f}, flagged below {cut:.2f} "
                       f"(Q1 - {SLOR_IQR} x IQR)" if flu_ok and len(alls)
                       else flu_note or "off"))
    W("")
    W("3. AGREEMENT BETWEEN THE MODELS")
    W("-" * 78)
    both = (C[f"caption_{m0}"] != "") & (C[f"caption_{m1}"] != "")
    same = both & (C[f"caption_{m0}"].map(norm_caption) == C[f"caption_{m1}"].map(norm_caption))
    d0, d1 = C[f"drop_{m0}"] != "", C[f"drop_{m1}"] != ""
    W(f"  both wrote a caption: {int(both.sum())}; the same caption: {int(same.sum())} "
      f"({(same.sum() / both.sum() if both.sum() else 0):.0%})")
    W(f"  both dropped: {int((d0 & d1).sum())}; only {m0} dropped: {int((d0 & ~d1).sum())}; "
      f"only {m1} dropped: {int((~d0 & d1).sum())}")
    W("")
    W("4. STATUS")
    W("-" * 78)
    W(f"  accepted automatically: {int((C.status == 'agreed').sum())} "
      "(same caption, every check passed)")
    W(f"  dropped by both: {int((C.status == 'agreed drop').sum())}")
    W(f"  to review (stageC_review.csv): {len(rvd)}; still open: {n_open}")
    if len(rvd):
        why = Counter(w.split(" (")[0].replace(f"{m0}: ", "").replace(f"{m1}: ", "")
                      for x in rvd.reason for w in x.split("; ") if w)
        for k, v in why.most_common():
            W(f"    {v:>4}  {k}")
        ns = int((rvd.suggestion != "").sum()) if len(rvd) else 0
        W(f"  with a suggestion (conventions): {ns}"
          + ("; taken where the decision is empty (ACCEPT_SUGGESTIONS)" if ACCEPT_SUGGESTIONS
             else "; set ACCEPT_SUGGESTIONS = True to take them where the decision is empty"))
        if len(rvd):
            for k, v in Counter(x.split(" (")[1].rstrip(")") for x in rvd.suggestion
                                if x).most_common():
                W(f"    {v:>4}  {k}")
        W(f"  dropped as not_scene by you: {int((F.how == 'review: dropped (not_scene)').sum())}")
        bad = rvd[rvd.problem != ""]
        if len(bad):
            W(f"  decisions that cannot be used ({len(bad)}): see the 'problem' column")
    W("")
    W("5. EDITS IN THE FINAL CAPTIONS (stageC_captions.csv)")
    W("-" * 78)
    fc = F[F.caption != ""]
    if len(fc):
        W(f"  unchanged quote: {int((fc.caption.map(norm_caption) == fc.quote.map(norm_caption)).sum())}"
          f" of {len(fc)}")
        W(f"  with a word restored from context: {int((fc.restored_from_context != '').sum())}")
        W(f"  with a change of form: {int((fc.form_changes != '').sum())}")
        W(f"  with a deleted content word (accepted by you): "
          f"{int((fc.content_deleted != '').sum())}")
        ad = Counter(w for x in fc.added for w in x.split())
        W("  added words: " + ", ".join(f"{k} {v}" for k, v in ad.most_common()))
    W("")
    W("6. SUPPORT AFTER STAGE C (keeping rules as in Stage B: >= "
      f"{MIN_TEXTS} texts, >= {MIN_DOCS} documents, >= {MIN_FIRST_PERSON} first-person)")
    W("-" * 78)
    W(f"    {'axis':<5}{'name':<30}{'before':>7}{'capt.':>6}{'distinct':>9}"
      f"{'docs':>5}{'1st':>5}   kept all/vis/n-s")
    for r in AX.itertuples():
        fl = "/".join("Y" if getattr(r, f"{sc}_kept") else "-" for sc in SCENARIOS)
        W(f"    {r.axis_id:<5}{str(r.name)[:29]:<30}{r.texts_before:>7}"
          f"{r.texts_captioned:>6}{r.distinct_captions:>9}{r.all_docs:>5}"
          f"{r.all_first_person:>5}   {fl}")
    k = AX[[f"{sc}_kept" for sc in SCENARIOS]].sum()
    W(f"  kept: all phrases {int(k.iloc[0])}, vision only {int(k.iloc[1])}, "
      f"not-specific only {int(k.iloc[2])}")
    W("  (distinct = different captions; identical captions from different texts")
    W("   become one anchor in Stage D.)")
    if n_open:
        W("  Decide the open rows of stageC_review.csv and run again; the numbers")
        W("  above change until then.")
    else:
        W("  STAGE C COMPLETE: every text decided. Next: Stage D (counterfactuals).")
    W("")
    W(f"Runtime {time.time() - t0:.1f} s")
    open(P("stageC_check_report.txt"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    json.dump({"script": "v4_stageC_check.py", "protocol": "1.6",
               "created": time.strftime("%Y-%m-%d %H:%M:%S"), "open": n_open,
               "n_review": len(rvd), "slor_median": med, "slor_cut": cut,
               "slor_iqr": SLOR_IQR, "accept_suggestions": ACCEPT_SUGGESTIONS,
               "adjacent_words": ADJACENT_WORDS},
              open(P("check_meta.json"), "w"), indent=2, default=str)


if __name__ == "__main__":
    main()
