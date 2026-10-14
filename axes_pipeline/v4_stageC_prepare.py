"""
v4_stageC_prepare.py  (v4 axes, Stage C: input for the caption forms)

Builds the input of Stage C (v4_protocol.md, version 1.5, section 6, Stage
C) and writes the messages to paste into the two models' chats.

What it does
------------
1. Reads the result of Stage B (stageB_final_assignment.csv) and the
   selection for CLIP (stageB_clip_selection.csv), and keeps the texts of
   the selected axes only (10 axes, protocol 6.2).
2. For every text, finds its phrases (stageB_phrase_map.csv) and their
   context sentences (stageA_verified.csv), and marks the phrase inside each
   context with [brackets]. The models may restore a word standing directly
   next to the brackets (protocol 6, Stage C); the check script verifies it.
3. Orders the texts by axis (A01, A02, ...) and, within an axis, in a fixed
   random order, and writes batches of BATCH_SIZE texts. Each batch message
   is complete on its own (the instructions are repeated), and every batch
   goes to the same chat.

Models see the phrase, its axis's name and definition, and its context:
no documents, speakers, counts or sense classes.

Outputs (OUTPUT_DIR)
--------------------
  stageC_texts.csv          one row per text: axis, text, contexts, phrase
                            ids, documents, sense class (text_class)
  C_batch_NN.txt            the messages to paste
  outputs/                  put the model outputs here
  stageC_prepare_report.txt, prepare_meta.json

How to run Stage C (each model in a new chat of the memory-free account)
------------------------------------------------------------------------
  In one new chat per model, paste C_batch_01.txt, then 02, ... in the same
  chat. Save each JSON answer as
      outputs/stageC_<model>_batchNN.json      (model = claude or chatgpt)
  Then run v4_stageC_check.py.

Usage
-----
    conda activate drl2drv
    python v4_stageC_prepare.py
"""

import json
import os
import re
import time
import unicodedata

import numpy as np
import pandas as pd


# ============================================================
# SETTINGS
# ============================================================

STAGEV_DIR    = "axes/v4/stageV"
STAGEB_DIR    = "axes/v4/stageB"
OUTPUT_DIR    = "axes/v4/stageC"
BATCH_SIZE    = 120
RANDOM_SEED   = 42
CONTEXT_WORDS = 12      # words of context shown on each side of the phrase
MAX_CONTEXTS  = 2       # contexts shown per text (texts can have several phrases)

# Parts removed from a definition for CLIP (protocol 6.2): a phrase that
# expresses only this part is dropped as "outside".
EXCLUDED = {
    "A01": "the world or things in it seeming not to exist",
    "A15": "seeming more real, without looking more vivid, intense or detailed",
}


# ============================================================
# Prompt (v4_protocol.md 1.5, Stage C; no derealization examples)
# ============================================================

PROMPT_C = """I have a list of short phrases. Each was taken word for word from a
published text, and each has already been assigned to a QUALITY: a way in
which the world, places or objects can appear. The qualities are defined
below. Next to each phrase is its context, the sentence it was taken from,
with the phrase itself in [square brackets].

TASK
Turn each phrase into a CAPTION: the words that complete the sentence

    a scene that looks ___

so that the whole sentence is grammatical and reads naturally, like a short
description of a picture. Change the phrase as little as possible.

WHAT YOU MAY DO
1. Add only these words: a, an, the, like, as, if, though, with, of, in, on,
   at, to, from, by, through, behind, into, is, are, was, were, be, been,
   being, seem, seems, seemed, it.
2. Add "seen" only as "as if seen" or "like seen", directly before words
   that begin with one of the prepositions in rule 1 (for example before
   "through ..." or "behind ...").
3. Restore a word that stands directly before or after the brackets in the
   context, for example a comparison word just before the phrase.
4. Delete words that place the experience in a person ("I see", "I feel",
   "it seems to me", "my"), hedges and intensifiers ("just", "almost",
   "somehow", "really", "very", "quite"), and articles, prepositions or
   forms of "be" when they would make the sentence ungrammatical.
5. Change the grammatical form of a word while keeping its root (a verb
   into its past participle, a noun into its adjective).

WHAT YOU MAY NOT DO
- Add any other word, replace a word with a synonym, or add a negation.
- Delete a word that carries the quality.
- Change the order of the phrase's words, unless the sentence needs it.

WHEN NOT TO WRITE A CAPTION
Write no caption, and give the reason in "drop", only when:
- "cannot": the phrase cannot become a grammatical caption under these
  rules;
- "outside": the phrase expresses only what its quality's definition
  EXCLUDES (only qualities with an "Excludes" line).
Do not leave out a phrase for any other reason.

ILLUSTRATION (an unrelated topic, taste, with the sentence "a drink that
tastes ___"; in this illustration the word allowed by rule 2 is "tasted";
in your task it is "seen")
- "watery"  ->  "watery"  (no change)
- "it tasted to me like water from a tin cup"  ->  "like water from a tin
  cup"  (deleted "it tasted to me")
- context "it was almost like [the metal of an old coin], I thought"  ->
  "like the metal of an old coin"  (restored "like" from the context)
- "I found it thin and metallic"  ->  "thin and metallic"
- "through a straw"  ->  "as if tasted through a straw"  (rule 2)
- "my grandmother's recipe"  ->  no caption, drop "cannot"

OUTPUT
One JSON array inside one code block labelled json, one object per phrase,
in the order of the list:
[{{"id": "T0001", "caption": "...", "drop": null}},
 {{"id": "T0002", "caption": null, "drop": "cannot"}}, ...]
After the code block write the number of phrases in the array and then the
line: END OF BATCH {k}

QUALITIES
{qualities}

THE PHRASES (batch {k} of {n})
"""


# ============================================================
# Helpers
# ============================================================

def P(*n):
    return os.path.join(OUTPUT_DIR, *n)


def clean(t):
    t = unicodedata.normalize("NFKC", str(t))
    return " ".join(t.replace("’", "'").replace("‘", "'").split())


def mark(quote, context):
    """The context with the quote in [brackets] and CONTEXT_WORDS words on
    each side; None if the quote is not found in the context."""
    q, c = clean(quote), clean(context)
    words = re.findall(r"\w+", q.lower())
    if not words or not c:
        return None
    pat = r"\W+".join(re.escape(w) for w in words)
    m = re.search(pat, c, flags=re.I)
    if not m:
        return None
    before, inside, after = c[:m.start()], c[m.start():m.end()], c[m.end():]
    bw, aw = before.split(), after.split()
    # keep the punctuation that touches the phrase
    pre = ("... " if len(bw) > CONTEXT_WORDS else "") + " ".join(bw[-CONTEXT_WORDS:])
    post = " ".join(aw[:CONTEXT_WORDS]) + (" ..." if len(aw) > CONTEXT_WORDS else "")
    sep_b = " " if before.endswith((" ", "\t")) else ""
    sep_a = " " if after.startswith((" ", "\t")) else ""
    return f"{pre}{sep_b}[{inside}]{sep_a}{post}".strip()


# ============================================================
# Main
# ============================================================

def main():
    t0 = time.time()
    os.makedirs(P("outputs"), exist_ok=True)
    print("=" * 70)
    print("v4_stageC_prepare.py")
    print("=" * 70)

    sel = pd.read_csv(os.path.join(STAGEB_DIR, "stageB_clip_selection.csv"),
                      dtype=str).fillna("")
    sel = sel[sel.clip_use.str.lower() == "yes"].reset_index(drop=True)
    axes = list(sel.axis_id)
    fa = pd.read_csv(os.path.join(STAGEB_DIR, "stageB_final_assignment.csv"),
                     dtype=str).fillna("")
    if (fa.how == "undecided").any():
        raise SystemExit("Stage B has undecided texts; finish Stage B first.")
    tx = pd.read_csv(os.path.join(STAGEB_DIR, "stageB_texts.csv"), dtype=str).fillna("")
    pm = pd.read_csv(os.path.join(STAGEB_DIR, "stageB_phrase_map.csv"),
                     dtype=str).fillna("")
    pm = pm[pm.included.str.lower() == "true"]
    pool = pd.read_csv(os.path.join(STAGEV_DIR, "stageA_verified.csv"),
                       dtype=str).fillna("")
    ctx = dict(zip(pool.phrase_id, pool.context))
    print(f"Selected axes: {', '.join(axes)}")

    sub = fa[fa.axis_id.isin(axes)][["text_id", "axis_id"]].merge(
        tx[["text_id", "text", "n_phrases", "documents", "text_class"]], on="text_id")
    rows, not_found = [], 0
    for r in sub.itertuples():
        ph = pm[pm.text_id == r.text_id]
        marked, contexts = [], []
        for p in ph.itertuples():
            c = ctx.get(p.phrase_id, "")
            contexts.append(c)
            m = mark(p.quote, c)
            if m and m not in marked:
                marked.append(m)
        if not marked:
            not_found += 1
        rows.append({"text_id": r.text_id, "axis_id": r.axis_id, "text": r.text,
                     "n_phrases": r.n_phrases, "documents": r.documents,
                     "text_class": r.text_class,
                     "phrase_ids": ";".join(ph.phrase_id),
                     "quotes": " || ".join(dict.fromkeys(clean(q) for q in ph.quote)),
                     "contexts": " || ".join(dict.fromkeys(clean(c) for c in contexts)),
                     "shown": " || ".join(marked[:MAX_CONTEXTS])})
    st = pd.DataFrame(rows)
    rng = np.random.default_rng(RANDOM_SEED)
    st["_r"] = rng.random(len(st))
    st["_a"] = st.axis_id.map({a: i for i, a in enumerate(axes)})
    st = st.sort_values(["_a", "_r"]).drop(columns=["_a", "_r"]).reset_index(drop=True)
    st.to_csv(P("stageC_texts.csv"), index=False)

    # --- messages ---
    defs = dict(zip(sel.axis_id, zip(sel.name_stageC, sel.definition_stageC)))
    nb = int(np.ceil(len(st) / BATCH_SIZE))
    for b in range(nb):
        part = st.iloc[b * BATCH_SIZE:(b + 1) * BATCH_SIZE]
        q = []
        for a in dict.fromkeys(part.axis_id):
            nm, df = defs[a]
            q.append(f"{a} \"{nm}\": {df}")
            if a in EXCLUDED:
                q.append(f"    Excludes: {EXCLUDED[a]}.")
        lines = []
        for r in part.itertuples():
            shown = r.shown.split(" || ") if r.shown else []
            c = " | ".join(f'context: "{s}"' for s in shown) or "context: (not available)"
            lines.append(f'{r.text_id} | {r.axis_id} | phrase: "{clean(r.text)}" | {c}')
        msg = PROMPT_C.format(k=b + 1, n=nb, qualities="\n".join(q)) + "\n".join(lines) + "\n"
        open(P(f"C_batch_{b + 1:02d}.txt"), "w", encoding="utf-8").write(msg)

    # --- report ---
    L = ["STAGE C PREPARATION (v4_protocol.md 1.5, section 6, Stage C)", "=" * 78,
         f"Date: {time.strftime('%Y-%m-%d %H:%M')}", "",
         f"Axes selected for CLIP: {len(axes)}", f"Texts: {len(st):,}", ""]
    vc = st.groupby("axis_id").size()
    cls = pd.crosstab(st.axis_id, st.text_class)
    L.append(f"  {'axis':<6}{'name':<36}{'texts':>6}  by class")
    for a in axes:
        c = cls.loc[a].to_dict() if a in cls.index else {}
        L.append(f"  {a:<6}{defs[a][0][:34]:<36}{int(vc.get(a, 0)):>6}  "
                 + ", ".join(f"{k} {v}" for k, v in c.items() if v))
    L += ["", f"Texts whose phrase was not found in any context: {not_found} "
          "(shown without context; restoring a word is then impossible)",
          f"Messages: {nb} batches of up to {BATCH_SIZE} texts (C_batch_NN.txt).",
          "Paste them in order, in one new chat per model; save the answers as",
          "outputs/stageC_<model>_batchNN.json, then run v4_stageC_check.py."]
    open(P("stageC_prepare_report.txt"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    json.dump({"script": "v4_stageC_prepare.py", "protocol": "1.5",
               "created": time.strftime("%Y-%m-%d %H:%M:%S"), "axes": axes,
               "n_texts": len(st), "n_batches": nb, "batch_size": BATCH_SIZE,
               "seed": RANDOM_SEED, "excluded": EXCLUDED},
              open(P("prepare_meta.json"), "w"), indent=2)
    print(f"Runtime {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
