"""
v4_stageD_prepare.py  (v4 axes, Stage D: input for the counterfactuals)

Builds the input of Stage D (v4_protocol.md, version 1.8, section 6, Stage
D) and writes the messages to paste into the two models' chats.

What it does
------------
1. Reads the final captions of Stage C (stageC_captions.csv) and the
   selection for CLIP, and stops if Stage C has open texts.
2. Keeps the axes selected for CLIP minus DROPPED_AXES (A15, protocol 6.3).
3. One ANCHOR per distinct caption of an axis (decision D1): identical
   captions from different texts become one anchor, with all their texts,
   phrases, documents and sense classes kept. Anchor ids: <axis>-NN.
4. Flags the anchors whose quality is carried by a negation word (decision
   D9: tested in CLIP after Stage D).
5. Writes batches of BATCH_SIZE anchors, ordered by axis, each message
   complete on its own (instructions, the qualities in the batch, the
   captions). All batches go to the same chat, one chat per model.

Models see only the captions and their axis's name and definition: no
phrases, documents, contexts or sense classes.

Outputs (OUTPUT_DIR)
--------------------
  stageD_anchors.csv        one row per anchor: axis, caption, texts,
                            phrases, documents, sense classes, negation flag
  D_batch_NN.txt            the messages to paste
  outputs/                  put the model outputs here
  stageD_prepare_report.txt, prepare_meta.json

How to run Stage D (each model in a new chat of the memory-free account)
------------------------------------------------------------------------
  In one new chat per model, paste D_batch_01.txt, then 02, ... in the same
  chat. Save each JSON answer as
      outputs/stageD_<model>_batchNN.json      (model = claude or chatgpt)
  Then run v4_stageD_check.py.

Usage
-----
    conda activate drl2drv
    python v4_stageD_prepare.py
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

STAGEB_DIR   = "axes/v4/stageB"
STAGEC_DIR   = "axes/v4/stageC"
OUTPUT_DIR   = "axes/v4/stageD"
DROPPED_AXES = ["A15"]          # dropped after Stage C (protocol 6.3)
BATCH_SIZE   = 80
TEMPLATE     = "a scene that looks {}"
NEGATION = {"not", "no", "never", "nothing", "none", "nobody", "nowhere", "neither",
            "nor", "without", "cannot", "can't", "don't", "doesn't", "didn't",
            "isn't", "aren't", "wasn't", "weren't", "won't", "couldn't"}


# ============================================================
# Prompt (v4_protocol.md 1.8, Stage D; no derealization examples)
# ============================================================

PROMPT_D = """Below is a list of short CAPTIONS. Each completes the sentence

    a scene that looks ___

and describes one QUALITY: a way in which the world, places or objects can
appear when perception is altered. The qualities are defined below.

TASK
For each caption, write 3 different COUNTERPARTS: captions that describe the
same thing as it appears in ordinary perception, without the altered
quality. Each counterpart must also complete "a scene that looks ___".

RULES
1. Change only the word or words that carry the quality. Keep every other
   word, in the same order. Change at most two words; small words (a, the,
   of, as, if, like, is, ...) do not count.
2. Keep the same form: if the caption begins with "like" or "as if", so does
   the counterpart; a single adjective is replaced by a single adjective.
3. Describe ordinary perception, not the opposite extreme. If the quality is
   a loss, the counterpart is the usual amount, not an excess; if it is an
   excess, the counterpart is the usual amount, not a lack.
4. No negation. Do not use not, no, never, nothing, none, without, or any
   word ending in n't, and do not add un-, non- or in- to a word of the
   caption. (If the caption itself contains a negation, replace that word
   by its positive counterpart.)
5. Be specific. The counterpart names the ordinary counterpart of THIS
   caption's quality. Do not use words that could replace any quality, such
   as normal, ordinary, usual, regular, everyday, fine or natural.
6. The three counterparts must differ from each other. Put the one you find
   best first.

ILLUSTRATION (an unrelated topic, taste, with the sentence "a drink that
tastes ___"; NOT from the list)
- "thin and metallic"  ->  "full and fresh", "smooth and clean",
  "rich and fresh"
- "like water from a tin cup"  ->  "like water from a glass",
  "like water from a jug", "like water from a clean cup"
- Not acceptable: "not metallic" (negation), "delicious" (a general word,
  not the counterpart of this quality), "like the best drink in the world"
  (the opposite extreme).

OUTPUT
One JSON array inside one code block labelled json, one object per caption,
in the order of the list:
[{{"id": "A01-01", "counterparts": ["...", "...", "..."]}}, ...]
After the code block write the number of captions in the array and then the
line: END OF BATCH {k}

QUALITIES
{qualities}

THE CAPTIONS (batch {k} of {n})
"""


# ============================================================
# Helpers
# ============================================================

def P(*n):
    return os.path.join(OUTPUT_DIR, *n)


def clean(t):
    t = unicodedata.normalize("NFKC", str(t))
    return " ".join(t.replace("’", "'").replace("‘", "'").split())


def words(t):
    return re.findall(r"[a-z0-9]+(?:'[a-z]+)?", clean(t).lower())


# ============================================================
# Main
# ============================================================

def main():
    t0 = time.time()
    os.makedirs(P("outputs"), exist_ok=True)
    print("=" * 70)
    print("v4_stageD_prepare.py")
    print("=" * 70)

    caps = pd.read_csv(os.path.join(STAGEC_DIR, "stageC_captions.csv"), dtype=str).fillna("")
    if (caps.how == "open").any():
        raise SystemExit(f"Stage C has {int((caps.how == 'open').sum())} open texts; "
                         "finish Stage C first.")
    sel = pd.read_csv(os.path.join(STAGEB_DIR, "stageB_clip_selection.csv"),
                      dtype=str).fillna("")
    sel = sel[(sel.clip_use.str.lower() == "yes") & ~sel.axis_id.isin(DROPPED_AXES)]
    axes = list(sel.axis_id)
    defs = dict(zip(sel.axis_id, zip(sel.name_stageC, sel.definition_stageC)))
    print(f"Axes: {', '.join(axes)} (dropped: {', '.join(DROPPED_AXES) or 'none'})")

    c = caps[(caps.caption != "") & caps.axis_id.isin(axes)].copy()
    c["key"] = c.caption.map(lambda x: " ".join(words(x)))
    rows = []
    for a in axes:
        g = c[c.axis_id == a]
        groups = sorted(g.groupby("key"), key=lambda kv: (-len(kv[1]), kv[0]))
        for i, (k, gg) in enumerate(groups):
            # the caption as written most often (case, punctuation)
            cap = clean(gg.caption.value_counts().index[0])
            neg = sorted({w for w in words(cap) if w in NEGATION})
            rows.append({
                "anchor_id": f"{a}-{i + 1:02d}", "axis_id": a, "caption": cap,
                "sentence": TEMPLATE.format(cap),
                "n_texts": len(gg),
                "n_phrases": int(sum(int(x or 1) for x in gg.n_phrases)),
                "text_ids": ";".join(gg.text_id),
                "phrase_ids": ";".join(p for x in gg.phrase_ids for p in x.split(";") if p),
                "documents": ";".join(sorted({d for x in gg.documents
                                              for d in x.split(";") if d})),
                "text_classes": ";".join(sorted(set(gg.text_class))),
                "negation_in_positive": " ".join(neg),
                "quotes": " || ".join(dict.fromkeys(gg.quote))})
    an = pd.DataFrame(rows)
    an.to_csv(P("stageD_anchors.csv"), index=False)

    nb = int(np.ceil(len(an) / BATCH_SIZE))
    for b in range(nb):
        part = an.iloc[b * BATCH_SIZE:(b + 1) * BATCH_SIZE]
        q = [f'{a} "{defs[a][0]}": {defs[a][1]}' for a in dict.fromkeys(part.axis_id)]
        lines = [f'{r.anchor_id} | {r.axis_id} | caption: "{r.caption}"'
                 for r in part.itertuples()]
        msg = PROMPT_D.format(k=b + 1, n=nb, qualities="\n".join(q)) + "\n".join(lines) + "\n"
        open(P(f"D_batch_{b + 1:02d}.txt"), "w", encoding="utf-8").write(msg)

    L = ["STAGE D PREPARATION (v4_protocol.md 1.8, section 6, Stage D)", "=" * 78,
         f"Date: {time.strftime('%Y-%m-%d %H:%M')}", "",
         f"Axes: {len(axes)} ({', '.join(axes)}); dropped: {', '.join(DROPPED_AXES)}",
         f"Captioned texts: {len(c)}; anchors (distinct captions): {len(an)}", "",
         f"  {'axis':<6}{'name':<36}{'texts':>6}{'anchors':>8}{'negation':>9}"]
    for a in axes:
        g = an[an.axis_id == a]
        L.append(f"  {a:<6}{defs[a][0][:34]:<36}{int(g.n_texts.sum()):>6}{len(g):>8}"
                 f"{int((g.negation_in_positive != '').sum()):>9}")
    L += ["", f"Anchors whose caption contains a negation word (decision D9): "
          f"{int((an.negation_in_positive != '').sum())}; tested in CLIP after Stage D.",
          f"Messages: {nb} batches of up to {BATCH_SIZE} captions (D_batch_NN.txt).",
          "Paste them in order, in one new chat per model; save the answers as",
          "outputs/stageD_<model>_batchNN.json, then run v4_stageD_check.py."]
    open(P("stageD_prepare_report.txt"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    json.dump({"script": "v4_stageD_prepare.py", "protocol": "1.8",
               "created": time.strftime("%Y-%m-%d %H:%M:%S"), "axes": axes,
               "dropped_axes": DROPPED_AXES, "n_anchors": len(an), "n_batches": nb,
               "batch_size": BATCH_SIZE, "template": TEMPLATE},
              open(P("prepare_meta.json"), "w"), indent=2)
    print(f"Runtime {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
