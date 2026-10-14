"""
v4_stageB_prepare.py  (v4 axes, Stage B: input for the grouping)

Builds the input of Stage B (v4_protocol.md, version 1.3, section 6) from
the verified phrase pool of Stage V, and writes the messages to paste into
the two models' chats (prompts B1 and B2, below).

What it does
------------
1. Reads stageA_verified.csv and checks that Stage V is complete (every
   review row decided).
2. Selects the phrases for grouping:
     state  included if either model labelled it "altered"; phrases that
            only the label "ordinary" describes are set aside for Stage D
            (stageB_ordinary.csv); the rest are excluded and listed.
     sense  excluded if every label given to it is hearing, touch or
            other_sense (no visual or general reading).
3. Gives every included phrase a SENSE CLASS, kept through every later
   stage (v4_protocol.md 6.0):
     vision        every label vision
     not_specific  every label not_sense_specific
     disputed      the two models disagree between vision and
                   not_sense_specific
   (hearing / touch / other_sense labels next to a visual one are flagged
   in also_other_sense.)
4. Identical texts (ignoring case, spacing and surrounding punctuation)
   become one TEXT to group, with its phrases, documents, first-person
   phrases and sense classes counted. Grouping depends only on what the
   words mean; support is counted later from the phrases (Stage B
   comparison).
5. The texts are shuffled (fixed seed: the same order for both models),
   numbered T0001..., and written as:
     B1_message.txt          prompt B1 + the full list (paste, or attach
                             the list as a file if it is too long)
     B2_batch_NN.txt         prompt B2 + one batch of BATCH_SIZE texts
   Models see only the numbered texts: no documents, speakers, labels or
   counts.

Outputs (OUTPUT_DIR)
--------------------
  stageB_texts.csv        one row per text: text_id, text, counts by
                          sense class, documents, first-person phrases
  stageB_phrase_map.csv   every phrase of the pool: its text_id (or why it
                          was left out), sense class, first person, labels
  stageB_ordinary.csv     the "ordinary" phrases, for Stage D
  stageB_excluded.csv     phrases left out, with the reason
  B1_message.txt, B2_batch_NN.txt   the messages to paste
  stageB_prepare_report.txt, run_meta.json

How to run Stage B (each model in a new chat of the memory-free account)
------------------------------------------------------------------------
  1. Paste B1_message.txt. Save the JSON as
       stageB_<model>_B1.json            (model = claude or chatgpt)
  2. In the SAME chat, paste B2_batch_01.txt, then 02, ... Save each JSON
     as stageB_<model>_B2_batch01.json, ..._batch02.json, ...
  3. Put the files in OUTPUT_DIR/outputs/ for the comparison script.

Usage
-----
    conda activate drl2drv
    python v4_stageB_prepare.py
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

STAGEV_DIR  = "axes/v4/stageV"
OUTPUT_DIR  = "axes/v4/stageB"
MODELS      = ["claude", "chatgpt"]
BATCH_SIZE  = 150
RANDOM_SEED = 42
REQUIRE_STAGEV_COMPLETE = True

VISUAL_SENSES = {"vision", "not_sense_specific"}
OTHER_SENSES  = {"hearing", "touch", "other_sense"}


# ============================================================
# Prompts (v4_protocol.md 1.3, section 6; no derealization examples)
# ============================================================

PROMPT_B1 = """I have a list of short phrases, numbered T0001, T0002, and so on. Each
phrase was taken word for word from a published text, and most of them
describe how the world, places, objects or other people appear or feel to
someone. Some may not.

TASK
Read the whole list. Then propose groups of phrases that describe the same
quality: the same way in which the world, places, objects or other people
appear or feel.

RULES
- Each group names ONE quality. If phrases that seem related actually
  describe two different qualities, make two groups.
- Base the groups only on what the phrases say. Do not add qualities that
  no phrase describes, and do not use knowledge from outside the list.
- Make the groups as distinct from each other as you can.
- A group must be supported by several phrases in the list. Do not make a
  group for a single phrase.
- Do not place every phrase now; that comes later. Phrases that do not
  describe how anything appears or feels need no group.
- Give each group a short name (2 to 5 words), using words that occur in
  its phrases where possible, and a one-sentence definition of the quality.

ILLUSTRATION (an unrelated topic; NOT from the list)
For phrases about how food tastes, such as "thin and metallic", "like water
from a tin cup", "watery", "a sharp tang", "like licking a coin", the groups
might be:
  G01 "watery, thin": the taste seems diluted or lacking in substance.
  G02 "metallic": the taste resembles metal.

OUTPUT
One JSON array inside one code block labelled json:
[
  {"group": "G01", "name": "...", "definition": "...",
   "examples": ["T0012", "T0345", "T0789"]},
  ...
]
"examples": 3 to 5 phrase numbers from the list that clearly belong to the
group. Number the groups G01, G02, ... After the code block write the
number of groups and then the line: END OF GROUPS

THE LIST
"""

PROMPT_B2 = """Now assign phrases to the groups you proposed. Below is batch {k} of {n}
from the same list.

For each phrase choose exactly one:
- the group (G01, G02, ...) whose quality the phrase describes. If it fits
  more than one group, choose the best fit and give the second in "also";
- "OTHER": the phrase describes how the world, places, objects or other
  people appear or feel, but fits none of your groups;
- "NOT_WORLD": the phrase does not describe how the world, places, objects
  or other people appear or feel; for example, it describes only the
  person's own body, self, thoughts, emotions or actions, or it is a
  statement about causes, treatment or theory.

Do not change, add or remove groups. Assign every phrase in this batch
exactly once.

OUTPUT
One JSON array inside one code block labelled json:
[{{"id": "T0001", "group": "G03", "also": null}}, ...]
After the code block write the number of phrases assigned and then the
line: END OF BATCH {k}

BATCH {k} OF {n}
"""


# ============================================================
# Helpers
# ============================================================

def P(*n):
    return os.path.join(OUTPUT_DIR, *n)


def labels(row, field):
    out = set()
    for m in MODELS:
        v = row.get(f"{field}_{m}")
        if isinstance(v, str) and v:
            out |= set(v.split("|"))
    return out


def text_key(t):
    """Identical texts: case, white space and surrounding punctuation or
    quotation marks ignored."""
    t = unicodedata.normalize("NFKC", str(t)).casefold()
    t = re.sub(r"\s+", " ", t).strip()
    return re.sub(r"^[\W_]+|[\W_]+$", "", t)


# ============================================================
# Main
# ============================================================

def main():
    t0 = time.time()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=" * 70)
    print("v4_stageB_prepare.py")
    print("=" * 70)

    pool_path = os.path.join(STAGEV_DIR, "stageA_verified.csv")
    pool = pd.read_csv(pool_path, dtype=str).fillna("")
    meta_path = os.path.join(STAGEV_DIR, "run_meta.json")
    if os.path.exists(meta_path):
        meta = json.load(open(meta_path))
        open_rows = meta.get("review", {}).get("open", 0)
        if open_rows and REQUIRE_STAGEV_COMPLETE:
            raise SystemExit(f"Stage V has {open_rows} review rows open; finish "
                             "Stage V first.")
    if (pool["first_person"] == "undecided").any() and REQUIRE_STAGEV_COMPLETE:
        raise SystemExit("Some phrases have first_person = undecided; finish "
                         "Stage V first.")
    print(f"Pool: {len(pool):,} phrases")

    # --- state and sense ---
    rows, excluded, ordinary = [], [], []
    for r in pool.to_dict("records"):
        st, se = labels(r, "state"), labels(r, "sense")
        base = {"phrase_id": r["phrase_id"], "doc_id": r["doc_id"],
                "quote": r["quote"], "state": "|".join(sorted(st)),
                "sense": "|".join(sorted(se)), "first_person": r["first_person"],
                "speaker": "|".join(sorted(labels(r, "speaker"))),
                "target": "|".join(sorted(labels(r, "target")))}
        if "altered" not in st:
            if "ordinary" in st:
                ordinary.append({**base, "context": r["context"]})
            else:
                excluded.append({**base, "reason": "state not altered"})
            continue
        vis = se & VISUAL_SENSES
        if not vis:
            excluded.append({**base, "reason": "non-visual sense only"})
            continue
        cls = ("vision" if vis == {"vision"} else
               "not_specific" if vis == {"not_sense_specific"} else "disputed")
        rows.append({**base, "sense_class": cls,
                     "also_other_sense": bool(se & OTHER_SENSES),
                     "tkey": text_key(r["quote"])})
    inc = pd.DataFrame(rows)
    print(f"Included {len(inc):,}; ordinary (for Stage D) {len(ordinary):,}; "
          f"excluded {len(excluded):,}")

    # --- unique texts ---
    texts = []
    for k, g in inc.groupby("tkey", sort=False):
        rep = g["quote"].value_counts().index[0]
        cls = g["sense_class"].value_counts()
        texts.append({"tkey": k, "text": rep, "n_phrases": len(g),
                      "n_documents": g["doc_id"].nunique(),
                      "documents": ";".join(sorted(g["doc_id"].unique())),
                      "n_first_person": int((g["first_person"] == "yes").sum()),
                      "n_vision": int(cls.get("vision", 0)),
                      "n_not_specific": int(cls.get("not_specific", 0)),
                      "n_disputed": int(cls.get("disputed", 0))})
    tx = pd.DataFrame(texts)
    order = np.random.default_rng(RANDOM_SEED).permutation(len(tx))
    tx = tx.iloc[order].reset_index(drop=True)
    tx.insert(0, "text_id", [f"T{i + 1:04d}" for i in range(len(tx))])
    tx["text_class"] = np.select(
        [(tx.n_vision > 0) & (tx.n_not_specific == 0) & (tx.n_disputed == 0),
         (tx.n_not_specific > 0) & (tx.n_vision == 0) & (tx.n_disputed == 0)],
        ["vision", "not_specific"], "mixed")
    tid = dict(zip(tx.tkey, tx.text_id))
    inc["text_id"] = inc["tkey"].map(tid)

    # --- outputs ---
    tx.drop(columns="tkey").to_csv(P("stageB_texts.csv"), index=False)
    pm = pd.concat([inc.drop(columns="tkey").assign(included=True),
                    pd.DataFrame(excluded).assign(included=False),
                    pd.DataFrame(ordinary).drop(columns="context", errors="ignore")
                    .assign(included=False, reason="ordinary (Stage D)")],
                   ignore_index=True)
    pm.to_csv(P("stageB_phrase_map.csv"), index=False)
    pd.DataFrame(ordinary).to_csv(P("stageB_ordinary.csv"), index=False)
    pd.DataFrame(excluded).to_csv(P("stageB_excluded.csv"), index=False)

    def clean(t):
        return " ".join(str(t).split())

    lines = [f"{a} | {clean(b)}" for a, b in zip(tx.text_id, tx.text)]
    open(P("B1_message.txt"), "w").write(PROMPT_B1 + "\n".join(lines) + "\n")
    open(P("B1_list_only.txt"), "w").write("\n".join(lines) + "\n")
    nb = int(np.ceil(len(lines) / BATCH_SIZE))
    for b in range(nb):
        part = lines[b * BATCH_SIZE:(b + 1) * BATCH_SIZE]
        open(P(f"B2_batch_{b + 1:02d}.txt"), "w").write(
            PROMPT_B2.format(k=b + 1, n=nb) + "\n".join(part) + "\n")
    os.makedirs(P("outputs"), exist_ok=True)

    # --- report ---
    words = sum(len(t.split()) for t in tx.text) + 2 * len(tx)
    L = ["STAGE B PREPARATION (v4_protocol.md 1.3, section 6)", "=" * 78,
         f"Date: {time.strftime('%Y-%m-%d %H:%M')}", "",
         f"Pool: {len(pool):,} phrases",
         f"  included for grouping: {len(inc):,}",
         f"  'ordinary', set aside for Stage D: {len(ordinary):,}",
         f"  excluded: {len(excluded):,} ("
         + ", ".join(f"{k} {v}" for k, v in pd.Series([e['reason'] for e in excluded])
                     .value_counts().items()) + ")" if excluded else "  excluded: 0",
         "",
         "Sense class of the included phrases:",
         *[f"  {k:<14}{v:>6}  ({100 * v / len(inc):.0f}%)"
           for k, v in inc.sense_class.value_counts().items()],
         f"  with a hearing / touch / other label as well: "
         f"{int(inc.also_other_sense.sum())}",
         f"First person: {int((inc.first_person == 'yes').sum()):,} of {len(inc):,}",
         "",
         f"Texts to group (identical phrases merged): {len(tx):,}",
         "  by class: " + ", ".join(f"{k} {v}" for k, v in
                                    tx.text_class.value_counts().items()),
         f"  texts occurring in more than one phrase: {int((tx.n_phrases > 1).sum())}"
         f" (largest: {tx.loc[tx.n_phrases.idxmax(), 'text']!r}, "
         f"{int(tx.n_phrases.max())} phrases)",
         "",
         f"Messages: B1_message.txt (about {words:,} words of list; if the chat "
         "refuses it, paste the prompt part and attach B1_list_only.txt) and "
         f"{nb} B2 batches of up to {BATCH_SIZE} texts.",
         "Save the outputs in outputs/ as stageB_<model>_B1.json and "
         "stageB_<model>_B2_batchNN.json."]
    open(P("stageB_prepare_report.txt"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    json.dump({"script": "v4_stageB_prepare.py", "protocol": "1.3",
               "created": time.strftime("%Y-%m-%d %H:%M:%S"),
               "n_pool": int(len(pool)), "n_included": int(len(inc)),
               "n_ordinary": len(ordinary), "n_excluded": len(excluded),
               "n_texts": int(len(tx)), "n_batches": nb,
               "batch_size": BATCH_SIZE, "seed": RANDOM_SEED,
               "runtime_s": round(time.time() - t0, 1)},
              open(P("run_meta.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
