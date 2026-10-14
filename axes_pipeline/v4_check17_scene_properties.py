"""
v4_check17_scene_properties.py  (v4 axes: decision 17, text only, no images)

Tests whether the negatives of some v4 axes bring a scene property into the
axis (v4_protocol.md 1.11, section 8, decision 17). Run after Stage D and
before any image is scored.

Why
---
Each negative of Stage D was checked one by one, but read as a whole:
  A05 (veil, glass wall, barrier, bubble): 39 of 61 negatives add "open",
      "air" or "space" ("as if in a bubble -> as if in open space");
  A07 (fog, blurry, hazy): 7 of 30 add "daylight" or "lit";
  A14 (lifeless, dead, empty): 1 adds "light".
The axis could then partly measure "enclosed versus open" (close to indoor
versus outdoor in Places365) or "dark versus bright".

What it does (only CLIP text embeddings)
----------------------------------------
1. Reads axes_v4.yaml and encodes every positive and negative in each
   template, exactly as step 05 (encode_hf: padding to 77 tokens, pooled
   output of the text tower, text_projection, L2 norm).
2. Builds every axis as in decision 20: the mean of the raw differences
   (positive - negative) of the unit-length sentence vectors, scaled to
   unit length.
3. Builds two CONTROL directions the same way, from fixed sentence pairs.
   Each control points towards the pole the suspect negatives would push
   the axis to (the altered pole of an axis is its positive), so a
   positive cosine means the axis leans towards the control's first pole:
      enclosure = enclosed - open      darkness = dark - bright
4. Reports, for every axis and template, the cosine with each control (the
   other axes are the reference). For A05, A07 and A14 it also builds:
      reduced  the axis without its suspect pairs
      suspect  the direction of the suspect pairs only
   and reports their cosines with the controls and the cosine between the
   full and the reduced axis.
5. Applies the rule of decision 17 in the primary template, with the
   control that matches the suspect words (TARGET_CONTROL):
      revise when  cos_full - cos_reduced >= max(RULE_ABS, RULE_REL * cos_full)
      (only when cos_full > 0)
6. Information only (not part of the rule): how large the drop is compared
   with removing the same number of pairs at random (N_RANDOM draws).

Suspect pairs, by rule: the negative adds a suspect word that the positive
does not contain.

Outputs (OUTPUT_DIR)
--------------------
  check17_axes.csv       one row per template x axis: cosines with the
                         controls; for suspect axes the reduced and suspect
                         directions, the drop, the rule, the random baseline
  check17_pairs.csv      one row per template x pair: the pair's cosine, the
                         length of its difference, its share of the axis, the
                         cosine of its difference with each control, suspect
  check17_controls.csv   the control sentence pairs and how well each pair
                         agrees with its control direction
  check17_report.txt     the summary
  check17_cache_clip.npz cached sentence vectors
  check17_meta.json      settings

Usage
-----
    conda activate drl2drv
    python v4_check17_scene_properties.py
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

AXES_YAML  = "axes/v4/stageD2/axes_v4.yaml" # "axes/v4/stageD/axes_v4.yaml"
OUTPUT_DIR = "axes/v4/check17_D2" # "axes/v4/check17"
CLIP_MODEL = "openai/clip-vit-large-patch14-336"     # as in step 05
BATCH_SIZE = 64
DEVICE     = None             # None: cuda if available, else mps, else cpu
ADD_PERIOD = False            # step 05 encodes the template without a period
USE_ROBUSTNESS_TEMPLATES = True   # also report the 3 robustness templates

# Control directions (decision 17): (pole the suspect words would push the
# axis towards, the other pole). Fixed before the test was run.
CONTROLS = {
    "enclosure": [("enclosed", "open"),
                  ("cramped", "spacious"),
                  ("like a closed room", "like a wide open landscape"),
                  ("like an indoor room", "like the outdoors")],
    "darkness":  [("dark", "bright"),
                  ("dimly lit", "brightly lit"),
                  ("like nighttime", "like daytime"),
                  ("shadowy", "sunlit")],
}

# Suspect words added by the negative, per axis, and the control they match
SUSPECT = {"A05": {"open", "air", "space"},
           "A07": {"daylight", "lit"},
           "A14": {"light"}}
TARGET_CONTROL = {"A05": "enclosure", "A07": "darkness", "A14": "darkness"}

# After the revision (Stage D2): the axes file before it. When set, the
# "suspect" pairs of the SUSPECT axes are the pairs whose negative was
# replaced (the words rule is not used), and every axis is compared with
# its version before the revision. None: the words rule (first run).
REVISED_FROM = "axes/v4/stageD/axes_v4.yaml"           # e.g. "axes/v4/stageD/axes_v4.yaml"

RULE_REL = 1 / 3              # the drop must be at least a third of cos_full
RULE_ABS = 0.05               # and at least 0.05
N_RANDOM = 2000               # random removals for the baseline (information)
SEED     = 17


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


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def cos(a, b):
    return float(unit(a) @ unit(b))


def pick_device():
    import torch
    if DEVICE:
        return DEVICE
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def encode_clip(texts):
    """CLIP text embeddings exactly as step 05 (encode_hf)."""
    import torch
    from transformers import CLIPModel, CLIPTokenizer
    dev = pick_device()
    tok = CLIPTokenizer.from_pretrained(CLIP_MODEL)
    m = CLIPModel.from_pretrained(CLIP_MODEL, use_safetensors=True)
    m.vision_model = None             # text tower only
    m.visual_projection = None
    m = m.to(dev).eval()
    out = []
    with torch.inference_mode():
        for i in range(0, len(texts), BATCH_SIZE):
            b = tok(texts[i:i + BATCH_SIZE], padding="max_length", max_length=77,
                    truncation=True, return_tensors="pt").to(dev)
            pooled = m.text_model(input_ids=b["input_ids"],
                                  attention_mask=b["attention_mask"]).pooler_output
            e = m.text_projection(pooled)
            e = e / e.norm(dim=-1, keepdim=True).clamp(min=1e-12)
            out.append(e.float().cpu().numpy())
    return np.concatenate(out).astype(np.float32)


def vectors(texts, encode=encode_clip):
    """Cached on disk; returns {sentence: unit vector}."""
    path = P("check17_cache_clip.npz")
    have = {}
    if os.path.exists(path):
        z = np.load(path, allow_pickle=False)
        have = dict(zip(z["keys"].tolist(), z["vecs"]))
    todo = [t for t in dict.fromkeys(texts) if t not in have]
    if todo:
        print(f"  encoding {len(todo)} sentences with {CLIP_MODEL} ...")
        for t, x in zip(todo, encode(todo)):
            have[t] = x
        ks = list(have)
        np.savez(path, keys=np.array(ks), vecs=np.stack([have[k] for k in ks]))
    return have


def direction(diffs):
    """Decision 20: mean of the raw differences, scaled to unit length."""
    return unit(np.mean(diffs, axis=0))


# ============================================================
# Main
# ============================================================

def main(encode=encode_clip):
    t0 = time.time()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=" * 70)
    print("v4_check17_scene_properties.py  (decision 17, text only)")
    print("=" * 70)

    import yaml
    doc = yaml.safe_load(open(AXES_YAML, encoding="utf-8"))
    templates = [doc["template"]] + (list(doc.get("templates_robustness", []))
                                     if USE_ROBUSTNESS_TEMPLATES else [])
    primary = templates[0]
    S = lambda tpl, c: tpl.format(clean(c)) + ("." if ADD_PERIOD else "")

    # pairs
    before = None
    if REVISED_FROM:
        before = yaml.safe_load(open(REVISED_FROM, encoding="utf-8"))
        old_neg = {p["anchor_id"]: clean(p["negative"]) for ax in before["axes"]
                   for p in ax["pairs"]}
    rows = []
    for ax in doc["axes"]:
        for p in ax["pairs"]:
            added = set(words(p["negative"])) - set(words(p["positive"]))
            if before is not None:
                was = old_neg.get(p["anchor_id"])
                rev = ax["id"] in SUSPECT and was is not None and was != clean(p["negative"])
                hit = ["revised"] if rev else []
            else:
                hit = sorted(added & SUSPECT.get(ax["id"], set()))
            rows.append({"axis_id": ax["id"], "axis_name": ax["name"],
                         "anchor_id": p["anchor_id"], "positive": clean(p["positive"]),
                         "negative": clean(p["negative"]),
                         "suspect": bool(hit), "suspect_words": " ".join(hit)})
    PR = pd.DataFrame(rows)
    axes = list(dict.fromkeys(PR.axis_id))
    names = dict(zip(PR.axis_id, PR.axis_name))
    print(f"Axes: {len(axes)}; pairs: {len(PR)}; templates: {len(templates)}")
    for a in SUSPECT:
        g = PR[PR.axis_id == a]
        rule = "replaced in the revision" if before is not None else ", ".join(sorted(SUSPECT[a]))
        print(f"  {a}: {int(g.suspect.sum())} of {len(g)} pairs suspect ({rule})")
    old_pairs = []
    if before is not None:
        old_pairs = [(ax["id"], clean(p["positive"]), clean(p["negative"]))
                     for ax in before["axes"] for p in ax["pairs"]]

    sents = [S(t, c) for t in templates
             for c in list(PR.positive) + list(PR.negative)
             + [x for prs in CONTROLS.values() for pr in prs for x in pr]
             + [x for _, a_, b_ in old_pairs for x in (a_, b_)]]
    V = vectors(sents, encode)

    rng = np.random.default_rng(SEED)
    ax_rows, pair_rows, ctl_rows = [], [], []
    for tpl in templates:
        # control directions
        ctl = {}
        for cname, prs in CONTROLS.items():
            d = np.stack([V[S(tpl, a)] - V[S(tpl, b)] for a, b in prs])
            ctl[cname] = direction(d)
            for (a, b), x in zip(prs, d):
                ctl_rows.append({"template": tpl, "control": cname, "pole": a,
                                 "other_pole": b, "pair_cosine": float(V[S(tpl, a)] @ V[S(tpl, b)]),
                                 "cos_with_control": cos(x, ctl[cname])})
        # axes
        for a in axes:
            g = PR[PR.axis_id == a]
            D = np.stack([V[S(tpl, r.positive)] - V[S(tpl, r.negative)]
                          for r in g.itertuples()])
            mean_d = D.mean(axis=0)
            full = unit(mean_d)
            # each pair's share of the axis: its projection on the axis over the sum
            proj = D @ full
            for r, x, pj in zip(g.itertuples(), D, proj):
                pr = {"template": tpl, "axis_id": a, "anchor_id": r.anchor_id,
                      "positive": r.positive, "negative": r.negative,
                      "suspect": r.suspect, "suspect_words": r.suspect_words,
                      "pair_cosine": float(V[S(tpl, r.positive)] @ V[S(tpl, r.negative)]),
                      "diff_length": float(np.linalg.norm(x)),
                      "share_of_axis": float(pj / proj.sum())}
                for cname, cv in ctl.items():
                    pr[f"cos_diff_{cname}"] = cos(x, cv)
                pair_rows.append(pr)
            row = {"template": tpl, "axis_id": a, "axis_name": names[a], "n_pairs": len(g)}
            for cname, cv in ctl.items():
                row[f"cos_{cname}"] = cos(full, cv)
            if old_pairs:
                Do = np.stack([V[S(tpl, p_)] - V[S(tpl, n_)] for ax_, p_, n_ in old_pairs
                               if ax_ == a])
                ob = direction(Do)
                row["n_pairs_before"] = len(Do)
                row["cos_with_before"] = cos(full, ob)
                for cname, cv in ctl.items():
                    row[f"before_cos_{cname}"] = cos(ob, cv)
            if a in SUSPECT:
                m = g.suspect.to_numpy()
                k = int(m.sum())
                tc = TARGET_CONTROL[a]
                row.update({"n_suspect": k, "target_control": tc,
                            "suspect_share_of_axis": float(proj[m].sum() / proj.sum())})
                if 0 < k < len(g):
                    red, sus = direction(D[~m]), direction(D[m])
                    row["cos_full_reduced"] = cos(full, red)
                    row["cos_suspect_rest"] = cos(sus, red)
                    for cname, cv in ctl.items():
                        row[f"reduced_cos_{cname}"] = cos(red, cv)
                        row[f"suspect_cos_{cname}"] = cos(sus, cv)
                    cf, cr = row[f"cos_{tc}"], row[f"reduced_cos_{tc}"]
                    drop = cf - cr
                    need = max(RULE_ABS, RULE_REL * cf)
                    row.update({"drop_value": drop, "drop_needed": need,
                                "revise": bool(cf > 0 and drop >= need)})
                    # random removals of the same number of pairs (information)
                    idx = np.arange(len(g))
                    rd = np.empty(N_RANDOM)
                    for i in range(N_RANDOM):
                        keep = np.ones(len(g), bool)
                        keep[rng.choice(idx, size=k, replace=False)] = False
                        rd[i] = cf - cos(direction(D[keep]), ctl[tc])
                    row["random_drop_median"] = float(np.median(rd))
                    row["random_drop_p95"] = float(np.quantile(rd, 0.95))
                    row["drop_percentile_vs_random"] = float((rd < drop).mean() * 100)
            ax_rows.append(row)

    AX = pd.DataFrame(ax_rows)
    PRS = pd.DataFrame(pair_rows)
    CT = pd.DataFrame(ctl_rows)
    AX.to_csv(P("check17_axes.csv"), index=False, float_format="%.4f")
    PRS.to_csv(P("check17_pairs.csv"), index=False, float_format="%.4f")
    CT.to_csv(P("check17_controls.csv"), index=False, float_format="%.4f")

    # ---------------- report ----------------
    cn = list(CONTROLS)
    L = ["CHECK 17: SCENE PROPERTIES IN THE NEGATIVES (v4_protocol.md 1.11, decision 17)",
         "=" * 78, f"Date: {time.strftime('%Y-%m-%d %H:%M')}",
         f"Axes file: {AXES_YAML} (protocol {doc.get('protocol')})",
         f"CLIP: {CLIP_MODEL}; text only, no images; period added: {ADD_PERIOD}",
         "Axis = mean of raw paired differences (positive - negative), unit length (decision 20).",
         "Controls point to the pole the suspect negatives would push the axis to:",
         "  " + "; ".join(f"{c} = {' / '.join(p[0] for p in CONTROLS[c])}  minus  "
                          f"{' / '.join(p[1] for p in CONTROLS[c])}" for c in cn),
         "A positive cosine: the axis leans towards enclosed (enclosure) or dark (darkness).",
         ""]

    L += ["1. CONTROLS (primary template): how well each sentence pair agrees with its control",
          "-" * 78]
    for r in CT[CT.template == primary].itertuples():
        L.append(f"  {r.control:<10} {r.pole:>22} / {r.other_pole:<28} "
                 f"cos with control {r.cos_with_control:6.3f}  (pair cosine {r.pair_cosine:.3f})")
    L.append("")

    L += ["2. EVERY AXIS AGAINST THE CONTROLS (cosine; the other axes are the reference)",
          "-" * 78,
          f"  {'axis':<6}{'name':<36}" + "".join(f"{c[:6] + ' t' + str(i + 1):>11}"
                                                 for c in cn for i in range(len(templates)))]
    for a in axes:
        vals = []
        for c in cn:
            for t in templates:
                v = AX[(AX.template == t) & (AX.axis_id == a)][f"cos_{c}"].iloc[0]
                vals.append(f"{v:11.3f}")
        L.append(f"  {a:<6}{names[a][:34]:<36}" + "".join(vals))
    L.append("  templates: " + "; ".join(f"t{i + 1} = \"{t}\"" for i, t in enumerate(templates)))
    for c in cn:
        x = AX[AX.template == primary][f"cos_{c}"]
        L.append(f"  {c}, primary template: median over the 9 axes {x.median():.3f}, "
                 f"range {x.min():.3f} to {x.max():.3f}")
    L.append("")

    if old_pairs:
        L += [f"2b. EACH AXIS AFTER THE REVISION AGAINST ITS VERSION BEFORE ({REVISED_FROM}; "
              "primary template)", "-" * 78,
              f"  {'axis':<6}{'name':<36}{'pairs':>6}{'before':>7}{'cos new/old':>12}"
              + "".join(f"{c[:6] + ' old':>11}{c[:6] + ' new':>11}" for c in cn)]
        for r in AX[AX.template == primary].itertuples():
            L.append(f"  {r.axis_id:<6}{r.axis_name[:34]:<36}{r.n_pairs:>6}"
                     f"{int(r.n_pairs_before):>7}{r.cos_with_before:>12.3f}"
                     + "".join(f"{getattr(r, 'before_cos_' + c):>11.3f}"
                               f"{getattr(r, 'cos_' + c):>11.3f}" for c in cn))
        L.append("")

    L += ["3. THE SUSPECT AXES: FULL, WITHOUT THE SUSPECT PAIRS, AND THE SUSPECT PAIRS ALONE"
          + (" (suspect = replaced in the revision)" if old_pairs else ""),
          "-" * 78]
    for a in SUSPECT:
        for t in templates:
            r = AX[(AX.template == t) & (AX.axis_id == a)].iloc[0]
            tag = "PRIMARY" if t == primary else "robustness"
            L.append(f"  {a} {names[a]}  [{tag}: \"{t}\"]")
            L.append(f"    pairs {int(r.n_pairs)}, suspect {int(r.n_suspect)} "
                     f"({'replaced' if old_pairs else ', '.join(sorted(SUSPECT[a]))}); "
                     f"their share of the axis "
                     f"{r.suspect_share_of_axis:.0%}")
            if "cos_full_reduced" in r and pd.notna(r.get("cos_full_reduced")):
                for c in cn:
                    star = "  <- target" if c == r.target_control else ""
                    L.append(f"    {c:<10} full {r[f'cos_{c}']:6.3f}   without suspect "
                             f"{r[f'reduced_cos_{c}']:6.3f}   suspect alone "
                             f"{r[f'suspect_cos_{c}']:6.3f}{star}")
                L.append(f"    cosine full vs without suspect: {r.cos_full_reduced:.3f}; "
                         f"suspect alone vs the other pairs: {r.cos_suspect_rest:.3f} "
                         "(do the two groups mean the same?)")
                L.append(f"    drop {r.drop_value:.3f}; needed {r.drop_needed:.3f} "
                         f"(max of {RULE_ABS} and a third of full) -> "
                         f"{'REVISE' if r.revise else 'keep'}"
                         + ("" if t == primary else "  (rule applies to the primary only)"))
                L.append(f"    random removal of {int(r.n_suspect)} pairs ({N_RANDOM} draws): "
                         f"median drop {r.random_drop_median:.3f}, 95th percentile "
                         f"{r.random_drop_p95:.3f}; observed drop above "
                         f"{r.drop_percentile_vs_random:.0f}% of them (information only)")
            L.append("")

    L += ["4. PAIRS LEANING MOST TOWARDS EACH TARGET CONTROL (primary template)",
          "-" * 78]
    for a in SUSPECT:
        tc = TARGET_CONTROL[a]
        g = PRS[(PRS.template == primary) & (PRS.axis_id == a)].sort_values(
            f"cos_diff_{tc}", ascending=False).head(8)
        L.append(f"  {a} ({tc}): cosine of the pair's difference with the control; * suspect")
        for r in g.itertuples():
            L.append(f"    {getattr(r, f'cos_diff_{tc}'):6.3f} {'*' if r.suspect else ' '} "
                     f"{r.positive[:42]:<42} -> {r.negative[:42]}")
        L.append("")

    prim = AX[(AX.template == primary) & AX.axis_id.isin(SUSPECT.keys())]
    L += ["5. DECISION 17 (primary template)", "-" * 78]
    for r in prim.itertuples():
        if pd.notna(getattr(r, "revise", np.nan)):
            L.append(f"  {r.axis_id}: {'REVISE the suspect negatives' if r.revise else 'keep'}"
                     f"  ({r.target_control}: full {getattr(r, 'cos_' + r.target_control):.3f}, "
                     f"without suspect {getattr(r, 'reduced_cos_' + r.target_control):.3f}, "
                     f"drop {r.drop_value:.3f}, needed {r.drop_needed:.3f})")
    L += ["", "Revision (if any): for each suspect anchor, another of its passing Stage D",
          "candidates without the suspect words (suggestion order of decision 14), or",
          "your own negative; then this check is run once more and both versions reported."]

    open(P("check17_report.txt"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    json.dump({"script": "v4_check17_scene_properties.py", "protocol": "1.11",
               "created": time.strftime("%Y-%m-%d %H:%M:%S"), "axes_yaml": AXES_YAML,
               "clip_model": CLIP_MODEL, "templates": templates, "add_period": ADD_PERIOD,
               "controls": CONTROLS, "suspect": {k: sorted(v) for k, v in SUSPECT.items()},
               "target_control": TARGET_CONTROL, "rule_rel": RULE_REL, "rule_abs": RULE_ABS,
               "n_random": N_RANDOM, "seed": SEED},
              open(P("check17_meta.json"), "w"), indent=2)
    print(f"\nRuntime {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
