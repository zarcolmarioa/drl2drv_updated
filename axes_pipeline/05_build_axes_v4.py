"""
05_build_axes.py  (pipeline step 05)

Builds every axis vector used later, checks the text encoder, and
measures how stable each axis is. Images are only used as a random
sample to measure stability; the image FEATURES (step 04) are never
read, so nothing about axis construction can be tuned to the results.

What an axis is
---------------
Each anchor is put into a caption ("a scene that looks foggy"), encoded
with CLIP's text encoder and L2-normalised. The axis is
    normalise( mean(positive captions) - mean(negative captions) ),
exactly as in the old Script A. un2CLIP does not change the text
encoder, so the standard openai/clip-vit-large-patch14-336 text tower
and its text_projection are used.

Axis sets built (one row each in axis_catalog.csv)
-------------------------------------------------
  main       v2, v2_merged, v3a, v3b (primary), v3b_neg
             v2 and v2_merged use the old caption for every anchor
             ("a scene that looks {anchor}"); v3 uses the captions in
             axes_v3.yaml (rule R5).
  template   v3a/v3b/v3b_neg with the other caption templates in
             axes_v3.yaml (photo, place, world)           [robustness]
  centroid   every main set with the positive captions against the
             average of ALL captions of that set (each caption once),
             instead of the negatives: a baseline that does not depend on
             the negative words                       [added 27 Sep]
  neutral    the same against the bare caption "a scene". Kept for
             reference only: every "a scene that looks X" caption differs
             from "a scene" in the same way, so these axes share a common
             component (seen in the first run: cosine 0.28-0.64 with the
             main axes). Use the centroid variant instead.
  subaxis    literal / non-literal sub-axes of SUBAXIS_AXES (lifeless
             only, decision of 27 Sep: option 1), per v3 version
  variant    v2 foggy without "bright" (Y5)
  control    the six control axes in axes_v3.yaml
  v4         [v4 mode, v4_protocol.md 1.14 §6.6] the v4 axes of V4_YAML
             (Stage D2), one positive and one negative caption per pair;
             a negative shared by several pairs counts once per pair, so the
             axis is the mean of the paired differences (decision 20). Built
             with the same template variants (v4@photo, v4@place, v4@world),
             centroid and neutral baselines, and one more variant:
             v4@nocontrol  the axis with the enclosure and darkness
                           directions of decision 17 removed (R2b): a
                           sensitivity version, never primary.
             No v4 sub-axes in the first run (S2b; Stage T comes later).
An axis with fewer than MIN_POS positives or MIN_NEG negatives is
listed as not included and gets no vector.

The v4 sets are added after the v2/v3 sets and their stability is drawn
after them, so every v2/v3 vector and stability result is the same as in
the v3 run (same seed, same order of random draws).

Checks (the run stops if a required one fails)
----------------------------------------------
  T1 TOKENS    HuggingFace and official token ids are identical for
               every caption, and no caption is longer than 77 tokens.
  T2 OFFICIAL  every caption embedding equals the official `clip`
               package's (min cosine >= TEXT_MIN_COS).      [required]
  T3 OLD AXES  the rebuilt v2 axes equal the old axis_vectors.npy
               (min cosine >= OLD_AXES_MIN_COS). Skipped if
               OLD_AXES_DIR is None. A failure is a warning: it means
               the old results were built from different axes.
  T4 CAPTIONS  the v3 captions in the yaml equal the primary template.

Diagnostics (reported, no thresholds, nothing is changed by them)
-----------------------------------------------------------------
  * Anchor counts per axis and set (Y7).
  * Axis-axis cosines per main set.
  * Movement: each v3 axis against its v2 counterpart (cosine, and the
    correlation of image scores on the image sample).
  * Anchor diagnostics: for every caption, its position along its own
    axis relative to the centroid of the set's captions (positives
    should lie on the positive side, negatives on the negative side),
    and the other axis it loads on most strongly (Y2). The value
    relative to "a scene" is kept as a second column. Single captions
    are noisy: this is a screen for the item analysis in 09.
  * Baselines and templates: cosine of each main axis with its centroid,
    neutral and template variants, and the correlation of their image
    scores on the sample.
  * Stability (main and subaxis sets), on N_IMAGES_SAMPLE random images:
      split-half   anchors split at random into two halves, per pole;
                   cosine between the two half-axes and correlation of
                   their image scores (raw and Spearman-Brown);
      pole halves  the same, splitting only the positives (negatives
                   kept whole) or only the negatives;
      leave-one-out  cosine and score correlation with the full axis
                   when each anchor is left out; the most influential
                   anchor per axis;
      bootstrap    anchors resampled with replacement;
      m-curve      axes built from m anchors per pole (m = 2 ... n),
                   compared with the full axis. Where m >= a pole's
                   size that pole is used whole.
    For v4 the halves, bootstrap, leave-one-out and m-curve are drawn by
    PAIR (a positive stays with its negative); the one-pole splits are
    kept as in v3 for comparison. Stability is reported, not used to drop
    an axis (S1a).
  * v4 only: each axis's cosine with the enclosure and darkness control
    directions (decision 17), and with its v4@nocontrol version; the
    v4 x v3b table of cosines and image-score correlations (no matching:
    counterparts are decided in 09, protocol 6.5); the pairs of v4 axes
    with cosine >= V4_MERGE_COS (the text half of the merge test,
    decision 13); pairs and documents per axis, with the largest
    document's share (reported instead of leave-one-document-out, L1c).

Inputs
------
  V2_YAML         axes/axes_consolidated_v2.yaml
  V3_YAML         axes/v3/axes_v3.yaml            (tools/make_axes_v3.py)
  TAGS_CSV        axes/v3/anchor_tags_final.csv   (tagging, 27 Sep)
  EMB_PATH        data/embeddings/embeddings_768.npy   (step 03)
  STATUS_PATH     data/embeddings/status.npy
  OLD_AXES_DIR    folder with the old axis_vectors.npy and axis_names.json
  V4_YAML         axes/v4/stageD2/axes_v4.yaml    (v4_stageD2_revise.py;
                  None: no v4 sets, as in the v3 run)

Outputs (under OUTPUT_DIR)
--------------------------
  axis_vectors.npy        (K, 768) float32, one row per included axis
  axis_catalog.csv        row, key, set, kind, version, template, axis,
                          n_pos, n_neg, included, primary, note
  axis_anchors.csv        key, pole, anchor, caption (every axis)
  caption_embeddings.npy  (C, 768) float32 and captions.csv (row, caption)
  anchor_diagnostics.csv  per anchor, main and subaxis sets
  baseline_variants.csv   main axes against centroid/neutral/template variants
  axis_cosines_<set>.csv  per main set
  version_movement.csv    v3 axes against v2
  stability_summary.csv   per set and axis
  stability_loo.csv       per anchor
  stability_mcurve.csv    per set, axis and m
  axis_cosines.png        v2 and v3b cosine matrices
  stability_mcurve.png    m-curves, v3b against v2
  v4 only: v4_v3b_cosines.csv, v4_v3b_score_r.csv, v4_axes.csv (pairs,
           documents, control cosines, nocontrol version),
           stability_mcurve_v4.png
  build_report.txt        summary (start here)
  run_meta.json           settings, versions and check results

Usage
-----
    conda activate drl2drv
    python 05_build_axes.py

Needs the official clip package for check T2 (already installed for 03):
    pip install ftfy regex
    pip install git+https://github.com/openai/CLIP.git
A few minutes; most of the time goes into the stability resampling.
"""

import csv
import json
import os
import time
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
import yaml
from transformers import CLIPModel, CLIPTokenizer

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


# ============================================================
# SETTINGS
# ============================================================

# --- Inputs ---
V2_YAML           = "axes/axes_consolidated_v2.yaml"
V3_YAML           = "axes/v3/axes_v3.yaml"
TAGS_CSV          = "axes/v3/anchor_tags_final.csv"
EMB_PATH          = "data/embeddings/embeddings_768.npy"
STATUS_PATH       = "data/embeddings/status.npy"
OLD_AXES_DIR      = "data/encoded_all_categories_un2clip_v2_full"  # None to skip
V4_YAML           = "axes/v4/stageD2/axes_v4.yaml"   # None: no v4 sets

CLIP_MODEL_NAME   = "openai/clip-vit-large-patch14-336"
OPENAI_MODEL_NAME = "ViT-L/14@336px"

# --- Output ---
OUTPUT_DIR        = "data/axes_v4"      # new folder (the v3 run: data/axes)

# --- Axis sets ---
V2_TEMPLATE       = "a scene that looks {}"     # the old Script A caption
MERGE_AXES        = ["world-as-staged / scripted", "simulation / digital world"]
V3_VERSIONS       = ["v3a", "v3b", "v3b_neg"]
PRIMARY_SET       = "v3b"
SUBAXIS_AXES      = ["lifeless / deadened"]     # option 1 (27 Sep)
V2_VARIANTS       = [                            # (set name, axis, pole, anchor removed)
    ("v2_foggy_nobright", "veiled / foggy", "negative", "bright"),
]

# --- v4 (v4_protocol.md 1.14, section 6.6) ---
V4_SET            = "v4"
V4_MIN_PAIRS      = 5                  # as the keeping rule's minimum of 5 texts
V4_TEMPLATE_NAMES = {"a photo that looks {}": "photo",      # robustness templates
                     "a place that looks {}": "place",      # (same names as v3)
                     "the world looks {}": "world"}
V4_MERGE_COS      = 0.8                # text half of the merge test (decision 13)
# Control directions of decision 17 (as in v4_check17_scene_properties.py):
# (pole an axis may lean to, other pole), in the primary template. Removed
# from every v4 axis in the v4@nocontrol version (R2b).
V4_CONTROLS = {
    "enclosure": [("enclosed", "open"), ("cramped", "spacious"),
                  ("like a closed room", "like a wide open landscape"),
                  ("like an indoor room", "like the outdoors")],
    "darkness":  [("dark", "bright"), ("dimly lit", "brightly lit"),
                  ("like nighttime", "like daytime"), ("shadowy", "sunlit")],
}

# --- Stability ---
N_IMAGES_SAMPLE   = 20_000
N_SPLITS          = 200
N_BOOT            = 200
MCURVE_DRAWS      = 100
RANDOM_SEED       = 42

# --- Checks ---
REQUIRE_OFFICIAL  = True
TEXT_MIN_COS      = 0.9999
OLD_AXES_MIN_COS  = 0.9999

# --- Numerics ---
BATCH_SIZE        = 64
ALLOW_TF32        = False
DEVICE            = "cuda" if torch.cuda.is_available() else "cpu"

# Figure colours (validated categorical slots 1-2; diverging blue/grey/red)
C_V2, C_V3        = "#2a78d6", "#eb6834"
DIVERGING         = ["#2a78d6", "#f0efec", "#d03b3b"]
INK, INK_MUTED    = "#1f1f1e", "#6b6a64"


# ============================================================
# Helpers
# ============================================================

def P(name):
    return os.path.join(OUTPUT_DIR, name)


def unit(v):
    return v / max(np.linalg.norm(v), 1e-12)


def make_axis(set_, kind, version, template, name, pos, neg, pos_anchors,
              neg_anchors, min_pos, min_neg, note=""):
    ok = len(pos) >= min_pos and len(neg) >= min_neg
    if not ok and not note:
        note = f"below minimum ({len(pos)} / {len(neg)})"
    return {"set": set_, "kind": kind, "version": version,
            "template": template, "axis": name, "pos": list(pos),
            "neg": list(neg), "pos_anchors": list(pos_anchors),
            "neg_anchors": list(neg_anchors), "included": ok,
            "note": note, "key": f"{set_}::{name}"}


def load_v2(path):
    with open(path) as f:
        data = yaml.safe_load(f)
    out = OrderedDict()
    for a in data["axes"]:
        out[a["name"]] = {
            "pos": [x.strip() for x in a["positive_anchors"] if x.strip()],
            "neg": [x.strip() for x in a["negative_anchors"] if x.strip()]}
    return out


def corr_cols(A, b):
    """Pearson correlation of each column of A with vector b."""
    A = A - A.mean(0)
    b = b - b.mean()
    return (A.T @ b) / (np.linalg.norm(A, axis=0) * np.linalg.norm(b) + 1e-12)


def sb(r):
    return 2 * r / (1 + r) if r > -1 else np.nan


# ============================================================
# Axis sets
# ============================================================

def build_specs(v2, v3, tags):
    meta = v3["meta"]
    mp, mn = meta["min_positive"], meta["min_negative"]
    neutral = meta["neutral_caption"]
    templates = v3["templates"]
    specs = []

    # v2 and v2_merged: old caption for every anchor, no minimum applied
    # (they are reproduced as they were).
    cap2 = lambda xs: [V2_TEMPLATE.format(x) for x in xs]
    for n, ax in v2.items():
        specs.append(make_axis("v2", "main", "v2", "v2", n, cap2(ax["pos"]),
                               cap2(ax["neg"]), ax["pos"], ax["neg"], 1, 1))
    ref = v3["reference"]["v2_merged"]
    merged_name = next(a["name"] for a in v3["versions"]["v3a"]["axes"]
                       if a["name"] not in v2)
    for n, ax in v2.items():
        if n == MERGE_AXES[1]:
            continue
        if n == MERGE_AXES[0]:
            specs.append(make_axis("v2_merged", "main", "v2_merged", "v2",
                                   merged_name, cap2(ref["positive"]),
                                   cap2(ref["negative"]), ref["positive"],
                                   ref["negative"], 1, 1))
        else:
            specs.append(make_axis("v2_merged", "main", "v2_merged", "v2", n,
                                   cap2(ax["pos"]), cap2(ax["neg"]),
                                   ax["pos"], ax["neg"], 1, 1))

    # v3 versions: main, template variants, sub-axes
    t4 = []
    for ver in V3_VERSIONS:
        for a in v3["versions"][ver]["axes"]:
            pe, ne = a["positive"], a["negative"]
            for e in pe + ne:
                expect = templates[0][e["frame"]].format(e["text"])
                if e["caption"] != expect:
                    t4.append((ver, a["name"], e["anchor"], e["caption"], expect))
            note = "" if a["included"] else "left out (R8)"
            specs.append(make_axis(ver, "main", ver, templates[0]["name"],
                                   a["name"], [e["caption"] for e in pe],
                                   [e["caption"] for e in ne],
                                   [e["anchor"] for e in pe],
                                   [e["anchor"] for e in ne], mp, mn, note))
            for t in templates[1:]:
                specs.append(make_axis(
                    f"{ver}@{t['name']}", "template", ver, t["name"], a["name"],
                    [t[e["frame"]].format(e["text"]) for e in pe],
                    [t[e["frame"]].format(e["text"]) for e in ne],
                    [e["anchor"] for e in pe], [e["anchor"] for e in ne],
                    mp, mn, note))
            if a["name"] in SUBAXIS_AXES:
                for tag in ("literal", "non_literal"):
                    def pick(es, pole):
                        out = []
                        for e in es:
                            k = (a["name"], pole, e["anchor"])
                            if k not in tags:
                                raise SystemExit(f"No tag for {k} in {TAGS_CSV}")
                            if tags[k] == tag:
                                out.append(e)
                        return out
                    ps, ns = pick(pe, "positive"), pick(ne, "negative")
                    specs.append(make_axis(
                        f"{ver}@{tag}", "subaxis", ver, templates[0]["name"],
                        a["name"], [e["caption"] for e in ps],
                        [e["caption"] for e in ns], [e["anchor"] for e in ps],
                        [e["anchor"] for e in ns], mp, mn))

    # centroid baseline for every main set: positives against the average
    # of all captions of the set's included axes (each caption once)
    mains = [x for x in specs if x["kind"] == "main"]
    for set_ in dict.fromkeys(x["set"] for x in mains):
        cent = list(dict.fromkeys(c for x in mains if x["set"] == set_
                                  and x["included"] for c in x["pos"] + x["neg"]))
        for s in [x for x in mains if x["set"] == set_]:
            specs.append(make_axis(f"{set_}@centroid", "centroid", s["version"],
                                   s["template"], s["axis"], s["pos"], cent,
                                   s["pos_anchors"], ["(centroid)"] * len(cent),
                                   mp if set_ in V3_VERSIONS else 1, 1,
                                   s["note"] if not s["included"] else ""))
            if not s["included"]:
                specs[-1]["included"] = False

    # neutral baseline for every main set (reference only, see docstring)
    for s in mains:
        specs.append(make_axis(f"{s['set']}@neutral", "neutral", s["version"],
                               s["template"], s["axis"], s["pos"], [neutral],
                               s["pos_anchors"], ["(neutral)"],
                               mp if s["set"] in V3_VERSIONS else 1, 1,
                               s["note"] if not s["included"] else ""))
        if not s["included"]:
            specs[-1]["included"] = False

    # v2 variants
    for set_, n, pole, drop in V2_VARIANTS:
        ax = {k: list(v) for k, v in v2[n].items()}
        key = "pos" if pole == "positive" else "neg"
        if drop not in ax[key]:
            raise SystemExit(f"V2_VARIANTS: '{drop}' not on {n} ({pole})")
        ax[key].remove(drop)
        specs.append(make_axis(set_, "variant", "v2", "v2", n, cap2(ax["pos"]),
                               cap2(ax["neg"]), ax["pos"], ax["neg"], 1, 1,
                               f"v2 without '{drop}' ({pole})"))

    # controls
    for c in v3["controls"]:
        specs.append(make_axis("controls", "control", "control", "full", c["name"],
                               c["positive"], c["negative"], c["positive"],
                               c["negative"], 1, 1))
    for s in specs:
        s["primary"] = s["set"] == PRIMARY_SET
    return specs, neutral, t4


def build_v4_specs(doc, neutral):
    """v4 axes from the Stage D2 yaml: main (one caption per pair, a shared
    negative counted once per pair), template variants, centroid and
    neutral baselines, and the nocontrol variant (R2b)."""
    tpl = doc["template"]
    if tpl != V2_TEMPLATE:
        raise SystemExit(f"T4 (v4): template '{tpl}' is not '{V2_TEMPLATE}'")
    rob = list(doc.get("templates_robustness", []))
    unknown = [t for t in rob if t not in V4_TEMPLATE_NAMES]
    if unknown:
        raise SystemExit(f"v4 robustness templates without a name: {unknown}")
    specs, info = [], []
    for ax in doc["axes"]:
        name = f"{ax['id']} {ax['name']}"
        pr = ax["pairs"]
        P_ = [p["positive"] for p in pr]
        N_ = [p["negative"] for p in pr]
        ids = [p["anchor_id"] for p in pr]
        mk = lambda set_, kind, tname, t, note="": make_axis(
            set_, kind, V4_SET, tname, name, [t.format(x) for x in P_],
            [t.format(x) for x in N_], P_, N_, V4_MIN_PAIRS, V4_MIN_PAIRS, note)
        specs.append(mk(V4_SET, "main", "scene", tpl))
        for t in rob:
            specs.append(mk(f"{V4_SET}@{V4_TEMPLATE_NAMES[t]}", "template",
                            V4_TEMPLATE_NAMES[t], t))
        nc = mk(f"{V4_SET}@nocontrol", "variant", "scene", tpl,
                "enclosure and darkness directions removed (R2b)")
        nc["remove_controls"] = True
        specs.append(nc)
        for s in specs[-(len(rob) + 2):]:
            s["pair_ids"] = ids
        docs = [ph.split("-P")[0] for p in pr for ph in p.get("phrase_ids", [])]
        dc = pd.Series(docs).value_counts() if docs else pd.Series(dtype=int)
        info.append({"axis": name, "pairs": len(pr), "documents": int(len(dc)),
                     "largest_document": dc.index[0] if len(dc) else "",
                     "largest_document_share": float(dc.iloc[0] / dc.sum())
                     if len(dc) else np.nan,
                     "distinct_negatives": len(set(N_))})
    mains = [s for s in specs if s["kind"] == "main"]
    cent = list(dict.fromkeys(c for s in mains if s["included"]
                              for c in s["pos"] + s["neg"]))
    for s in mains:
        for kind, neg, lab in (("centroid", cent, "(centroid)"),
                               ("neutral", [neutral], "(neutral)")):
            b = make_axis(f"{V4_SET}@{kind}", kind, V4_SET, s["template"], s["axis"],
                          s["pos"], neg, s["pos_anchors"], [lab] * len(neg),
                          V4_MIN_PAIRS, 1, s["note"] if not s["included"] else "")
            b["included"] = b["included"] and s["included"]
            specs.append(b)
    for s in specs:
        s["primary"] = False
    return specs, pd.DataFrame(info)


# ============================================================
# Text encoding and checks
# ============================================================

def load_hf():
    model = CLIPModel.from_pretrained(CLIP_MODEL_NAME, use_safetensors=True)
    model.vision_model = None           # text tower only
    model.visual_projection = None
    tok = CLIPTokenizer.from_pretrained(CLIP_MODEL_NAME)
    return model.to(DEVICE).eval(), tok


@torch.inference_mode()
def encode_hf(model, tok, captions):
    out = []
    for i in range(0, len(captions), BATCH_SIZE):
        b = tok(captions[i:i + BATCH_SIZE], padding="max_length",
                max_length=77, truncation=False, return_tensors="pt").to(DEVICE)
        pooled = model.text_model(input_ids=b["input_ids"],
                                  attention_mask=b["attention_mask"]).pooler_output
        e = model.text_projection(pooled)
        e = e / e.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        out.append(e.float().cpu().numpy())
    return np.concatenate(out).astype(np.float32)


def check_tokens(tok, captions):
    """T1. Returns (n_mismatch, examples, max_len, clip_module or None)."""
    lens = [len(tok(c)["input_ids"]) for c in captions]
    if max(lens) > 77:
        long = [c for c, n in zip(captions, lens) if n > 77]
        raise SystemExit(f"T1: captions longer than 77 tokens: {long[:3]}")
    try:
        import clip
    except ImportError:
        return None, [], max(lens), None
    ref = clip.tokenize(captions).numpy()
    eot = tok.eos_token_id
    bad = []
    for c, r in zip(captions, ref):
        r = list(r[: int(np.argmax(r == eot)) + 1])
        h = tok(c)["input_ids"]
        if r != h:
            bad.append(c)
    return len(bad), bad[:5], max(lens), clip


@torch.inference_mode()
def check_official(clip, captions, E_hf):
    """T2. Cosine between official and HF embeddings, per caption."""
    ref, _ = clip.load(OPENAI_MODEL_NAME, device=DEVICE, jit=False)
    ref = ref.float().eval()
    out = []
    for i in range(0, len(captions), BATCH_SIZE):
        t = clip.tokenize(captions[i:i + BATCH_SIZE]).to(DEVICE)
        e = ref.encode_text(t).float()
        e = e / e.norm(dim=-1, keepdim=True)
        out.append(e.cpu().numpy())
    del ref
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    E_ref = np.concatenate(out)
    return (E_ref * E_hf).sum(1)


def check_old_axes(v2_vectors):
    """T3. Cosine of each rebuilt v2 axis with the old axis_vectors.npy."""
    A = np.load(os.path.join(OLD_AXES_DIR, "axis_vectors.npy"))
    with open(os.path.join(OLD_AXES_DIR, "axis_names.json")) as f:
        names = json.load(f)
    if isinstance(names, dict):
        names = names.get("axis_names", names.get("names", list(names)))
    res = {}
    for n, a in zip(names, A):
        if n in v2_vectors:
            res[n] = float(unit(a) @ v2_vectors[n])
    missing = [n for n in v2_vectors if n not in res]
    return res, missing


# ============================================================
# Stability
# ============================================================

def stability(Pe, Ne, E, s0, a0, rng):
    """Pe, Ne: caption embeddings of the two poles; E: image sample;
    s0, a0: full-axis scores and vector. Returns (summary, loo_rows,
    mcurve_rows)."""
    n, m = len(Pe), len(Ne)
    ax = lambda pi, ni: unit(Pe[pi].mean(0) - Ne[ni].mean(0))
    summ = {}

    def halves(k):
        p = rng.permutation(k)
        return p[: k // 2], p[k // 2:]

    def pair_stats(pairs):
        A = np.stack([a for a, _ in pairs])
        B = np.stack([b for _, b in pairs])
        cos = (A * B).sum(1)
        SA, SB_ = E @ A.T, E @ B.T
        SA -= SA.mean(0)
        SB_ -= SB_.mean(0)
        r = (SA * SB_).sum(0) / (np.linalg.norm(SA, axis=0)
                                 * np.linalg.norm(SB_, axis=0) + 1e-12)
        return cos, r

    allp, alln = np.arange(n), np.arange(m)
    if n >= 2 and m >= 2:
        pairs = []
        for _ in range(N_SPLITS):
            (p1, p2), (n1, n2) = halves(n), halves(m)
            pairs.append((ax(p1, n1), ax(p2, n2)))
        c, r = pair_stats(pairs)
        summ.update(split_cos_med=np.median(c), split_cos_p05=np.percentile(c, 5),
                    split_r_med=np.median(r), split_r_p05=np.percentile(r, 5),
                    split_r_sb_med=sb(np.median(r)))
    if n >= 2:
        pairs = []
        for _ in range(N_SPLITS):
            p1, p2 = halves(n)
            pairs.append((ax(p1, alln), ax(p2, alln)))
        c, r = pair_stats(pairs)
        summ.update(pos_split_cos_med=np.median(c), pos_split_r_med=np.median(r))
    if m >= 2:
        pairs = []
        for _ in range(N_SPLITS):
            n1, n2 = halves(m)
            pairs.append((ax(allp, n1), ax(allp, n2)))
        c, r = pair_stats(pairs)
        summ.update(neg_split_cos_med=np.median(c), neg_split_r_med=np.median(r))

    # bootstrap
    B = np.stack([ax(rng.integers(0, n, n), rng.integers(0, m, m))
                  for _ in range(N_BOOT)])
    c = B @ a0
    r = corr_cols(E @ B.T, s0)
    summ.update(boot_cos_med=np.median(c), boot_cos_p05=np.percentile(c, 5),
                boot_r_med=np.median(r), boot_r_p05=np.percentile(r, 5))

    # leave-one-out
    loo = []
    vecs = []
    for pole, k in (("positive", n), ("negative", m)):
        for i in range(k):
            if k < 2:
                continue
            keep = np.delete(np.arange(k), i)
            v = ax(keep, alln) if pole == "positive" else ax(allp, keep)
            vecs.append(v)
            loo.append({"pole": pole, "index": i})
    if vecs:
        V = np.stack(vecs)
        c, r = V @ a0, corr_cols(E @ V.T, s0)
        for row, ci, ri in zip(loo, c, r):
            row.update(loo_cos=float(ci), loo_r=float(ri))
        worst = int(np.argmin(c))
        summ.update(loo_cos_min=float(c.min()), loo_r_min=float(r.min()),
                    loo_worst=(loo[worst]["pole"], loo[worst]["index"]))

    # m-curve
    mc = []
    for mm in range(2, max(n, m) + 1):
        V = []
        for _ in range(MCURVE_DRAWS):
            pi = rng.choice(n, min(mm, n), replace=False)
            ni = rng.choice(m, min(mm, m), replace=False)
            V.append(ax(pi, ni))
        V = np.stack(V)
        c, r = V @ a0, corr_cols(E @ V.T, s0)
        mc.append({"m": mm, "cos_mean": float(c.mean()),
                   "cos_p05": float(np.percentile(c, 5)),
                   "r_mean": float(r.mean()),
                   "r_p05": float(np.percentile(r, 5))})
    return summ, loo, mc


def stability_paired(Pe, Ne, E, s0, a0, rng):
    """v4: the same battery as stability(), drawn by pair (row i of Pe and
    of Ne is pair i). The one-pole splits are kept as in v3. Leave-one-out
    rows have pole 'pair'."""
    k = len(Pe)
    ax = lambda idx: unit(Pe[idx].mean(0) - Ne[idx].mean(0))
    axpn = lambda pi, ni: unit(Pe[pi].mean(0) - Ne[ni].mean(0))
    allk = np.arange(k)
    summ = {}

    def halves():
        p = rng.permutation(k)
        return p[: k // 2], p[k // 2:]

    def pair_stats(pairs):
        A = np.stack([a for a, _ in pairs])
        B = np.stack([b for _, b in pairs])
        cos = (A * B).sum(1)
        SA, SB_ = E @ A.T, E @ B.T
        SA -= SA.mean(0)
        SB_ -= SB_.mean(0)
        r = (SA * SB_).sum(0) / (np.linalg.norm(SA, axis=0)
                                 * np.linalg.norm(SB_, axis=0) + 1e-12)
        return cos, r

    if k >= 2:
        prs = []
        for _ in range(N_SPLITS):
            h1, h2 = halves()
            prs.append((ax(h1), ax(h2)))
        c, r = pair_stats(prs)
        summ.update(split_cos_med=np.median(c), split_cos_p05=np.percentile(c, 5),
                    split_r_med=np.median(r), split_r_p05=np.percentile(r, 5),
                    split_r_sb_med=sb(np.median(r)))
        for key, f in (("pos", lambda h: axpn(h, allk)), ("neg", lambda h: axpn(allk, h))):
            prs = []
            for _ in range(N_SPLITS):
                h1, h2 = halves()
                prs.append((f(h1), f(h2)))
            c, r = pair_stats(prs)
            summ.update({f"{key}_split_cos_med": np.median(c),
                         f"{key}_split_r_med": np.median(r)})

    B = np.stack([ax(rng.integers(0, k, k)) for _ in range(N_BOOT)])
    c = B @ a0
    r = corr_cols(E @ B.T, s0)
    summ.update(boot_cos_med=np.median(c), boot_cos_p05=np.percentile(c, 5),
                boot_r_med=np.median(r), boot_r_p05=np.percentile(r, 5))

    loo = []
    if k >= 2:
        V = np.stack([ax(np.delete(allk, i)) for i in range(k)])
        c, r = V @ a0, corr_cols(E @ V.T, s0)
        loo = [{"pole": "pair", "index": i, "loo_cos": float(ci), "loo_r": float(ri)}
               for i, (ci, ri) in enumerate(zip(c, r))]
        worst = int(np.argmin(c))
        summ.update(loo_cos_min=float(c.min()), loo_r_min=float(r.min()),
                    loo_worst=("pair", worst))

    mc = []
    for mm in range(2, k + 1):
        V = np.stack([ax(rng.choice(k, mm, replace=False)) for _ in range(MCURVE_DRAWS)])
        c, r = V @ a0, corr_cols(E @ V.T, s0)
        mc.append({"m": mm, "cos_mean": float(c.mean()),
                   "cos_p05": float(np.percentile(c, 5)),
                   "r_mean": float(r.mean()), "r_p05": float(np.percentile(r, 5))})
    return summ, loo, mc


# ============================================================
# Figures
# ============================================================

def fig_cosines(mats, path):
    cmap = LinearSegmentedColormap.from_list("div", DIVERGING)
    fig, axs = plt.subplots(1, len(mats), figsize=(7.2 * len(mats), 6.6))
    axs = np.atleast_1d(axs)
    lim = max(np.abs(M.values[~np.eye(len(M), dtype=bool)]).max()
              for _, M in mats)
    for axp, (title, M) in zip(axs, mats):
        V = M.values.copy()
        np.fill_diagonal(V, np.nan)
        axp.imshow(V, cmap=cmap, vmin=-lim, vmax=lim)
        k = len(M)
        for i in range(k):
            for j in range(k):
                if i != j:
                    axp.text(j, i, f"{V[i, j]:.2f}", ha="center", va="center",
                             fontsize=8, color=INK)
        labels = [n.split(" / ")[0] for n in M.index]
        axp.set_xticks(range(k), labels, rotation=45, ha="right", fontsize=9,
                       color=INK)
        axp.set_yticks(range(k), labels, fontsize=9, color=INK)
        axp.set_title(title, fontsize=11, color=INK, loc="left")
        for s in axp.spines.values():
            s.set_visible(False)
    fig.suptitle(f"Cosine between axis vectors (colour scale +/-{lim:.2f}; "
                 "red = similar direction, blue = opposite)",
                 fontsize=10, color=INK_MUTED, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_mcurve(mdf, pairs, path):
    k = len(pairs)
    cols = 4
    rows = int(np.ceil(k / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(3.4 * cols, 2.8 * rows),
                            sharey=True)
    axs = np.atleast_1d(axs).ravel()
    for axp, (name, s3, a3, s2, a2) in zip(axs, pairs):
        for s, a, col, lab in ((s2, a2, C_V2, s2), (s3, a3, C_V3, s3)):
            d = mdf[(mdf.set == s) & (mdf.axis == a)]
            if len(d):
                axp.plot(d.m, d.cos_mean, color=col, lw=2, marker="o", ms=4,
                         label=lab)
        axp.set_title(name.split(" / ")[0], fontsize=10, color=INK, loc="left")
        axp.grid(axis="y", color="#e4e3de", lw=0.8)
        axp.tick_params(colors=INK_MUTED, labelsize=8)
        for s in ("top", "right"):
            axp.spines[s].set_visible(False)
    for axp in axs[k:]:
        axp.axis("off")
    axs[0].set_ylabel("cosine with full axis", fontsize=9, color=INK)
    axs[0].legend(fontsize=8, frameon=False, loc="lower right")
    fig.supxlabel("anchors per pole (m)", fontsize=9, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ============================================================
# Main
# ============================================================

def main():
    t0 = time.time()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = ALLOW_TF32
    torch.backends.cudnn.allow_tf32 = ALLOW_TF32
    rng = np.random.default_rng(RANDOM_SEED)
    print("=" * 70)
    print("05_build_axes.py")
    print("=" * 70)

    # --- Inputs ---
    v2 = load_v2(V2_YAML)
    with open(V3_YAML) as f:
        v3 = yaml.safe_load(f)
    tg = pd.read_csv(TAGS_CSV)
    tags = {(r.axis, r.pole, r.anchor): r.tag for r in tg.itertuples()}
    specs, neutral, t4 = build_specs(v2, v3, tags)
    if t4:
        raise SystemExit(f"T4: yaml captions differ from the primary template: "
                         f"{t4[:3]}")
    v4doc, v4info, ctrl_caps = None, None, set()
    if V4_YAML:
        with open(V4_YAML) as f:
            v4doc = yaml.safe_load(f)
        v4specs, v4info = build_v4_specs(v4doc, neutral)
        specs += v4specs               # after v2/v3: their rows are unchanged
        ctrl_caps = {V2_TEMPLATE.format(x) for prs in V4_CONTROLS.values()
                     for pr in prs for x in pr}
        print(f"v4: {len(v4info)} axes, {int(v4info.pairs.sum())} pairs "
              f"({V4_YAML})")
    captions = sorted({c for s in specs for c in s["pos"] + s["neg"]}
                      | {neutral} | ctrl_caps)
    cidx = {c: i for i, c in enumerate(captions)}
    print(f"Axis specs: {len(specs)}   unique captions: {len(captions)}")

    # --- Encode and check ---
    print("\nText encoder (HuggingFace)...")
    model, tok = load_hf()
    n_bad, bad_ex, max_len, clip_mod = check_tokens(tok, captions)
    C = encode_hf(model, tok, captions)
    del model
    checks = {"max_tokens": max_len}
    if clip_mod is None:
        msg = "T1/T2: official clip package not installed."
        if REQUIRE_OFFICIAL:
            raise SystemExit(msg + " Install it (see docstring) or set "
                             "REQUIRE_OFFICIAL = False.")
        print("  " + msg + " Skipped.")
        checks["official"] = "skipped"
    else:
        print(f"  T1 tokens: {n_bad} mismatching captions; longest "
              f"{max_len} tokens -> {'PASS' if n_bad == 0 else 'FAIL'}")
        if n_bad:
            raise SystemExit(f"T1: token ids differ, e.g. {bad_ex}")
        cs = check_official(clip_mod, captions, C)
        ok2 = cs.min() >= TEXT_MIN_COS
        print(f"  T2 official: min cos {cs.min():.7f} (caption "
              f"'{captions[int(np.argmin(cs))]}') -> "
              f"{'PASS' if ok2 else 'FAIL'}")
        checks.update(official="run", token_mismatch=n_bad,
                      official_min_cos=float(cs.min()),
                      official_mean_cos=float(cs.mean()))
        if not ok2:
            raise SystemExit("T2: caption embeddings differ from the "
                             "official clip package. Stopping.")

    # --- v4 control directions (decision 17), built like an axis ---
    ctrl_vec, Q = {}, None
    if v4doc is not None:
        for cn, prs in V4_CONTROLS.items():
            ctrl_vec[cn] = unit(np.mean([C[cidx[V2_TEMPLATE.format(a)]]
                                         - C[cidx[V2_TEMPLATE.format(b)]]
                                         for a, b in prs], axis=0))
        Q = np.linalg.qr(np.stack(list(ctrl_vec.values()), axis=1))[0]

    # --- Axis vectors ---
    rows, vecs = [], []
    for s in specs:
        if s["included"]:
            a = unit(C[[cidx[c] for c in s["pos"]]].mean(0)
                     - C[[cidx[c] for c in s["neg"]]].mean(0))
            if s.get("remove_controls"):
                a = unit(a - Q @ (Q.T @ a))      # R2b: controls removed
            s["vec"] = a
            s["row"] = len(vecs)
            vecs.append(a.astype(np.float32))
        else:
            s["vec"], s["row"] = None, -1
    AV = np.stack(vecs)
    bykey = {s["key"]: s for s in specs}

    # --- T3: old axes ---
    if OLD_AXES_DIR:
        v2vec = {s["axis"]: s["vec"] for s in specs if s["set"] == "v2"}
        try:
            t3, t3_missing = check_old_axes(v2vec)
            m3 = min(t3.values()) if t3 else np.nan
            ok3 = bool(t3) and m3 >= OLD_AXES_MIN_COS and not t3_missing
            print(f"  T3 old axes: {len(t3)} matched, min cos {m3:.7f}"
                  + (f", missing {t3_missing}" if t3_missing else "")
                  + f" -> {'PASS' if ok3 else 'WARN'}")
            checks.update(old_axes=t3, old_axes_missing=t3_missing,
                          old_axes_pass=ok3)
        except FileNotFoundError as e:
            print(f"  T3 old axes: SKIPPED ({e})")
            checks["old_axes"] = "skipped"
    else:
        checks["old_axes"] = "skipped"

    # --- Image sample ---
    print(f"\nImage sample: {N_IMAGES_SAMPLE:,} random encoded images...")
    status = np.load(STATUS_PATH)
    ok_rows = np.where(status == 1)[0]
    pick = np.sort(rng.choice(ok_rows, min(N_IMAGES_SAMPLE, len(ok_rows)),
                              replace=False))
    E = np.asarray(np.load(EMB_PATH, mmap_mode="r")[pick], dtype=np.float64)

    # --- Axis cosines per main set ---
    main_sets = ["v2", "v2_merged"] + V3_VERSIONS
    v3_main_sets = list(main_sets)
    if v4doc is not None:
        main_sets.append(V4_SET)
    cos_mats = {}
    for ms in main_sets:
        ss = [s for s in specs if s["set"] == ms and s["included"]]
        M = np.stack([s["vec"] for s in ss])
        cos_mats[ms] = pd.DataFrame(M @ M.T, index=[s["axis"] for s in ss],
                                    columns=[s["axis"] for s in ss])
        cos_mats[ms].to_csv(P(f"axis_cosines_{ms}.csv"), float_format="%.4f")

    # --- Movement from v2 ---
    v2k = {s["axis"]: s for s in specs if s["set"] == "v2"}
    v2mk = {s["axis"]: s for s in specs if s["set"] == "v2_merged"}
    mov = []
    for ver in V3_VERSIONS:
        for s in [x for x in specs if x["set"] == ver and x["included"]]:
            ref = v2k.get(s["axis"]) or v2mk.get(s["axis"])
            refm = v2mk.get(s["axis"])
            row = {"version": ver, "axis": s["axis"],
                   "v2_counterpart": "v2" if s["axis"] in v2k else "v2_merged",
                   "cos_v2": float(s["vec"] @ ref["vec"]),
                   "r_v2": float(np.corrcoef(E @ s["vec"], E @ ref["vec"])[0, 1]),
                   "cos_v2_merged": float(s["vec"] @ refm["vec"]),
                   "r_v2_merged": float(np.corrcoef(E @ s["vec"],
                                                    E @ refm["vec"])[0, 1])}
            mov.append(row)
    mov = pd.DataFrame(mov)
    mov.to_csv(P("version_movement.csv"), index=False, float_format="%.4f")

    # --- Anchor diagnostics and stability ---
    # v2/v3 first, in the order of the v3 run (same random draws); v4 last
    stab_sets = v3_main_sets + [f"{v}@{t}" for v in V3_VERSIONS
                                for t in ("literal", "non_literal")]
    if v4doc is not None:
        stab_sets.append(V4_SET)
    e_neu = C[cidx[neutral]]
    # centroid of each main set (the reference for the anchor diagnostics)
    cent_vec = {}
    for ms in main_sets:
        caps = list(dict.fromkeys(c for s in specs if s["set"] == ms
                                  and s["included"] for c in s["pos"] + s["neg"]))
        cent_vec[ms] = C[[cidx[c] for c in caps]].mean(0)
    diag, summ_rows, loo_rows, mc_rows = [], [], [], []
    print("Stability and anchor diagnostics...")
    for st in stab_sets:
        ss = [s for s in specs if s["set"] == st and s["included"]]
        others = [s for s in specs if s["set"] == st.split("@")[0]
                  and s["included"]]
        for s in ss:
            Pe = C[[cidx[c] for c in s["pos"]]]
            Ne = C[[cidx[c] for c in s["neg"]]]
            s0 = E @ s["vec"]
            anchors = {"positive": s["pos_anchors"], "negative": s["neg_anchors"]}
            if st == V4_SET:            # v4: drawn by pair
                summ, loo, mc = stability_paired(Pe, Ne, E, s0, s["vec"], rng)
                anchors["pair"] = [f"{i}: {p} -> {n}" for i, p, n in
                                   zip(s["pair_ids"], s["pos_anchors"],
                                       s["neg_anchors"])]
            else:
                summ, loo, mc = stability(Pe, Ne, E, s0, s["vec"], rng)
            if "loo_worst" in summ:
                pole, i = summ.pop("loo_worst")
                summ["loo_worst_anchor"] = (anchors[pole][i] if pole == "pair"
                                            else f"{anchors[pole][i]} ({pole[:3]})")
            summ_rows.append({"set": st, "axis": s["axis"],
                              "n_pos": len(Pe), "n_neg": len(Ne), **summ})
            for r in loo:
                loo_rows.append({"set": st, "axis": s["axis"], "pole": r["pole"],
                                 "anchor": anchors[r["pole"]][r["index"]],
                                 "loo_cos": r["loo_cos"], "loo_r": r["loo_r"]})
            for r in mc:
                mc_rows.append({"set": st, "axis": s["axis"], **r})
            # anchor diagnostics
            for pole, caps, ancs, sign in (("positive", s["pos"], s["pos_anchors"], 1),
                                           ("negative", s["neg"], s["neg_anchors"], -1)):
                e_cen = cent_vec[st.split("@")[0]]
                for c, a in zip(caps, ancs):
                    d = C[cidx[c]] - e_cen
                    own = sign * float(d @ s["vec"])
                    own_neu = sign * float((C[cidx[c]] - e_neu) @ s["vec"])
                    best, bname = -np.inf, ""
                    for o in others:
                        if o["axis"] == s["axis"]:
                            continue
                        v = abs(float(d @ o["vec"]))
                        if v > best:
                            best, bname = v, o["axis"]
                    diag.append({"set": st, "axis": s["axis"], "pole": pole,
                                 "anchor": a, "caption": c,
                                 "cos_with_axis": float(C[cidx[c]] @ s["vec"]),
                                 "own_side_vs_centroid": own,
                                 "own_side_vs_neutral": own_neu,
                                 "wrong_side": own < 0,
                                 "strongest_other_axis": bname,
                                 "other_axis_abs": best if bname else np.nan,
                                 "other_stronger": bool(bname) and best > abs(own)})
    diag = pd.DataFrame(diag)
    summ_df = pd.DataFrame(summ_rows)
    loo_df = pd.DataFrame(loo_rows)
    mc_df = pd.DataFrame(mc_rows)
    diag.to_csv(P("anchor_diagnostics.csv"), index=False, float_format="%.4f")
    summ_df.to_csv(P("stability_summary.csv"), index=False, float_format="%.4f")
    loo_df.to_csv(P("stability_loo.csv"), index=False, float_format="%.4f")
    mc_df.to_csv(P("stability_mcurve.csv"), index=False, float_format="%.4f")

    # --- Baseline and template variants against each main axis ---
    base_rows = []
    for ms in main_sets:
        gen = unit(cent_vec[ms] - e_neu)       # 'a scene that looks X' vs 'a scene'
        for s in [x for x in specs if x["set"] == ms and x["included"]]:
            row = {"set": ms, "axis": s["axis"],
                   "cos_generic_direction": float(s["vec"] @ gen)}
            sm = E @ s["vec"]
            for var in ("centroid", "neutral", "photo", "place", "world"):
                o = bykey.get(f"{ms}@{var}::{s['axis']}")
                if o is not None and o["included"]:
                    row[f"cos_{var}"] = float(s["vec"] @ o["vec"])
                    row[f"r_{var}"] = float(np.corrcoef(sm, E @ o["vec"])[0, 1])
            base_rows.append(row)
    base_df = pd.DataFrame(base_rows)
    base_df.to_csv(P("baseline_variants.csv"), index=False, float_format="%.4f")

    # --- v4: controls, nocontrol version, v4 x v3b, merge (text half) ---
    v4_tab, v4v3_cos, v4v3_r, merge_pairs = None, None, None, []
    if v4doc is not None:
        v4m = [s for s in specs if s["set"] == V4_SET and s["included"]]
        rows4 = []
        for s in v4m:
            row = {"axis": s["axis"]}
            for cn, cv in ctrl_vec.items():
                row[f"cos_{cn}"] = float(s["vec"] @ cv)
            nc = bykey.get(f"{V4_SET}@nocontrol::{s['axis']}")
            if nc is not None and nc["included"]:
                row["cos_nocontrol"] = float(s["vec"] @ nc["vec"])
                row["r_nocontrol"] = float(np.corrcoef(E @ s["vec"], E @ nc["vec"])[0, 1])
            rows4.append(row)
        v4_tab = v4info.merge(pd.DataFrame(rows4), on="axis", how="left")
        v4_tab.to_csv(P("v4_axes.csv"), index=False, float_format="%.4f")
        v3b = [s for s in specs if s["set"] == PRIMARY_SET and s["included"]]
        A4 = np.stack([s["vec"] for s in v4m])
        A3 = np.stack([s["vec"] for s in v3b])
        n4, n3 = [s["axis"] for s in v4m], [s["axis"] for s in v3b]
        v4v3_cos = pd.DataFrame(A4 @ A3.T, index=n4, columns=n3)
        S4, S3 = E @ A4.T, E @ A3.T
        S4 = (S4 - S4.mean(0)) / S4.std(0)
        S3 = (S3 - S3.mean(0)) / S3.std(0)
        v4v3_r = pd.DataFrame(S4.T @ S3 / len(E), index=n4, columns=n3)
        v4v3_cos.to_csv(P("v4_v3b_cosines.csv"), float_format="%.4f")
        v4v3_r.to_csv(P("v4_v3b_score_r.csv"), float_format="%.4f")
        M4 = cos_mats[V4_SET]
        for i in range(len(M4)):
            for j in range(i + 1, len(M4)):
                if M4.values[i, j] >= V4_MERGE_COS:
                    merge_pairs.append((M4.index[i], M4.index[j], M4.values[i, j]))

    # --- Sub-axes ---
    sub = []
    for ver in V3_VERSIONS:
        for n in SUBAXIS_AXES:
            full = bykey.get(f"{ver}::{n}")
            L, N_ = bykey.get(f"{ver}@literal::{n}"), bykey.get(f"{ver}@non_literal::{n}")
            row = {"version": ver, "axis": n,
                   "literal": f"{len(L['pos'])} / {len(L['neg'])}"
                              + ("" if L["included"] else " (not built)"),
                   "non_literal": f"{len(N_['pos'])} / {len(N_['neg'])}"
                                  + ("" if N_["included"] else " (not built)")}
            if full["included"] and L["included"] and N_["included"]:
                row.update(
                    cos_lit_nonlit=float(L["vec"] @ N_["vec"]),
                    r_lit_nonlit=float(np.corrcoef(E @ L["vec"], E @ N_["vec"])[0, 1]),
                    cos_nonlit_full=float(N_["vec"] @ full["vec"]),
                    cos_lit_full=float(L["vec"] @ full["vec"]))
            sub.append(row)
    sub = pd.DataFrame(sub)

    # --- Save vectors, catalog, anchors, captions ---
    np.save(P("axis_vectors.npy"), AV)
    np.save(P("caption_embeddings.npy"), C)
    pd.DataFrame({"row": range(len(captions)), "caption": captions}).to_csv(
        P("captions.csv"), index=False)
    cat = pd.DataFrame([{k: s[k] for k in ("row", "key", "set", "kind", "version",
                                           "template", "axis")}
                        | {"n_pos": len(s["pos"]), "n_neg": len(s["neg"]),
                           "included": s["included"], "primary": s["primary"],
                           "note": s["note"]} for s in specs])
    cat.to_csv(P("axis_catalog.csv"), index=False)
    with open(P("axis_anchors.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["key", "pole", "anchor", "caption"])
        for s in specs:
            for pole, caps, ancs in (("positive", s["pos"], s["pos_anchors"]),
                                     ("negative", s["neg"], s["neg_anchors"])):
                for c, a in zip(caps, ancs):
                    w.writerow([s["key"], pole, a, c])

    # --- Figures ---
    fig_cosines([("v2 (9 axes)", cos_mats["v2"]),
                 (f"{PRIMARY_SET} (primary)", cos_mats[PRIMARY_SET])]
                + ([("v4", cos_mats[V4_SET])] if v4doc is not None else []),
                P("axis_cosines.png"))
    pairs = []
    for s in [x for x in specs if x["set"] == PRIMARY_SET and x["included"]]:
        s2 = "v2" if s["axis"] in v2k else "v2_merged"
        pairs.append((s["axis"], PRIMARY_SET, s["axis"], s2, s["axis"]))
    fig_mcurve(mc_df, pairs, P("stability_mcurve.png"))
    if v4doc is not None:
        fig_mcurve(mc_df, [(s["axis"], V4_SET, s["axis"], "-", s["axis"])
                           for s in specs if s["set"] == V4_SET and s["included"]],
                   P("stability_mcurve_v4.png"))

    # --- Report ---
    L = ["05_build_axes.py -- summary", "=" * 70,
         f"Date      : {time.strftime('%Y-%m-%d %H:%M')}",
         f"Inputs    : {V2_YAML}, {V3_YAML}, {TAGS_CSV}",
         f"Encoder   : {CLIP_MODEL_NAME} text tower + text_projection "
         "(un2CLIP does not change it)",
         f"Images    : {len(pick):,} random encoded images (stability only; "
         "features never read)", "",
         "CHECKS"]
    if checks.get("official") == "run":
        L += [f"  T1 tokens      : identical for all {len(captions)} captions; "
              f"longest {max_len} of 77 tokens",
              f"  T2 official    : min cos {checks['official_min_cos']:.7f}, "
              f"mean {checks['official_mean_cos']:.7f} -> PASS"]
    else:
        L += ["  T1/T2          : SKIPPED (clip package not installed)"]
    if isinstance(checks.get("old_axes"), dict):
        t3 = checks["old_axes"]
        L += [f"  T3 old axes    : {len(t3)} axes, min cos "
              f"{min(t3.values()):.7f} -> "
              f"{'PASS' if checks['old_axes_pass'] else 'WARN'}"]
        if not checks["old_axes_pass"]:
            L += [f"      {n}: {c:.6f}" for n, c in t3.items()]
            L += ["      The old results were built from axes that differ "
                  "from v2 as rebuilt here."]
    else:
        L += ["  T3 old axes    : skipped"]
    L += ["  T4 captions    : v3 captions equal the primary template"
          + ("; v4 template is the primary template" if v4doc is not None else ""),
          ""]
    if v4doc is not None:
        L.insert(4, f"v4        : {V4_YAML} (protocol {v4doc.get('protocol')})")

    L += ["AXIS SETS (included / listed)"]
    for (st, kind), g in cat.groupby(["set", "kind"], sort=False):
        L.append(f"  {st:<22} {kind:<9} {int(g.included.sum())} / {len(g)}")
    ex = cat[~cat.included]
    if len(ex):
        L += ["", "Not built (fewer than the minimum anchors):"]
        L += [f"  {r.set:<22} {r.axis:<44} {r.n_pos} / {r.n_neg}  {r.note}"
              for r in ex.itertuples()]

    L += ["", "ANCHOR COUNTS, main sets (positives / negatives)"]
    cnt = cat[cat.kind == "main"].pivot_table(
        index="axis", columns="set", values="n_pos", aggfunc="first")
    cnn = cat[cat.kind == "main"].pivot_table(
        index="axis", columns="set", values="n_neg", aggfunc="first")
    cols = [c for c in main_sets if c in cnt.columns]
    tbl = pd.DataFrame({c: [("-" if pd.isna(cnt.at[a, c]) else
                             f"{int(cnt.at[a, c])}/{int(cnn.at[a, c])}")
                            for a in cnt.index] for c in cols}, index=cnt.index)
    L += tbl.to_string().splitlines()

    L += ["", "AXIS COSINES (largest off-diagonal pairs)"]
    for ms in main_sets:
        M = cos_mats[ms]
        iu = np.triu_indices(len(M), 1)
        order = np.argsort(-M.values[iu])[:3]
        top = [f"{M.index[iu[0][k]].split(' / ')[0]}~"
               f"{M.index[iu[1][k]].split(' / ')[0]} {M.values[iu][k]:.2f}"
               for k in order]
        L.append(f"  {ms:<10} " + "; ".join(top))
    L += ["", f"  {PRIMARY_SET}:"]
    Mp = cos_mats[PRIMARY_SET].copy()
    Mp.index = Mp.columns = [n.split(" / ")[0][:12] for n in Mp.index]
    L += ["  " + x for x in Mp.round(2).to_string().splitlines()]

    L += ["", "MOVEMENT FROM v2 (cosine of axis vectors; r = correlation of "
          "image scores on the sample)"]
    for ver in V3_VERSIONS:
        L.append(f"  {ver}")
        for r in mov[mov.version == ver].itertuples():
            L.append(f"    {r.axis:<44} cos {r.cos_v2:.3f}  r {r.r_v2:.3f}"
                     f"  (vs {r.v2_counterpart})")

    L += ["", "STABILITY (medians over resamples; r = correlation of image "
          "scores; SB = Spearman-Brown)",
          "  split: both poles halved; pos/neg: one pole halved; LOO: leave "
          "one anchor out; boot p05: 5th percentile of bootstrap cosine"]
    for st in stab_sets:
        d = summ_df[summ_df.set == st]
        if not len(d):
            continue
        L.append(f"  {st}")
        for r in d.itertuples():
            g = lambda k: getattr(r, k, np.nan)
            L.append(f"    {r.axis[:34]:<34} {r.n_pos:>2}/{r.n_neg:<2} "
                     f"split cos {g('split_cos_med'):.2f} r {g('split_r_med'):.2f}"
                     f" (SB {g('split_r_sb_med'):.2f})  pos r "
                     f"{g('pos_split_r_med'):.2f}  neg r {g('neg_split_r_med'):.2f}"
                     f"  LOO min cos {g('loo_cos_min'):.3f}  boot p05 "
                     f"{g('boot_cos_p05'):.2f}")
            if isinstance(g("loo_worst_anchor"), str):
                L.append(f"    {'':<34}       most influential: "
                         f"{r.loo_worst_anchor}")

    L += ["", "BASELINES AND TEMPLATES (primary set): cosine with the main axis; "
          "r = correlation of image scores",
          "  centroid = positives vs the average of all captions (preferred "
          "negative-free baseline); neutral = positives vs 'a scene' (reference "
          "only); generic = cosine of the main axis with the direction from "
          "'a scene' to the average 'a scene that looks X' caption"]
    for r in base_df[base_df.set == PRIMARY_SET].itertuples():
        g = lambda k: getattr(r, k, np.nan)
        L.append(f"    {r.axis[:30]:<30} centroid {g('cos_centroid'):.2f} "
                 f"(r {g('r_centroid'):.2f})  neutral {g('cos_neutral'):.2f} "
                 f"(r {g('r_neutral'):.2f})  photo {g('cos_photo'):.2f}  place "
                 f"{g('cos_place'):.2f}  world {g('cos_world'):.2f}  generic "
                 f"{g('cos_generic_direction'):+.2f}")

    L += ["", "ANCHOR DIAGNOSTICS (primary set)"]
    dp = diag[diag.set == PRIMARY_SET]
    ws = dp[dp.wrong_side]
    L.append("  Reference: the centroid of all the set's captions. (The bare "
             "caption 'a scene' is not used: every 'a scene that looks X' "
             "caption differs from it in the same way; see BASELINES.)")
    L.append(f"  on the wrong side of the centroid: {len(ws)} of {len(dp)}"
             "  (value in brackets; value against 'a scene' after it)")
    L += [f"    {r.axis[:30]:<30} {r.pole[:3]}  {r.anchor}  "
          f"({r.own_side_vs_centroid:+.3f}; {r.own_side_vs_neutral:+.3f})"
          for r in ws.itertuples()]
    os_ = dp[dp.other_stronger]
    L.append(f"  loading more strongly on another axis than on their own: "
             f"{len(os_)} of {len(dp)}  (single captions are noisy; a screen "
             "for step 09, not a verdict)")
    L += [f"    {r.axis[:30]:<30} {r.pole[:3]}  {r.anchor}  -> "
          f"{r.strongest_other_axis.split(' / ')[0]}" for r in os_.itertuples()]

    L += ["", "SUB-AXES (literal / non-literal; positives / negatives)"]
    L += ["  " + x for x in sub.to_string(index=False,
                                          float_format="%.3f").splitlines()]

    if v4doc is not None:
        L += ["", "=" * 70, "v4 AXES (v4_protocol.md section 6.6)", "=" * 70,
              "Stability of v4 is in STABILITY above (set v4; drawn by pair; "
              "n_pos/n_neg = pairs). Reported, not used to drop an axis (S1a).",
              "", "PAIRS AND DOCUMENTS (reported instead of leave-one-document-out, L1c)"]
        for r in v4_tab.itertuples():
            L.append(f"  {r.axis[:40]:<40} pairs {r.pairs:>3}  distinct negatives "
                     f"{r.distinct_negatives:>3}  documents {r.documents}  largest "
                     f"{r.largest_document} {r.largest_document_share:.0%}")
        L += ["", "CONTROL DIRECTIONS (decision 17; cosine of the axis with the "
              "control) AND THE NOCONTROL VERSION (R2b)",
              "  enclosure = enclosed / cramped / closed room / indoor room minus "
              "open / spacious / wide open landscape / outdoors;",
              "  darkness = dark / dimly lit / nighttime / shadowy minus bright / "
              "brightly lit / daytime / sunlit.",
              "  nocontrol: both directions removed; cos and r (image scores) "
              "with the main axis"]
        for r in v4_tab.itertuples():
            g = lambda k: getattr(r, k, np.nan)
            L.append(f"  {r.axis[:40]:<40} enclosure {g('cos_enclosure'):+.3f}  "
                     f"darkness {g('cos_darkness'):+.3f}  nocontrol cos "
                     f"{g('cos_nocontrol'):.3f} r {g('r_nocontrol'):.3f}")
        L += ["", "BASELINES AND TEMPLATES (v4): cosine with the main axis; "
              "r = correlation of image scores"]
        for r in base_df[base_df.set == V4_SET].itertuples():
            g = lambda k: getattr(r, k, np.nan)
            L.append(f"    {r.axis[:30]:<30} centroid {g('cos_centroid'):.2f} "
                     f"(r {g('r_centroid'):.2f})  neutral {g('cos_neutral'):.2f} "
                     f"(r {g('r_neutral'):.2f})  photo {g('cos_photo'):.2f}  place "
                     f"{g('cos_place'):.2f}  world {g('cos_world'):.2f}  generic "
                     f"{g('cos_generic_direction'):+.2f}")
        L += ["", "AXIS COSINES (v4)"]
        M4 = cos_mats[V4_SET].copy()
        M4.index = M4.columns = [n[:3] for n in M4.index]
        L += ["  " + x for x in M4.round(2).to_string().splitlines()]
        L += [f"  Text half of the merge test (decision 13): pairs with cosine >= "
              f"{V4_MERGE_COS}: "
              + ("; ".join(f"{a[:3]}~{b[:3]} {c:.2f}" for a, b, c in merge_pairs)
                 if merge_pairs else "none")
              + ". A merge also needs image scores r >= 0.8 within categories (09)."]
        L += ["", f"v4 x {PRIMARY_SET} (no matching here: counterparts are decided "
              "in 09 by within-category image correlations, protocol 6.5)",
              "  highest |cosine| and highest |r| (image scores on the sample) "
              "per v4 axis:"]
        for a in v4v3_cos.index:
            jc = v4v3_cos.loc[a].abs().idxmax()
            jr = v4v3_r.loc[a].abs().idxmax()
            L.append(f"  {a[:34]:<34} cos {v4v3_cos.at[a, jc]:+.2f} "
                     f"{jc.split(' / ')[0][:20]:<20}  r {v4v3_r.at[a, jr]:+.2f} "
                     f"{jr.split(' / ')[0][:20]}")
        L += ["", "ANCHOR DIAGNOSTICS (v4; one row per pair)"]
        d4 = diag[diag.set == V4_SET]
        w4 = d4[d4.wrong_side]
        L.append(f"  on the wrong side of the centroid: {len(w4)} of {len(d4)}")
        L += [f"    {r.axis[:30]:<30} {r.pole[:3]}  {r.anchor}  "
              f"({r.own_side_vs_centroid:+.3f})" for r in w4.itertuples()]

    L += ["", f"Files in {OUTPUT_DIR}: axis_vectors.npy ({AV.shape[0]} axes), "
          "axis_catalog.csv, axis_anchors.csv, caption_embeddings.npy, "
          "captions.csv, anchor_diagnostics.csv, axis_cosines_*.csv, "
          "version_movement.csv, baseline_variants.csv, stability_*.csv, "
          "axis_cosines.png, "
          "stability_mcurve.png"
          + (", v4_axes.csv, v4_v3b_cosines.csv, v4_v3b_score_r.csv, "
             "stability_mcurve_v4.png" if v4doc is not None else ""),
          f"Elapsed: {(time.time() - t0) / 60:.1f} min"]
    report = "\n".join(L)
    with open(P("build_report.txt"), "w") as f:
        f.write(report + "\n")
    print("\n" + report)

    meta = {"script": "05_build_axes.py",
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "inputs": {"v2_yaml": V2_YAML, "v3_yaml": V3_YAML,
                       "tags_csv": TAGS_CSV, "emb": EMB_PATH,
                       "old_axes_dir": OLD_AXES_DIR, "v4_yaml": V4_YAML},
            "v4": None if v4doc is None else {
                "protocol": v4doc.get("protocol"), "set": V4_SET,
                "min_pairs": V4_MIN_PAIRS, "merge_cos": V4_MERGE_COS,
                "controls": V4_CONTROLS, "merge_pairs_text": merge_pairs},
            "clip_model_name": CLIP_MODEL_NAME, "primary_set": PRIMARY_SET,
            "v2_template": V2_TEMPLATE, "subaxis_axes": SUBAXIS_AXES,
            "n_images_sample": int(len(pick)), "n_splits": N_SPLITS,
            "n_boot": N_BOOT, "mcurve_draws": MCURVE_DRAWS,
            "seed": RANDOM_SEED, "allow_tf32": ALLOW_TF32,
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
            "n_axes": int(AV.shape[0]), "n_captions": len(captions),
            "checks": checks}
    with open(P("run_meta.json"), "w") as f:
        json.dump(meta, f, indent=2, default=float)


if __name__ == "__main__":
    main()
