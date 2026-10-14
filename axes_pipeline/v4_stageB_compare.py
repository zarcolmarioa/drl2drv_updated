"""
v4_stageB_compare.py  (v4 axes, Stage B: comparison and reconciliation)

Reads the two models' Stage B outputs (B1 groups, B2 assignments),
compares the two groupings, counts the support of every group, and runs
the reconciliation (v4_protocol.md, version 1.3, section 6). Never edits
the model outputs.

It works in three rounds; run it again after each of your steps.

ROUND 1, comparison (only the model outputs needed)
  - checks every B2 batch: texts missing, assigned twice, unknown, or
    assigned to a group the model did not define (listed; you can ask the
    model again for those texts and add the answer as an extra batch file,
    e.g. stageB_claude_B2_batch09.json);
  - agreement: adjusted Rand index (all texts; and texts both models put in
    a group), pairwise co-assignment, and NESTING: for every group, the
    share of its texts that the other model put in a single group (one
    model's groups can be finer than the other's; nesting shows it where
    the Rand index cannot);
  - a cross-walk of the groups (stageB_crosswalk.csv);
  - support of every group, counted from the PHRASES behind its texts:
    distinct texts, documents, first-person phrases, by sense class; the
    keeping rules with all phrases, vision only and not-specific only;
  - a starting point for the final axes: stageB_axes_TEMPLATE.csv.

ROUND 2, your axes (stageB_axes.csv)
  Copy the template to stageB_axes.csv and edit it: one row per final
  axis, with
    axis_id        A01, A02, ... (your choice)
    name, definition   one quality per axis (protocol 6)
    claude_groups, chatgpt_groups   the groups it takes from each model,
                   separated by ";" (e.g. G27;G28), or empty
    visual         yes / no: can the quality plausibly show in a scene?
    comment        why (logged)
  A group may belong to one axis only. Groups in no axis = "none".
  Run again: every text whose two models' groups lead to the same axis is
  assigned; texts where they lead to different axes (or one leads to an
  axis and the other to none) go to stageB_review.csv. With SKIP_NONVISUAL,
  a text whose options are only visual = no axes or none is not asked
  (axis "not_needed"): it cannot change a kept axis.

ROUND 3, the texts the models placed differently (stageB_review.csv)
  Write the axis_id (or none) in the decision column, with a comment if
  useful, and run again. Decisions are kept across runs (by text_id).
  When every row is decided, the final axes and their support are
  written, with the keeping rules applied.

Keeping rules (protocol 6, confirmed 5 October 2026): at least
MIN_TEXTS distinct texts, from at least MIN_DOCS documents, at least
MIN_FIRST_PERSON first-person phrase, visual = yes. Applied with all
phrases, with vision phrases only, and with not-specific phrases only.

Inputs (STAGEB_DIR)
-------------------
  stageB_texts.csv, stageB_phrase_map.csv, B2_batch_NN.txt   (prepare)
  outputs/stageB_<model>_B1.json, outputs/stageB_<model>_B2_batchNN.json

Outputs (STAGEB_DIR)
--------------------
  stageB_compare_report.txt     summary (start here)
  stageB_assignments.csv        every text with both models' groups
  stageB_groups_<model>.csv     every group: support, keeping rules
  stageB_crosswalk.csv          group x group counts, with nesting shares
  stageB_axes_TEMPLATE.csv      starting point for stageB_axes.csv
  stageB_review.csv             (round 2) texts to decide
  stageB_final_assignment.csv   (round 3) text -> axis, with how decided
  stageB_final_axes.csv         (round 3) every axis: support, rules
  stageB_problems.csv           format problems in the model outputs

Usage
-----
    conda activate drl2drv
    python v4_stageB_compare.py
"""

import glob
import json
import os
import re
import time
from collections import Counter

import numpy as np
import pandas as pd


# ============================================================
# SETTINGS
# ============================================================

STAGEB_DIR       = "axes/v4/stageB"
MODELS           = ["claude", "chatgpt"]
MIN_TEXTS        = 5
MIN_DOCS         = 2
MIN_FIRST_PERSON = 1
SCENARIOS        = {"all": {"vision", "not_specific", "disputed"},
                    "vision": {"vision"},
                    "not_specific": {"not_specific"}}
SPECIAL          = ["OTHER", "NOT_WORLD"]
NESTING_MIN      = 0.6     # report a group as nested in another at this share
SKIP_NONVISUAL   = True    # round 2-3: a disagreement whose options are only
                           # visual = no axes or none cannot change a kept
                           # axis; mark it "not needed" instead of asking


# ============================================================
# Helpers
# ============================================================

def P(*n):
    return os.path.join(STAGEB_DIR, *n)


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


def norm_group(g):
    """'G3', 'g03', 'G03 ' -> 'G03'; 'other', 'Not world' -> OTHER,
    NOT_WORLD; None / '' -> ''."""
    if g is None:
        return ""
    s = str(g).strip()
    if not s or s.lower() in ("null", "none"):
        return ""
    u = re.sub(r"[\s\-]+", "_", s.upper())
    if u in SPECIAL:
        return u
    m = re.match(r"^G_?0*(\d+)$", u)
    return f"G{int(m.group(1)):02d}" if m else s


def norm_tid(t):
    m = re.match(r"^\s*T0*(\d+)\s*$", str(t), flags=re.I)
    return f"T{int(m.group(1)):04d}" if m else str(t).strip()


def comb2(x):
    x = np.asarray(x, dtype=float)
    return x * (x - 1) / 2


def ari(a, b):
    """Adjusted Rand index of two labelings (lists of equal length)."""
    if len(a) < 2:
        return float("nan")
    ct = pd.crosstab(pd.Series(a), pd.Series(b)).to_numpy()
    s_ij = comb2(ct).sum()
    s_a, s_b = comb2(ct.sum(1)).sum(), comb2(ct.sum(0)).sum()
    tot = comb2(len(a))
    exp = s_a * s_b / tot
    mx = (s_a + s_b) / 2
    return float((s_ij - exp) / (mx - exp)) if mx != exp else float("nan")


def pair_agreement(a, b):
    """Of the pairs placed together by A, the share also together in B, and
    the reverse."""
    ct = pd.crosstab(pd.Series(a), pd.Series(b)).to_numpy()
    s_ij = comb2(ct).sum()
    s_a, s_b = comb2(ct.sum(1)).sum(), comb2(ct.sum(0)).sum()
    return (s_ij / s_a if s_a else float("nan"),
            s_ij / s_b if s_b else float("nan"))


# ============================================================
# Main
# ============================================================

def main():
    t0 = time.time()
    print("=" * 70)
    print("v4_stageB_compare.py")
    print("=" * 70)
    problems = []

    def problem(kind, model="", detail=""):
        problems.append({"kind": kind, "model": model, "detail": detail})

    tx = pd.read_csv(P("stageB_texts.csv"), dtype={"text_id": str})
    pm = pd.read_csv(P("stageB_phrase_map.csv"), dtype=str).fillna("")
    pm = pm[pm["included"].astype(str).str.lower() == "true"].copy()
    all_ids = list(tx["text_id"])
    id_set = set(all_ids)
    texts = dict(zip(tx.text_id, tx.text))

    # batches as given to the models
    given = {}
    for f in sorted(glob.glob(P("B2_batch_*.txt"))):
        k = int(re.search(r"(\d+)\.txt$", f).group(1))
        ids = re.findall(r"^(T\d{4}) \|", open(f).read(), flags=re.M)
        for t in ids:
            given[t] = k
    if set(given) != id_set:
        raise SystemExit("The B2 batch files do not cover stageB_texts.csv; "
                         "rerun v4_stageB_prepare.py?")

    # --- model outputs ---
    groups, assign = {}, {}
    for m in MODELS:
        f1 = P("outputs", f"stageB_{m}_B1.json")
        if not os.path.exists(f1):
            raise SystemExit(f"Missing {f1}")
        g = {}
        for x in parse_json(f1):
            gid = norm_group(x.get("group"))
            if not re.match(r"^G\d{2}$", gid):
                problem("bad_group_id", m, repr(x.get("group")))
                continue
            if gid in g:
                problem("duplicate_group", m, gid)
            g[gid] = {"name": x.get("name", ""), "definition": x.get("definition", ""),
                      "examples": [norm_tid(e) for e in x.get("examples", [])]}
            bad = [e for e in g[gid]["examples"] if e not in id_set]
            if bad:
                problem("unknown_example", m, f"{gid}: {bad}")
        groups[m] = g
        files = sorted(glob.glob(P("outputs", f"stageB_{m}_B2_batch*.json")))
        if not files:
            raise SystemExit(f"No B2 files for {m} in {P('outputs')}")
        a = {}
        seen = Counter()
        for f in files:
            for x in parse_json(f):
                if not isinstance(x, dict):
                    problem("not_an_object", m, f"{os.path.basename(f)}: {x!r}"[:200])
                    continue
                t = norm_tid(x.get("id", x.get("text_id", "")))
                gp, al = norm_group(x.get("group")), norm_group(x.get("also"))
                if t not in id_set:
                    problem("unknown_text", m, f"{os.path.basename(f)}: {x.get('id')!r}")
                    continue
                seen[t] += 1
                if gp not in g and gp not in SPECIAL:
                    problem("unknown_group", m, f"{t}: {x.get('group')!r}")
                    gp = "INVALID"
                if al and al not in g:
                    problem("unknown_also", m, f"{t}: {x.get('also')!r}")
                    al = ""
                if t in a:
                    if a[t][0] != gp:
                        problem("assigned_twice_differently", m,
                                f"{t}: {a[t][0]} then {gp} (first kept)")
                    continue
                a[t] = (gp, al)
        missing = [t for t in all_ids if t not in a]
        for t in missing:
            problem("missing_text", m, f"{t} (batch {given[t]})")
        for t, c in seen.items():
            if c > 1:
                problem("assigned_more_than_once", m, f"{t}: {c} times")
        assign[m] = a
        print(f"  {m}: {len(g)} groups, {len(files)} B2 files, "
              f"{len(a)} of {len(all_ids)} texts assigned, {len(missing)} missing")

    # --- per-text table ---
    A = pd.DataFrame({"text_id": all_ids, "text": [texts[t] for t in all_ids],
                      "batch": [given[t] for t in all_ids]})
    A = A.merge(tx.drop(columns="text"), on="text_id", how="left")
    for m in MODELS:
        A[f"group_{m}"] = [assign[m].get(t, ("MISSING", ""))[0] for t in all_ids]
        A[f"also_{m}"] = [assign[m].get(t, ("", ""))[1] for t in all_ids]
        A[f"name_{m}"] = [groups[m].get(gp, {}).get("name", gp)
                          for gp in A[f"group_{m}"]]
    A.to_csv(P("stageB_assignments.csv"), index=False)

    # --- agreement ---
    m0, m1 = MODELS
    ok = (~A[f"group_{m0}"].isin(["MISSING", "INVALID"]) &
          ~A[f"group_{m1}"].isin(["MISSING", "INVALID"]))
    a0, a1 = A.loc[ok, f"group_{m0}"].tolist(), A.loc[ok, f"group_{m1}"].tolist()
    both_g = ok & A[f"group_{m0}"].str.match(r"^G\d") & A[f"group_{m1}"].str.match(r"^G\d")
    g0, g1 = A.loc[both_g, f"group_{m0}"].tolist(), A.loc[both_g, f"group_{m1}"].tolist()
    agree = {"ari_all": ari(a0, a1), "ari_grouped": ari(g0, g1),
             "n_all": int(ok.sum()), "n_grouped": int(both_g.sum())}
    agree["pairs_0_in_1"], agree["pairs_1_in_0"] = pair_agreement(g0, g1)
    spec = pd.crosstab(A.loc[ok, f"group_{m0}"].where(
        A.loc[ok, f"group_{m0}"].isin(SPECIAL), "group"),
        A.loc[ok, f"group_{m1}"].where(A.loc[ok, f"group_{m1}"].isin(SPECIAL), "group"))

    # --- cross-walk and nesting ---
    ct = pd.crosstab(A.loc[ok, f"group_{m0}"], A.loc[ok, f"group_{m1}"])
    cw = []
    for gi in ct.index:
        for gj in ct.columns:
            n = int(ct.at[gi, gj])
            if n:
                cw.append({f"group_{m0}": gi, f"name_{m0}": groups[m0].get(gi, {}).get("name", gi),
                           f"group_{m1}": gj, f"name_{m1}": groups[m1].get(gj, {}).get("name", gj),
                           "n_texts": n,
                           f"share_of_{m0}_group": n / ct.loc[gi].sum(),
                           f"share_of_{m1}_group": n / ct[gj].sum()})
    cw = pd.DataFrame(cw).sort_values("n_texts", ascending=False)
    cw.to_csv(P("stageB_crosswalk.csv"), index=False, float_format="%.3f")

    def best(m_from, m_to):
        """For each group of m_from: its best group in m_to and the share."""
        out = {}
        c = ct if m_from == m0 else ct.T
        for gi in c.index:
            row = c.loc[gi]
            out[gi] = (row.idxmax(), float(row.max() / row.sum()) if row.sum() else 0.0)
        return out
    best01, best10 = best(m0, m1), best(m1, m0)

    # --- support of groups ---
    pm_t = pm.merge(A[["text_id"] + [f"group_{m}" for m in MODELS]], on="text_id",
                    how="left")

    def support(sub):
        """sub: phrases of a group. Support per scenario."""
        out = {}
        for sc, classes in SCENARIOS.items():
            s = sub[sub.sense_class.isin(classes)]
            nt, nd = s.text_id.nunique(), s.doc_id.nunique()
            nf = int((s.first_person == "yes").sum())
            out[sc] = (nt, nd, nf, nt >= MIN_TEXTS and nd >= MIN_DOCS and
                       nf >= MIN_FIRST_PERSON)
        cls = sub.drop_duplicates("text_id").merge(tx[["text_id", "text_class"]],
                                                    on="text_id").text_class.value_counts()
        return out, cls

    def group_table(m):
        rows = []
        for gid in list(groups[m]) + SPECIAL:
            sub = pm_t[pm_t[f"group_{m}"] == gid]
            sp, cls = support(sub)
            ntx = sub.text_id.nunique()
            r = {"group": gid, "name": groups[m].get(gid, {}).get("name", gid),
                 "definition": groups[m].get(gid, {}).get("definition", ""),
                 "n_texts": ntx, "n_phrases": len(sub),
                 "texts_vision": int(cls.get("vision", 0)),
                 "texts_not_specific": int(cls.get("not_specific", 0)),
                 "texts_mixed": int(cls.get("mixed", 0)),
                 "share_vision": (int(cls.get("vision", 0)) / ntx) if ntx else np.nan}
            for sc, (nt, nd, nf, keep) in sp.items():
                r.update({f"{sc}_texts": nt, f"{sc}_docs": nd,
                          f"{sc}_first_person": nf, f"{sc}_passes": keep})
            if gid in groups[m]:
                bb = (best01 if m == m0 else best10).get(gid)
                if bb:
                    r["best_match_other_model"] = bb[0]
                    r["share_in_best_match"] = bb[1]
            rows.append(r)
        return pd.DataFrame(rows)
    gt = {m: group_table(m) for m in MODELS}
    for m in MODELS:
        gt[m].to_csv(P(f"stageB_groups_{m}.csv"), index=False, float_format="%.3f")

    # --- template for the final axes ---
    # One row per m0 group. An m1 group joins the row of its best m0 group
    # when the match is mutual or when the m1 group is nested in it (>=
    # NESTING_MIN of its texts there); otherwise it gets a row of its own.
    # m0 groups nested in the same m1 group are flagged in the note (you
    # may want to merge them). Only a starting point: edit freely.
    rows0 = {gi: {"m1": [], "note": []} for gi in groups[m0]}
    alone1 = []
    for gj in groups[m1]:
        if gj not in best10 or best10[gj][0] not in rows0:
            alone1.append((gj, "no texts in a shared group"))
            continue
        gi, sh = best10[gj]
        mutual = gi in best01 and best01[gi][0] == gj
        if mutual or sh >= NESTING_MIN:
            rows0[gi]["m1"].append(gj)
            rows0[gi]["note"].append(f"{m1} {gj} " + ("mutual best match" if mutual
                                     else "nested") + f" ({sh:.0%} of its texts here)")
        else:
            alone1.append((gj, f"best match {m0} {gi} ({sh:.0%}), not mutual"))
    tmpl = []
    for gi in groups[m0]:
        note = list(rows0[gi]["note"])
        if gi in best01:
            gj, sh = best01[gi]
            if gj not in rows0[gi]["m1"] and sh >= NESTING_MIN:
                note.append(f"nested in {m1} {gj} ({sh:.0%}), which is in another row:"
                            " merge?")
        tmpl.append({f"{m0}_groups": gi, f"{m1}_groups": ";".join(rows0[gi]["m1"]),
                     "name": groups[m0][gi]["name"],
                     "definition": groups[m0][gi]["definition"],
                     "note": "; ".join(note) or f"no {m1} group"})
    for gj, why in alone1:
        tmpl.append({f"{m0}_groups": "", f"{m1}_groups": gj,
                     "name": groups[m1][gj]["name"],
                     "definition": groups[m1][gj]["definition"], "note": why})
    tm = pd.DataFrame(tmpl)
    tm.insert(0, "axis_id", [f"A{i + 1:02d}" for i in range(len(tm))])
    tm["visual"] = ""
    tm["comment"] = ""
    tm[["axis_id", "name", "definition", f"{m0}_groups", f"{m1}_groups", "visual",
        "comment", "note"]].to_csv(P("stageB_axes_TEMPLATE.csv"), index=False)

    # ========================================================
    # Rounds 2 and 3: the final axes
    # ========================================================
    final = None
    review_info = {}
    axes_path = P("stageB_axes.csv")
    if os.path.exists(axes_path):
        ax = pd.read_csv(axes_path, dtype=str).fillna("")
        need = {"axis_id", "name", "definition", f"{m0}_groups", f"{m1}_groups",
                "visual"}
        if not need <= set(ax.columns):
            raise SystemExit(f"stageB_axes.csv needs the columns {sorted(need)}")
        ax["axis_id"] = ax["axis_id"].str.strip()
        if ax.axis_id.duplicated().any():
            raise SystemExit("stageB_axes.csv: duplicate axis_id")
        to_axis = {m: {} for m in MODELS}
        for r in ax.itertuples():
            for m in MODELS:
                for gid in str(getattr(r, f"{m}_groups")).replace(",", ";").split(";"):
                    gid = norm_group(gid)
                    if not gid:
                        continue
                    if gid not in groups[m]:
                        problem("axes_unknown_group", m, f"{r.axis_id}: {gid}")
                        continue
                    if gid in to_axis[m] and to_axis[m][gid] != r.axis_id:
                        raise SystemExit(f"stageB_axes.csv: {m} {gid} is in two axes "
                                         f"({to_axis[m][gid]}, {r.axis_id})")
                    to_axis[m][gid] = r.axis_id
        bad_vis = ax[~ax.visual.str.lower().isin(["yes", "no"])]
        axis_ids = list(ax.axis_id)
        # an axis counts as non-visual only when visual is exactly "no"
        # (a missing or invalid value keeps its texts in the review)
        nonvis = set(ax.axis_id[ax.visual.str.strip().str.lower() == "no"]) | {"none"}

        # texts: agreed or to review
        old = {}
        if os.path.exists(P("stageB_review.csv")):
            o = pd.read_csv(P("stageB_review.csv"), dtype=str).fillna("")
            old = {r.text_id: (r.decision.strip(), r.comment) for r in o.itertuples()}
        fa, rv = [], []
        for r in A.itertuples():
            ax0 = to_axis[m0].get(getattr(r, f"group_{m0}"), "none")
            ax1 = to_axis[m1].get(getattr(r, f"group_{m1}"), "none")
            al0 = to_axis[m0].get(getattr(r, f"also_{m0}"), "")
            al1 = to_axis[m1].get(getattr(r, f"also_{m1}"), "")
            if ax0 == ax1:
                final_ax, how = ax0, "agreed"
            elif ax0 != "none" and ax0 == al1:
                final_ax, how = ax0, f"agreed ({m1} second choice)"
            elif ax1 != "none" and ax1 == al0:
                final_ax, how = ax1, f"agreed ({m0} second choice)"
            elif (SKIP_NONVISUAL and {ax0, ax1} <= nonvis
                  and not old.get(r.text_id, ("", ""))[0]):
                # no visual axis among the options: no decision needed
                final_ax, how = "not_needed", "not needed (no visual axis)"
            else:
                dec, com = old.get(r.text_id, ("", ""))
                valid = dec in axis_ids or dec.lower() == "none"
                final_ax = (dec if dec in axis_ids else "none") if valid else "undecided"
                how = "review" if valid else "undecided"
                rv.append({"text_id": r.text_id, "text": r.text,
                           f"axis_{m0}": ax0, f"group_{m0}": getattr(r, f"group_{m0}"),
                           f"name_{m0}": getattr(r, f"name_{m0}"),
                           f"axis_{m1}": ax1, f"group_{m1}": getattr(r, f"group_{m1}"),
                           f"name_{m1}": getattr(r, f"name_{m1}"),
                           "n_phrases": r.n_phrases, "text_class": r.text_class,
                           "options": f"{ax0} / {ax1} / none / another axis_id",
                           "decision": dec, "comment": com,
                           "decision_valid": valid or not dec})
            fa.append({"text_id": r.text_id, "text": r.text, "axis_id": final_ax,
                       "how": how, f"axis_{m0}": ax0, f"axis_{m1}": ax1,
                       "text_class": r.text_class})
        fa = pd.DataFrame(fa)
        rvd = pd.DataFrame(rv)
        pairs = pd.Series(dtype=int)
        if len(rvd):
            # rows sorted so that the same disagreement (axis pair) sits together,
            # most frequent pair first
            pairs = rvd.groupby([f"axis_{m0}", f"axis_{m1}"]).size().sort_values(
                ascending=False)
            rvd["pair_count"] = [pairs[(a, b)] for a, b in
                                 zip(rvd[f"axis_{m0}"], rvd[f"axis_{m1}"])]
            rvd = rvd.sort_values(["pair_count", f"axis_{m0}", f"axis_{m1}", "text_id"],
                                  ascending=[False, True, True, True])
            rvd.drop(columns="decision_valid").to_csv(P("stageB_review.csv"), index=False)
        open_rows = int((fa.how == "undecided").sum())
        bad_dec = int((~rvd.decision_valid).sum()) if len(rvd) else 0
        fa.to_csv(P("stageB_final_assignment.csv"), index=False)
        review_info = {"n_review": len(rvd), "open": open_rows, "invalid": bad_dec,
                       "agreed": int(fa.how.str.startswith("agreed").sum()),
                       "not_needed": int((fa.axis_id == "not_needed").sum()),
                       "bad_visual": list(bad_vis.axis_id)}

        # final axes support
        pm_f = pm.merge(fa[["text_id", "axis_id"]], on="text_id", how="left")
        rows = []
        for r in ax.itertuples():
            sub = pm_f[pm_f.axis_id == r.axis_id]
            sp, cls = support(sub)
            ntx = sub.text_id.nunique()
            vis = r.visual.strip().lower() == "yes"
            row = {"axis_id": r.axis_id, "name": r.name, "definition": r.definition,
                   f"{m0}_groups": getattr(r, f"{m0}_groups"),
                   f"{m1}_groups": getattr(r, f"{m1}_groups"),
                   "visual": r.visual, "n_texts": ntx, "n_phrases": len(sub),
                   "texts_vision": int(cls.get("vision", 0)),
                   "texts_not_specific": int(cls.get("not_specific", 0)),
                   "texts_mixed": int(cls.get("mixed", 0)),
                   "share_vision": (int(cls.get("vision", 0)) / ntx) if ntx else np.nan}
            for sc, (nt, nd, nf, keep) in sp.items():
                row.update({f"{sc}_texts": nt, f"{sc}_docs": nd,
                            f"{sc}_first_person": nf,
                            f"{sc}_kept": bool(keep and vis)})
            rows.append(row)
        final = pd.DataFrame(rows)
        final.to_csv(P("stageB_final_axes.csv"), index=False, float_format="%.3f")

    pr = pd.DataFrame(problems, columns=["kind", "model", "detail"])
    pr.to_csv(P("stageB_problems.csv"), index=False)

    # ========================================================
    # Report
    # ========================================================
    L = []
    W = L.append
    W("STAGE B: COMPARISON AND RECONCILIATION (v4_protocol.md 1.3, section 6)")
    W("=" * 78)
    W(f"Date: {time.strftime('%Y-%m-%d %H:%M')}")
    W(f"Texts: {len(all_ids):,} ({len(pm):,} phrases); groups: "
      + ", ".join(f"{m} {len(groups[m])}" for m in MODELS))
    W("")
    W("1. OUTPUT CHECKS (stageB_problems.csv)")
    W("-" * 78)
    if len(pr):
        for (k, m), n in pr.groupby(["kind", "model"]).size().items():
            W(f"  {m:<9}{k:<30}{n:>5}")
        W("  Missing texts can be asked again in the same chat; save the answer as")
        W("  an extra batch file (e.g. stageB_<model>_B2_batch09.json).")
    else:
        W("  no problems: every text assigned exactly once, to a defined group")
    W("")
    W("2. HOW THE TEXTS WERE PLACED")
    W("-" * 78)
    for m in MODELS:
        vc = A[f"group_{m}"].where(A[f"group_{m}"].str.match(r"^G\d"), A[f"group_{m}"])
        n_g = int(A[f"group_{m}"].str.match(r"^G\d").sum())
        W(f"  {m:<9} in a group {n_g:>5}   OTHER {int((vc == 'OTHER').sum()):>4}   "
          f"NOT_WORLD {int((vc == 'NOT_WORLD').sum()):>4}   with a second group "
          f"{int((A[f'also_{m}'] != '').sum()):>4}")
    W(f"  group / OTHER / NOT_WORLD, {m0} (rows) x {m1} (columns):")
    for line in spec.to_string().splitlines():
        W("    " + line)
    W("")
    W("3. AGREEMENT")
    W("-" * 78)
    W(f"  adjusted Rand index, all texts (OTHER, NOT_WORLD as categories): "
      f"{agree['ari_all']:.2f} (n = {agree['n_all']})")
    W(f"  adjusted Rand index, texts both models put in a group:           "
      f"{agree['ari_grouped']:.2f} (n = {agree['n_grouped']})")
    W(f"  pairs together in {m0} also together in {m1}: {agree['pairs_0_in_1']:.0%}; "
      f"pairs together in {m1} also together in {m0}: {agree['pairs_1_in_0']:.0%}")
    W("  (The Rand index is low when one model's groups are finer than the "
      "other's even if")
    W("   they agree; nesting below shows that case.)")
    for mf, bb in ((m0, best01), (m1, best10)):
        sh = np.array([v[1] for k, v in bb.items() if k in groups[mf]])
        W(f"  nesting, {mf} groups in one {m1 if mf == m0 else m0} group: median share "
          f"{np.median(sh):.0%}; groups with >= {NESTING_MIN:.0%} of their texts in "
          f"one group: {int((sh >= NESTING_MIN).sum())} of {len(sh)}")
    W("")
    W(f"4. CROSS-WALK ({m0} -> {m1}; best match and share; stageB_crosswalk.csv)")
    W("-" * 78)
    t0_ = gt[m0][gt[m0].group.str.match(r"^G\d")]
    for r in t0_.itertuples():
        bj = getattr(r, "best_match_other_model", "")
        W(f"  {r.group} {str(r.name)[:30]:<30}{r.n_texts:>5} -> {bj:<9} "
          f"{str(groups[m1].get(bj, {}).get('name', bj))[:28]:<28} "
          f"{getattr(r, 'share_in_best_match', 0):>4.0%}")
    W("")
    W("5. SUPPORT OF THE GROUPS (keeping rules: >= "
      f"{MIN_TEXTS} texts, >= {MIN_DOCS} documents, >= {MIN_FIRST_PERSON} "
      "first-person phrase;")
    W("   pass with all phrases / vision only / not-specific only; vision = "
      "share of vision texts)")
    W("-" * 78)
    for m in MODELS:
        g = gt[m][gt[m].group.str.match(r"^G\d")]
        W(f"  {m}: {int(g.all_passes.sum())} of {len(g)} groups pass with all "
          f"phrases, {int(g.vision_passes.sum())} with vision only, "
          f"{int(g.not_specific_passes.sum())} with not-specific only")
        W(f"    {'group':<6}{'name':<31}{'texts':>6}{'docs':>5}{'1st':>5}"
          f"{'vision':>7}   all/vis/n-s")
        for r in g.itertuples():
            fl = "/".join("Y" if getattr(r, f"{sc}_passes") else "-"
                          for sc in SCENARIOS)
            W(f"    {r.group:<6}{str(r.name)[:30]:<31}{r.all_texts:>6}{r.all_docs:>5}"
              f"{r.all_first_person:>5}{r.share_vision:>7.0%}   {fl}")
    W("")
    W("6. FINAL AXES (stageB_axes.csv)")
    W("-" * 78)
    if final is None:
        W(f"  Template: {len(tm)} rows ({len(groups[m0])} {m0} groups with the "
          f"{m1} groups matched or nested in them; {len(tm) - len(groups[m0])} "
          f"{m1} groups on their own).")
        W("  Not yet: copy stageB_axes_TEMPLATE.csv to stageB_axes.csv, edit it")
        W("  (one quality per axis; visual = yes / no; comment), and run again.")
    else:
        ri = review_info
        W(f"  {len(final)} axes; texts agreed {ri['agreed']:,}; texts to review "
          f"{ri['n_review']:,} (stageB_review.csv), open {ri['open']:,}"
          + (f", invalid decisions {ri['invalid']}" if ri["invalid"] else ""))
        if ri["not_needed"]:
            W(f"  not needed: {ri['not_needed']:,} texts the models placed differently "
              "but whose only options are visual = no")
            W("  axes or none; not asked (SKIP_NONVISUAL), so the counts of the "
              "visual = no axes are incomplete.")
        if ri["bad_visual"]:
            W(f"  visual must be yes or no: {', '.join(ri['bad_visual'])}")
        if len(pairs):
            W(f"  Most frequent disagreements ({m0} axis vs {m1} axis; a frequent pair")
            W("  may mean two axes should be one, or a group belongs in another axis:")
            W("  then edit stageB_axes.csv rather than deciding text by text):")
            nm = {**dict(zip(ax.axis_id, ax.name)), "none": "(no axis)"}
            for (a, b), n in pairs.head(12).items():
                W(f"    {n:>4}  {a:<5}{str(nm.get(a, a))[:28]:<29} vs {b:<5}"
                  f"{str(nm.get(b, b))[:28]}")
        if ri["open"]:
            W("  Decide the open rows of stageB_review.csv (an axis_id or none) and "
              "run again; the numbers below change until then.")
        W(f"    {'axis':<5}{'name':<31}{'vis':>4}{'texts':>6}{'docs':>5}{'1st':>5}"
          f"{'vision':>7}   kept all/vis/n-s")
        for r in final.itertuples():
            fl = "/".join("Y" if getattr(r, f"{sc}_kept") else "-" for sc in SCENARIOS)
            sv = f"{r.share_vision:>7.0%}" if r.all_texts else f"{'-':>7}"
            W(f"    {r.axis_id:<5}{str(r.name)[:30]:<31}{r.visual[:3]:>4}{r.all_texts:>6}"
              f"{r.all_docs:>5}{r.all_first_person:>5}{sv}   {fl}")
        k = final[[f"{sc}_kept" for sc in SCENARIOS]].sum()
        W(f"  kept: all phrases {int(k.iloc[0])}, vision only {int(k.iloc[1])}, "
          f"not-specific only {int(k.iloc[2])}")
        if not ri["open"] and not ri["invalid"] and not ri["bad_visual"]:
            W("  STAGE B COMPLETE: every text decided. Kept axes go to Stage C.")
    W("")
    W(f"Runtime {time.time() - t0:.1f} s")
    open(P("stageB_compare_report.txt"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    json.dump({"script": "v4_stageB_compare.py", "protocol": "1.3",
               "created": time.strftime("%Y-%m-%d %H:%M:%S"),
               "agreement": agree, "review": review_info,
               "rules": {"min_texts": MIN_TEXTS, "min_docs": MIN_DOCS,
                         "min_first_person": MIN_FIRST_PERSON},
               "n_groups": {m: len(groups[m]) for m in MODELS}},
              open(P("compare_meta.json"), "w"), indent=2, default=str)


if __name__ == "__main__":
    main()
