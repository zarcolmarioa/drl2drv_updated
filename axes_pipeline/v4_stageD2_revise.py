"""
v4_stageD2_revise.py  (v4 axes, Stage D2: manual revision of the final pairs)

Revises the 232 pairs of Stage D (axes/v4/stageD/axes_v4.yaml) before any
image is scored (v4_protocol.md 1.12, section 6.4a):

  1. DROP  pairs whose positive is not about how a scene looks (the
           not_scene rule of Stage C applied to every pair: touch or action,
           time, the body, a judgement, a story or reference, emotion, the
           self-world relation) and every pair whose positive contains a
           negation (the 9 kept by decision 15).
  2. REPLACE  negatives that bring in content of their own (stone, a town,
           a bad day, people, light). The new negative is another candidate
           of Stage D that passed the filters and does not add the avoided
           words, or your own negative.
  3. A05 (decision 17, R1b)  every A05 negative that adds a word for open
           places or for light is replaced the same way; candidates with
           "direct" are preferred, "clear" (A07's counterpart) is avoided.
  No new negative (2 or 3) may add a word for a place, the weather, a
  material, people or light (GLOBAL_AVOID). The proposals are a starting
  point: you read every replaced row.
  4. CHECK  borderline pairs, kept unless you decide otherwise.

The earlier rules on negatives still apply to every new negative:
  - only candidates that passed the Stage D filters can be chosen; your own
    negative must pass the same filters (no negation, same form, at most 2
    content words changed, length within 2 words, a content word of the
    positive replaced; decisions 4, 14 and Prompt D);
  - among the allowed candidates the order is: a preferred word, fewest
    convention conflicts (a word owned by another axis by the conventions of
    1.10: "real"/"reality" belong to A01, "clear" to A07), then the order of
    Stage D (fewest flags, both models, consistency, the models' rank);
  - the flags of Stage D (grammar, generic, shared word, clip_close,
    crowded) are shown with every candidate.

How to use
----------
1. Run the script. It writes stageD2_revision.csv, one row per pair, with
   the proposed action, the reason, the proposed negative and the allowed
   alternatives.
2. Read every row whose proposed action is not "keep" (and the "check"
   rows). Write a decision only where you disagree:
       (empty)    the proposed action
       keep       keep the Stage D pair as it is
       drop       drop the pair
       K3         candidate K3 of Stage D (must have passed the filters)
       any text   your own negative (checked by the filters), or the text
                  of a candidate
   Rows marked "open" (no allowed candidate) need a decision.
3. Run again. Your decisions are kept. When nothing is open and there are
   no problems, the revised axes are written to OUTPUT_DIR/axes_v4.yaml.

Inputs
------
  STAGED_DIR/axes_v4.yaml, STAGED_DIR/stageD_candidates.csv
  STAGEB_DIR/stageB_phrase_map.csv, STAGEB_DIR/stageB_clip_selection.csv

Outputs (OUTPUT_DIR, a new folder)
----------------------------------
  stageD2_revision.csv   one row per Stage D pair (your decisions go here)
  stageD2_pairs.csv      the final pairs with how each was decided
  stageD2_axes.csv       pairs and support per axis (keeping rules)
  stageD2_report.txt     the summary
  axes_v4.yaml           the revised v4 axes (when complete)

Usage
-----
    conda activate drl2drv
    python v4_stageD2_revise.py
"""

import difflib
import os
import re
import time
import unicodedata
from collections import Counter

import pandas as pd


# ============================================================
# SETTINGS
# ============================================================

STAGEB_DIR = "axes/v4/stageB"
STAGED_DIR = "axes/v4/stageD"
OUTPUT_DIR = "axes/v4/stageD2"
PROTOCOL   = "1.12"

# 1. Pairs dropped (reason, note). Negation: the 9 pairs kept by decision 15.
DROP = {
    "A01-04": ("negation", "don't; also the body (limbs)"),
    "A01-10": ("negation", "without"),
    "A01-13": ("negation", "nothing"),
    "A01-16": ("negation", "not"),
    "A01-39": ("negation", "nothing"),
    "A01-47": ("negation", "not; also time (as before)"),
    "A07-14": ("negation", "never"),
    "A14-03": ("negation", "no"),
    "A14-05": ("negation", "nothing"),
    "A01-20": ("not_scene: time", "the present"),
    "A01-44": ("not_scene: body", "under the feet"),
    "A01-45": ("not_scene: touch or action", "a zipper to unzip"),
    "A01-51": ("not_scene: judgement", "half-true"),
    "A01-52": ("not_scene: judgement", "pointless"),
    "A02-14": ("not_scene: judgement", "could only be a dream"),
    "A02-15": ("not_scene: judgement", "must be a dream (as A02-14)"),
    "A02-25": ("not_scene: story or reference", "Inception"),
    "A03-08": ("not_scene: story or reference", "a fictional narrative"),
    "A03-13": ("not_scene: story or reference", "Truman Show plot"),
    "A04-20": ("not_scene: life", "playing the life game"),
    "A04-22": ("not_scene: life", "the life game"),
    "A05-14": ("not_scene: self-world", "a barrier that separates me from reality"),
    "A05-41": ("not_scene: self-world", "bodies wrapped in an isolating substance"),
    "A13-06": ("not_scene: emotion", "perceptual or emotional vividness"),
}

# 2. Negatives replaced because they bring in content of their own
#    (reason, words the new negative may not add)
PEOPLE = {"populated", "inhabited", "people", "crowd", "crowded", "busy", "bustling",
          "activity", "lively", "occupied"}
LIGHT = {"light", "lit", "bright", "daylight", "sunlit", "sunny", "illuminated", "well-lit"}
REPLACE = {
    "A01-26": ("negative brings in a material", {"earth", "stone", "rock", "brick", "concrete"}),
    "A01-46": ("negative brings in a material", {"stone", "rock", "brick", "concrete", "earth"}),
    "A03-03": ("negative brings in a town", {"town", "day", "street", "city", "square"}),
    "A03-04": ("negative brings in a town", {"town", "square", "street", "city", "plaza"}),
    "A03-12": ("negative brings in a town", {"neighbourhood", "neighborhood", "town", "street",
                                             "city"}),
    "A04-07": ("negative brings in a mood", {"day", "bad"}),
    "A05-06": ("negative brings in light", LIGHT),
    "A14-01": ("negative brings in people", PEOPLE),
    "A14-09": ("negative brings in people", PEOPLE),
}

# Words no new negative may add, in any axis: places, weather, materials,
# people and light are scene content of their own (the problem of 2 and 3)
SCENE_CONTENT = {"park", "town", "city", "street", "square", "plaza", "village",
                 "neighbourhood", "neighborhood", "room", "house", "home", "building",
                 "garden", "forest", "beach", "sea", "mountain", "sky", "sun", "snow",
                 "frost", "ice", "winter", "summer", "rain", "earth", "stone", "rock", "brick",
                 "concrete", "wood"}
GLOBAL_AVOID = SCENE_CONTENT | PEOPLE | LIGHT

# 3. Decision 17 (R1b): A05 negatives adding any of these words are replaced
AXIS_AVOID = {
    "A05": {"open", "opened", "air", "space", "opening", "doorway", "door", "gate", "gateway",
            "entrance",
            "clearing", "meadow", "field", "landscape", "outdoors", "outside",
            "window"} | LIGHT,
}
PREFER = {"A05": {"direct", "directly"}}

# Words owned by another axis by the conventions of 1.10 (avoided, not banned)
CONVENTION = {"A02": {"real", "reality"}, "A03": {"real", "reality"},
              "A04": {"real", "reality"}, "A05": {"clear"}}

# 4. Borderline pairs: kept unless you decide otherwise
CHECK = {
    "A01-34": "detached: the viewer, or the world?",
    "A01-40": "your negative; 'behind appearances' is abstract",
    "A01-41": "a paradox: real and unreal at once",
    "A01-53": "detached from one's surroundings: the viewer",
    "A01-54": "detached with respect to surroundings: the viewer",
    "A02-19": "autopilot: behaviour more than look",
    "A02-21": "a dream state come true",
    "A02-27": "the dream of a sleeping god: a metaphor",
    "A04-15": "unplugged from the matrix: a story reference, meaning reversed",
    "A04-19": "playing a video game: an action",
    "A05-23": "self-world: between one's self, body and the world",
    "A05-25": "self-world: between the self and the world",
    "A05-26": "self-world: between themself and the world",
    "A05-35": "self-world: separated from reality by a covering",
    "A05-36": "self-world: separated from the world by a barrier",
    "A14-11": "negative 'an inhabited space' brings in people",
    "A14-12": "experiences of blackness: experience, and darkness",
    "A14-14": "nothingness choking the life out of everything: a metaphor",
}

# Filters of Stage D (decision 4)
MAX_CONTENT_EDIT = 2
MAX_LEN_DIFF     = 2
MIN_TEXTS, MIN_DOCS, MIN_FIRST_PERSON = 5, 2, 1           # keeping rules
SCENARIOS = {"all": {"vision", "not_specific", "disputed"},
             "vision": {"vision"}, "not_specific": {"not_specific"}}
N_ALTERNATIVES = 6                                         # shown per row

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
# Helpers (the filters are those of v4_stageD_check.py)
# ============================================================

def P(*n):
    return os.path.join(OUTPUT_DIR, *n)


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
    a, b = words(pos), words(cand)
    rem, add = [], []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(a=a, b=b,
                                                      autojunk=False).get_opcodes():
        if op != "equal":
            rem += a[i1:i2]
            add += b[j1:j2]
    rc = [w for w in rem if w not in SMALL]
    ac = [w for w in add if w not in SMALL]
    return max(len(rc), len(ac)), abs(len(a) - len(b)), rc, ac


def filters(pos, c):
    """The hard filters of Stage D; returns the reasons a negative fails."""
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
    ce, ld, rc, _ = edit_size(pos, c)
    if not rc:
        why.append("no word of the positive replaced")
    if ce > MAX_CONTENT_EDIT:
        why.append(f"edit ({ce} content words)")
    if ld > MAX_LEN_DIFF:
        why.append(f"length (±{ld} words)")
    return why


def added(pos, c):
    return set(edit_size(pos, c)[3])


def truthy(x):
    return str(x).strip().lower() in {"true", "1", "yes"}


# ============================================================
# Main
# ============================================================

def main():
    t0 = time.time()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=" * 70)
    print("v4_stageD2_revise.py  (manual revision of the Stage D pairs)")
    print("=" * 70)

    import yaml
    doc = yaml.safe_load(open(os.path.join(STAGED_DIR, "axes_v4.yaml"), encoding="utf-8"))
    pairs = []
    for ax in doc["axes"]:
        for p in ax["pairs"]:
            pairs.append(dict(p, axis_id=ax["id"]))
    ids = [p["anchor_id"] for p in pairs]
    unknown = sorted((set(DROP) | set(REPLACE) | set(CHECK)) - set(ids))
    if unknown:
        raise SystemExit(f"Anchors in the settings that are not pairs of Stage D: {unknown}")
    both_set = sorted(set(DROP) & (set(REPLACE) | set(CHECK)))
    if both_set:
        raise SystemExit(f"Anchors both dropped and replaced/checked: {both_set}")

    K = pd.read_csv(os.path.join(STAGED_DIR, "stageD_candidates.csv"), dtype=str).fillna("")
    K = K[K.passes.map(truthy)].copy()
    K["Knum"] = K.K.str[1:].astype(int)
    K = K.sort_values(["anchor_id", "Knum"])
    pm = pd.read_csv(os.path.join(STAGEB_DIR, "stageB_phrase_map.csv"), dtype=str).fillna("")
    pm = pm[pm.included.str.lower() == "true"]
    sel = pd.read_csv(os.path.join(STAGEB_DIR, "stageB_clip_selection.csv"),
                      dtype=str).fillna("").set_index("axis_id")

    old = {}
    if os.path.exists(P("stageD2_revision.csv")):
        o = pd.read_csv(P("stageD2_revision.csv"), dtype=str).fillna("")
        old = {r.anchor_id: (r.decision.strip(), r.comment) for r in o.itertuples()}

    def avoid_words(p):
        a = set(AXIS_AVOID.get(p["axis_id"], set())) | GLOBAL_AVOID
        if p["anchor_id"] in REPLACE:
            a |= REPLACE[p["anchor_id"]][1]
        return a

    def ranked(p):
        """Allowed candidates of a pair, best first."""
        g = K[K.anchor_id == p["anchor_id"]]
        av, pref = avoid_words(p), PREFER.get(p["axis_id"], set())
        conv = CONVENTION.get(p["axis_id"], set())
        out = []
        for r in g.itertuples():
            if key(r.candidate) == key(p["negative"]):
                continue
            ad = added(p["positive"], r.candidate)
            if ad & av:
                continue
            out.append({"K": r.K, "text": r.candidate, "models": r.models,
                        "flags": r.flag_list, "pref": bool(ad & pref),
                        "conv": len(ad & conv), "Knum": r.Knum})
        out.sort(key=lambda x: (not x["pref"], x["conv"], x["Knum"]))
        return out

    def describe(c):
        tags = [c["models"]] + (["preferred word"] if c["pref"] else []) + \
               (["convention conflict"] if c["conv"] else [])
        fl = f" [{c['flags']}]" if c["flags"] else ""
        return f"{c['K']}: {c['text']} ({', '.join(tags)}){fl}"

    rows, final = [], []
    for p in pairs:
        a, ax_ = p["anchor_id"], p["axis_id"]
        pos, neg0 = clean(p["positive"]), clean(p["negative"])
        av = avoid_words(p)
        hit_axis = added(pos, neg0) & AXIS_AVOID.get(ax_, set())
        alts = []
        if a in DROP:
            action, reason, note = "drop", DROP[a][0], DROP[a][1]
        elif a in REPLACE or hit_axis:
            alts = ranked(p)
            if a in REPLACE:
                reason, note = REPLACE[a][0], "avoid: " + ", ".join(sorted(REPLACE[a][1]))
            else:
                reason = "decision 17 (R1b): negative adds " + ", ".join(sorted(hit_axis))
                note = "prefer: " + ", ".join(sorted(PREFER.get(ax_, set()))) if ax_ in PREFER else ""
            action = "replace" if alts else "open"
            if a in CHECK:
                note = (note + "; " if note else "") + "check: " + CHECK[a]
        elif a in CHECK:
            action, reason, note = "keep", "check", CHECK[a]
        else:
            action, reason, note = "keep", "", ""
        prop = alts[0] if alts and action == "replace" else None

        dec, com = old.get(a, ("", ""))
        f_action, f_neg, f_src, how, problem, warning = "", "", "", "", "", ""
        d = dec.strip()
        if not d:
            if action == "drop":
                f_action, how = "drop", "dropped (proposed)"
            elif action == "replace":
                f_action, f_neg, f_src = "replace", prop["text"], f"{prop['K']} ({prop['models']})"
                how = "replaced (proposed)"
            elif action == "open":
                f_action, how = "open", "open: no allowed candidate; write your negative or drop"
            else:
                f_action, f_neg, f_src = "keep", neg0, p["negative_source"]
                how = "kept"
        elif d.lower() == "keep":
            f_action, f_neg, f_src, how = "keep", neg0, p["negative_source"], "kept by you"
        elif d.lower() == "drop":
            f_action, how = "drop", "dropped by you"
        else:
            allk = K[K.anchor_id == a]
            if not re.fullmatch(r"(?i)K\d+", d):
                same = allk[allk.candidate.map(key) == key(d)]
                if len(same):
                    d = same.K.iloc[0]
            if re.fullmatch(r"(?i)K\d+", d):
                h = allk[allk.K == d.upper()]
                if not len(h):
                    problem = f"{d} is not a candidate of this anchor that passed the filters"
                else:
                    f_action, f_neg = "replace", h.candidate.iloc[0]
                    f_src, how = f"{d.upper()} ({h.models.iloc[0]})", "replaced by you"
            else:
                why = filters(pos, d)
                if why:
                    problem = "your negative fails: " + "; ".join(why)
                else:
                    f_action, f_neg, f_src, how = "replace", clean(d), "yours", "your negative"
            if f_neg:
                bad = added(pos, f_neg) & av
                conv = added(pos, f_neg) & CONVENTION.get(ax_, set())
                if bad:
                    warning = "adds an avoided word: " + ", ".join(sorted(bad))
                if conv:
                    warning = (warning + "; " if warning else "") + \
                        "word of another axis: " + ", ".join(sorted(conv))
        if problem:
            f_action, how = "problem", problem

        rows.append({"anchor_id": a, "axis_id": ax_, "positive": pos,
                     "negative_stageD": neg0, "proposed_action": action, "reason": reason,
                     "note": note,
                     "proposed_negative": prop["text"] if prop else "",
                     "proposed_K": describe(prop) if prop else "",
                     "alternatives": " | ".join(describe(c) for c in alts[1:N_ALTERNATIVES + 1]),
                     "decision": dec, "comment": com, "final_action": f_action,
                     "final_negative": f_neg, "warning": warning, "problem": problem})
        final.append(dict(p, positive=pos, negative=f_neg, negative_source=f_src,
                          action=f_action, how=how, reason=reason))

    RV = pd.DataFrame(rows)
    RV.to_csv(P("stageD2_revision.csv"), index=False)
    FP = pd.DataFrame(final)
    FP["in_axes"] = FP.action.isin(["keep", "replace"])
    FP.drop(columns=["sense_classes", "text_ids", "phrase_ids", "documents"]).assign(
        documents=[";".join(x) for x in FP.documents]).to_csv(P("stageD2_pairs.csv"),
                                                               index=False)

    # --- per axis: pairs and support (keeping rules as in Stage B) ---
    axrows = []
    for ax_ in dict.fromkeys(FP.axis_id):
        g = FP[FP.axis_id == ax_]
        k = g[g.in_axes]
        tids = {t for x in k.text_ids for t in x}
        sub = pm[pm.text_id.isin(tids)]
        row = {"axis_id": ax_, "name": sel.at[ax_, "name_stageC"] if ax_ in sel.index else "",
               "pairs_stageD": len(g), "pairs": int(len(k)),
               "dropped": int((g.action == "drop").sum()),
               "replaced": int((g.action == "replace").sum()),
               "open": int(g.action.isin(["open", "problem"]).sum())}
        for sc, classes in SCENARIOS.items():
            s = sub[sub.sense_class.isin(classes)]
            nt, nd = s.text_id.nunique(), s.doc_id.nunique()
            nf = int((s.first_person == "yes").sum())
            row.update({f"{sc}_texts": nt, f"{sc}_docs": nd, f"{sc}_first_person": nf,
                        f"{sc}_kept": nt >= MIN_TEXTS and nd >= MIN_DOCS and
                        nf >= MIN_FIRST_PERSON})
        if len(k):
            dc = Counter(ph.split("-P")[0] for x in k.phrase_ids for ph in x)
            top, n = dc.most_common(1)[0]
            row["largest_doc_share"] = f"{top} {n / sum(dc.values()):.0%}"
        axrows.append(row)
    AX = pd.DataFrame(axrows)
    AX.to_csv(P("stageD2_axes.csv"), index=False)

    n_open = int(FP.action.isin(["open", "problem"]).sum())
    complete = n_open == 0
    if complete:
        out = {"version": "v4", "protocol": PROTOCOL,
               "created": time.strftime("%Y-%m-%d %H:%M:%S"),
               "revised_from": os.path.join(STAGED_DIR, "axes_v4.yaml"),
               "template": doc["template"],
               "templates_robustness": doc.get("templates_robustness", []),
               "dropped_pairs": [{"anchor_id": r.anchor_id, "axis_id": r.axis_id,
                                  "positive": r.positive, "reason": r.reason or "your decision"}
                                 for r in FP[FP.action == "drop"].itertuples()],
               "axes": []}
        for ax in doc["axes"]:
            k = FP[(FP.axis_id == ax["id"]) & FP.in_axes]
            out["axes"].append({
                "id": ax["id"], "name": ax["name"], "definition": ax["definition"],
                "pairs": [{"anchor_id": r.anchor_id, "positive": r.positive,
                           "negative": r.negative, "negative_source": r.negative_source,
                           "revision": r.how, "sense_classes": r.sense_classes,
                           "text_ids": r.text_ids, "phrase_ids": r.phrase_ids,
                           "documents": r.documents} for r in k.itertuples()]})
        yaml.safe_dump(out, open(P("axes_v4.yaml"), "w", encoding="utf-8"),
                       sort_keys=False, allow_unicode=True, width=1000)

    # --- report ---
    L = ["STAGE D2: MANUAL REVISION OF THE PAIRS (v4_protocol.md 1.12, section 6.4a)",
         "=" * 78, f"Date: {time.strftime('%Y-%m-%d %H:%M')}",
         f"Pairs of Stage D: {len(FP)}; final pairs: {int(FP.in_axes.sum())}", "",
         "1. PROPOSED ACTIONS", "-" * 78]
    for k_, n in RV.proposed_action.value_counts().items():
        L.append(f"  {k_:<10}{n:>5}")
    L.append("  reasons:")
    for k_, n in RV[RV.reason != ""].reason.map(
            lambda s: s.split(":")[0] if s.startswith("decision 17") else s).value_counts().items():
        L.append(f"    {n:>4}  {k_}")
    L += ["", "2. FINAL ACTIONS (after your decisions)", "-" * 78]
    for k_, n in FP.how.map(lambda s: s.split(":")[0]).value_counts().items():
        L.append(f"  {k_:<34}{n:>5}")
    if (RV.warning != "").any():
        L += ["", "  warnings (allowed, but check):"]
        for r in RV[RV.warning != ""].itertuples():
            L.append(f"    {r.anchor_id}: {r.warning}")
    if (RV.problem != "").any():
        L += ["", "  PROBLEMS:"]
        for r in RV[RV.problem != ""].itertuples():
            L.append(f"    {r.anchor_id}: {r.problem}")
    op = RV[RV.final_action == "open"]
    if len(op):
        L += ["", f"  OPEN ({len(op)}): no allowed candidate; write your negative or drop:"]
        for r in op.itertuples():
            L.append(f"    {r.anchor_id}  {r.positive}  (Stage D: {r.negative_stageD})")
    L += ["", "3. A05 NEGATIVES AFTER THE REVISION (words added by the final negatives)",
          "-" * 78]
    g = FP[(FP.axis_id == "A05") & FP.in_axes & (FP.negative != "")]
    cnt = Counter(w for r in g.itertuples() for w in added(r.positive, r.negative))
    L.append("  " + ", ".join(f"{w} {n}" for w, n in cnt.most_common(15)))
    L += ["", "4. AXES (keeping rules as in Stage B)", "-" * 78,
          f"  {'axis':<6}{'name':<36}{'D':>4}{'pairs':>6}{'drop':>5}{'repl':>5}{'open':>5}"
          f"{'docs':>5}{'1st':>5}  kept all/vis/n-s  largest document"]
    for r in AX.itertuples():
        f3 = "/".join("Y" if getattr(r, f"{sc}_kept") else "-" for sc in SCENARIOS)
        L.append(f"  {r.axis_id:<6}{r.name[:34]:<36}{r.pairs_stageD:>4}{r.pairs:>6}"
                 f"{r.dropped:>5}{r.replaced:>5}{r.open:>5}{r.all_docs:>5}"
                 f"{r.all_first_person:>5}  {f3:<17}{getattr(r, 'largest_doc_share', '')}")
    L += ["", (f"STAGE D2 COMPLETE: {P('axes_v4.yaml')} written. Next: run "
               "v4_check17_scene_properties.py on it."
               if complete else
               f"Not complete: {n_open} rows open or with problems "
               "(stageD2_revision.csv, column decision).")]
    open(P("stageD2_report.txt"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"Runtime {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
