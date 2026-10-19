"""
make_screening_page_corrected.py  (stimulus pipeline, screening tool
                                   for step 2)

WHAT CHANGED IN THIS VERSION
----------------------------
Only BASE_DIR, which now points at data/select_scenes_v4_ext to match
select_scenes_v4_corrected.py. Nothing else: the page, the decisions
and the saved CSV are unchanged.

The page MUST live beside the crops it shows, because it references
them by relative path (crops/<scene_id>.jpg). So BASE_DIR follows the
selection output folder rather than the project root.

Your existing screening is not affected. Decisions are stored in the
browser under a name derived from the SET OF SCENES in the worksheet
(STORAGE_SCOPE = "auto"), not from the folder, so a page rebuilt for
the same shortlist still finds them even though the folder name has
changed. A different shortlist starts clean, which is the behaviour you
want if you screen extra scenes later.

Builds a self-contained web page for screening the shortlisted scenes by
hand, and writes it next to the crops it shows.

Why a page rather than the contact sheets
-----------------------------------------
The contact sheets are for a fast first pass. This is for the decision:
one scene at a time, large, on a neutral surround, with the keep/discard
state visible as a coloured frame and recorded as you go. Four hundred
scenes is a long sitting, so the page remembers where you were, saves
after every decision, and can be closed and reopened.

What it produces
----------------
  screening.html        open it in a browser (double-click is enough;
                        no server, no internet)

and, when you press Save in the page, your browser downloads

  screening_list_done.csv   the same columns as screening_list.csv with
                            `keep` and `reason` filled in. Step 3 reads
                            the rows with keep == 1.

NOT PRESSING KEEP MEANS DISCARD. Every scene starts undecided; anything
you leave undecided is written out as keep = 0, so you can stop whenever
you like and the result is well defined. The page tells you how many are
still undecided before you save.

Working by category
-------------------
Scenes are grouped by category, and the sidebar shows how many you have
kept in each one out of how many it offers. Set PER_CATEGORY_TARGET to
the number you want from a category and the sidebar turns amber once you
pass it -- a reminder, not a lock, so you can still take a fourth if it
earns its place.

This exists because the first two screening rounds produced six
gymnasia out of twenty indoor scenes: a third of the indoor set was one
category, which is a weak position if indoor-vs-outdoor ever has to be
defended. Working category by category, with the running count in view,
is what stops that happening again.

"This category only" confines the arrows to the current category, so you
can finish one and move on with `]` rather than drifting across the
boundary without noticing.

Inputs
------
  WORKSHEET   data/select_scenes_v4_ext/screening_list.csv  (step 2)
  CROP_DIR    data/select_scenes_v4_ext/crops/              (step 2)

If the worksheet is missing, the page is built from whatever images are
in CROP_DIR, so this also works on any folder of pictures.

Usage
-----
    conda activate drl2drv
    python make_screening_page_corrected.py
    # then open data/select_scenes_v4_ext/screening.html

Keyboard, once the page is open
-------------------------------
    right / left arrow   next / previous
    [ and ]              previous / next category
    K or space           keep (press again to undo)
    X                    discard explicitly
    F                    confine the arrows to this category
    G                    overview grid, grouped by category
    U                    jump to the next undecided scene
    S                    save
"""

import csv
import hashlib
import html
import json
import os


# ============================================================
# SETTINGS
# ============================================================

BASE_DIR    = "data/select_scenes_v4_ext"
WORKSHEET   = os.path.join(BASE_DIR, "screening_list.csv")
CROP_DIR    = os.path.join(BASE_DIR, "crops")
OUTPUT_HTML = os.path.join(BASE_DIR, "screening.html")

# OUTPUT_CSV_NAME
#   The file the page downloads when you press Save. Kept distinct from
#   screening_list.csv so a save can never overwrite the worksheet the
#   page was built from.
OUTPUT_CSV_NAME = "screening_list_done.csv"

# TITLE
#   Shown in the page header and the browser tab.
TITLE = "Scene screening"

# META_FIELDS
#   Worksheet columns shown beside each image. Keep this short: the
#   judgement is about content, and numbers invite second-guessing the
#   automatic checks that already passed.
META_FIELDS = ["category", "scene_id"]

# GROUP_FIELD
#   The worksheet column the sidebar groups by. "category" is the one
#   that matters here: it is what decides whether the final set is a
#   spread of places or six views of a gymnasium. Set to "" for a flat
#   list with no sidebar.
GROUP_FIELD = "category"

# GROUP_ORDER
#   "count" puts the categories offering the most images first, so the
#   ones that could flood the set are the ones you deal with while
#   fresh. "name" sorts alphabetically.
GROUP_ORDER = "count"

# SORT_BY_GROUP
#   True reorders the page so a category's images are consecutive, which
#   is what makes "judge this category as a set" possible at all. False
#   keeps the worksheet order and uses the sidebar only as a counter.
#   Either way the saved CSV keeps the worksheet's own row order.
SORT_BY_GROUP = True

# PER_CATEGORY_TARGET
#   How many scenes you intend to take from any one category. The
#   sidebar greys the count until you reach it and turns amber past it.
#   Soft on purpose: a category with several genuinely different rooms
#   can earn more, and one with nothing usable should give none.
#
#   Setting it: aim for (scenes you want) / (categories available), then
#   round down. For ~40 scenes over ~30 categories that is 1; 2 leaves
#   room to prefer the better categories. It is 0 to switch the warning
#   off entirely.
PER_CATEGORY_TARGET = 2

# STORAGE_SCOPE
#   Which saved decisions a page picks up when it opens. The browser
#   keeps them under this name.
#
#   "auto" (recommended) derives the name from the SET OF SCENES in the
#   worksheet, so a new shortlist is a clean slate automatically and a
#   regenerated page for the same shortlist still finds your decisions.
#   This matters because two screening rounds drawn from the same corpus
#   share some scenes: keyed on the folder, round two would open with
#   round one's choices already made and you would be auditing them
#   instead of judging afresh.
#
#   Any other string pins the name, so several runs deliberately share
#   one set of decisions. To carry decisions across different shortlists
#   on purpose, prefer Save then "Load decisions" -- it is explicit and
#   leaves a file behind.
#
#   The Reset button clears the current page's decisions at any time.
STORAGE_SCOPE = "auto"

# IMAGE_EXTS
#   Used only when the worksheet is missing and the folder is scanned.
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp")

# PRELOAD_AHEAD
#   Images fetched ahead of the current one, so paging forward is
#   instant. Raise on a fast disk, lower if memory is tight.
PRELOAD_AHEAD = 4


# ============================================================
# Rows
# ============================================================

def load_rows():
    """One record per scene: the image path, plus whatever the worksheet
    carries. Falls back to scanning CROP_DIR."""
    if os.path.isfile(WORKSHEET):
        with open(WORKSHEET, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            raise RuntimeError(f"{WORKSHEET} has no rows.")
        cols = list(rows[0].keys())
        out, missing = [], 0
        for i, r in enumerate(rows):
            rel = r.get("crop") or ""
            if not rel:
                sid = r.get("scene_id") or ""
                rel = f"crops/{sid}.jpg"
            if not os.path.isfile(os.path.join(BASE_DIR, rel)):
                missing += 1
                continue
            rec = dict(r)
            rec["_src"] = rel.replace(os.sep, "/")
            rec["_i"] = i
            # _key identifies the scene, not its position. Decisions are
            # saved against it, so rebuilding the page from a reordered or
            # filtered worksheet keeps every decision attached to the
            # picture it was made about.
            rec["_key"] = (r.get("scene_id") or "").strip() or rec["_src"]
            out.append(rec)
        if missing:
            print(f"  WARNING: {missing} worksheet rows have no image on "
                  "disk; they are left out of the page and will keep "
                  "whatever `keep` the worksheet already had.")
        print(f"  {len(out)} scenes from {WORKSHEET}")
        return out, cols, rows

    # --- no worksheet: scan the folder ---
    if not os.path.isdir(CROP_DIR):
        raise FileNotFoundError(
            f"Neither {WORKSHEET} nor {CROP_DIR} exists. Run step 2 "
            "first, or point CROP_DIR at a folder of images.")
    names = sorted(n for n in os.listdir(CROP_DIR)
                   if n.lower().endswith(IMAGE_EXTS))
    if not names:
        raise RuntimeError(f"No images in {CROP_DIR}.")
    rel_dir = os.path.relpath(CROP_DIR, BASE_DIR).replace(os.sep, "/")
    out = []
    for i, n in enumerate(names):
        sid = os.path.splitext(n)[0]
        out.append({"n": str(i + 1), "scene_id": sid, "category": "",
                    "crop": f"{rel_dir}/{n}", "keep": "", "reason": "",
                    "_src": f"{rel_dir}/{n}", "_i": i, "_key": sid})
    print(f"  {len(out)} images scanned from {CROP_DIR} "
          "(no worksheet found)")
    return out, ["n", "scene_id", "category", "crop", "keep",
                 "reason"], out


def build_groups(rows):
    """Bucket the rows by GROUP_FIELD and, if asked, reorder the page so
    each category's images sit together.

    Returns (rows_in_display_order, groups), where each group carries
    the display positions of its own rows. `_g` on a row is its group's
    index, so the page can tell at a glance which category it is in.

    Reordering is safe: every decision is stored against the scene's own
    id and the saved CSV is written in the worksheet's row order, so
    what changes here is only the order you are shown things in.
    """
    if not GROUP_FIELD:
        for r in rows:
            r["_g"] = 0
        return rows, [{"name": "all", "idx": list(range(len(rows)))}]

    buckets = {}
    for r in rows:
        name = (r.get(GROUP_FIELD) or "").strip() or "(none)"
        buckets.setdefault(name, []).append(r)

    names = list(buckets)
    if GROUP_ORDER == "count":
        # biggest first: the categories that could flood the set are the
        # ones worth judging while you are still fresh
        names.sort(key=lambda n: (-len(buckets[n]), n))
    elif GROUP_ORDER == "name":
        names.sort()
    else:
        raise ValueError(
            f"GROUP_ORDER must be 'count' or 'name', got {GROUP_ORDER!r}")

    for gi, n in enumerate(names):
        for r in buckets[n]:
            r["_g"] = gi

    ordered = ([r for n in names for r in buckets[n]] if SORT_BY_GROUP
               else rows)
    groups = [{"name": n, "idx": []} for n in names]
    for i, r in enumerate(ordered):
        groups[r["_g"]]["idx"].append(i)

    sizes = [len(g["idx"]) for g in groups]
    print(f"  {len(groups)} categories, "
          f"{min(sizes)}-{max(sizes)} images each")
    if PER_CATEGORY_TARGET:
        over = sum(1 for s in sizes if s > PER_CATEGORY_TARGET)
        print(f"  {over} categories offer more than the "
              f"per-category target of {PER_CATEGORY_TARGET}")
    return ordered, groups


# ============================================================
# The page
# ============================================================

CSS = """
:root{
  --bg:#1c1c1e; --panel:#242427; --line:#3a3a3f; --text:#e9e9ec;
  --muted:#9a9aa2; --keep:#36c46a; --drop:#8a8a92; --accent:#5aa9e6;
  --warn:#e0a33a;
  /* a neutral mat around the photograph, so the surround does not
     bias how bright or colourful it looks */
  --mat:#6e6e72;
}
*{box-sizing:border-box}
html,body{height:100%}
body{
  margin:0; background:var(--bg); color:var(--text);
  font:14px/1.5 ui-sans-serif,system-ui,-apple-system,"Segoe UI",
       Roboto,Helvetica,Arial,sans-serif;
  display:flex; flex-direction:column; overflow:hidden;
}
header{
  display:flex; align-items:center; gap:16px; flex-wrap:wrap;
  padding:10px 16px; background:var(--panel);
  border-bottom:1px solid var(--line);
}
h1{font-size:15px; font-weight:600; margin:0; letter-spacing:.01em}
.counts{color:var(--muted); font-variant-numeric:tabular-nums}
.counts b{color:var(--text); font-weight:600}
.spacer{flex:1}
button{
  font:inherit; color:var(--text); background:#303036;
  border:1px solid var(--line); border-radius:7px;
  padding:7px 13px; cursor:pointer;
}
button:hover{background:#3a3a42}
button:active{transform:translateY(1px)}
button.primary{background:var(--accent); border-color:var(--accent);
  color:#06243a; font-weight:600}
button.keep.on{background:var(--keep); border-color:var(--keep);
  color:#05240f; font-weight:600}
kbd{
  font:11px ui-monospace,SFMono-Regular,Menlo,monospace;
  background:#15151a; border:1px solid var(--line); border-radius:4px;
  padding:1px 5px; color:var(--muted);
}
main{flex:1; display:flex; min-height:0}

/* --- category sidebar --- */
aside{
  width:236px; flex:none; overflow-y:auto; background:var(--panel);
  border-right:1px solid var(--line); padding:8px 0;
}
.asidehead{
  padding:4px 14px 8px; color:var(--muted); font-size:12px;
  display:flex; align-items:center; gap:8px; flex-wrap:wrap;
}
.grp{
  display:flex; align-items:baseline; gap:8px; padding:6px 14px;
  cursor:pointer; border-left:3px solid transparent;
}
.grp:hover{background:#2c2c31}
.grp.current{background:#30303a; border-left-color:var(--accent)}
.gname{
  flex:1; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;
  font-size:13px;
}
.gcount{
  font-variant-numeric:tabular-nums; font-size:12px; color:var(--muted);
}
/* at the target: you have what you came for. past it: the count you
   actually need to see, so one category cannot quietly take over */
.grp.at   .gcount{color:var(--keep); font-weight:600}
.grp.over .gcount{color:var(--warn); font-weight:600}
.grp.over .gname{color:var(--warn)}
.grp.done .gname{color:var(--muted)}

.stage{
  flex:1; display:flex; flex-direction:column; align-items:center;
  justify-content:center; gap:12px; padding:16px; min-height:0;
}
.frame{
  /* the rectangle that tells you the decision at a glance */
  border:6px solid var(--drop); border-radius:4px; background:var(--mat);
  padding:10px; line-height:0; max-height:100%;
  transition:border-color .12s ease;
}
.frame.keep{border-color:var(--keep)}
.frame.undecided{border-color:var(--line); border-style:dashed}
.frame img{
  max-width:100%; max-height:calc(100vh - 230px);
  display:block; border-radius:2px;
}
.meta{
  display:flex; gap:18px; flex-wrap:wrap; justify-content:center;
  color:var(--muted); font-size:13px;
}
.meta b{color:var(--text); font-weight:600}
.state{font-weight:600}
.state.keep{color:var(--keep)}
.state.drop{color:var(--drop)}
.state.undecided{color:var(--muted)}
.controls{display:flex; gap:10px; align-items:center}
.reason{
  font:inherit; color:var(--text); background:#1a1a1f;
  border:1px solid var(--line); border-radius:7px; padding:7px 10px;
  width:320px;
}
footer{
  background:var(--panel); border-top:1px solid var(--line);
  padding:8px 12px; overflow-x:auto; white-space:nowrap;
}
.thumb{
  display:inline-block; width:62px; height:62px; margin-right:6px;
  border:3px solid var(--drop); border-radius:4px; overflow:hidden;
  cursor:pointer; vertical-align:middle; background:var(--mat);
}
.thumb.keep{border-color:var(--keep)}
/* dashed, as in the big frame, so the three states stay distinct even
   at thumbnail size where grey-on-grey is hard to read */
.thumb.undecided{border-color:var(--line); border-style:dashed}
.thumb.current{outline:2px solid var(--accent); outline-offset:2px}
.thumb img{width:100%; height:100%; object-fit:cover; display:block}
.grid{
  position:fixed; inset:0; background:rgba(16,16,18,.97); z-index:10;
  overflow:auto; padding:70px 20px 24px; display:none;
}
.grid.open{display:block}
.grid .bar{
  position:fixed; top:0; left:0; right:0; padding:12px 16px;
  background:var(--panel); border-bottom:1px solid var(--line);
  display:flex; gap:14px; align-items:center;
}
.cells{
  display:grid; gap:10px; margin-bottom:22px;
  grid-template-columns:repeat(auto-fill,minmax(150px,1fr));
}
.ghead{
  display:flex; align-items:baseline; gap:10px; margin:4px 0 8px;
  padding-bottom:5px; border-bottom:1px solid var(--line);
  font-weight:600; font-size:14px;
}
.ghead span{color:var(--muted); font-weight:400; font-size:12px}
.ghead.over, .ghead.over span{color:var(--warn)}
.cell{
  border:4px solid var(--drop); border-radius:5px; overflow:hidden;
  cursor:pointer; background:var(--mat); position:relative;
}
.cell.keep{border-color:var(--keep)}
.cell.undecided{border-color:var(--line); border-style:dashed}
.cell img{width:100%; aspect-ratio:1; object-fit:cover; display:block}
.cell .tag{
  position:absolute; left:0; right:0; bottom:0; padding:3px 6px;
  background:rgba(0,0,0,.66); font-size:11px; color:#fff;
  white-space:nowrap; overflow:hidden; text-overflow:ellipsis;
}
.hint{color:var(--muted); font-size:12px}
@media (max-width:900px){
  aside{width:160px}
  .reason{width:160px}
  .frame img{max-height:calc(100vh - 300px)}
}
"""

JS = r"""
const $ = s => document.querySelector(s);
const KEY = 'scene-screening:' + (DATA.id || 'default');

// ---- state ----
// decided[key] is undefined (untouched), true (keep) or false (discard),
// where key is the scene's own id rather than its position in the list.
// Anything left untouched is written out as keep = 0.
let decided = {};
let reasons = {};
let cur = 0;
let groupFilter = false;   // confine the arrows to the current category
try {
  const raw = localStorage.getItem(KEY);
  if (raw) {
    const s = JSON.parse(raw);
    decided = s.decided || {};
    reasons = s.reasons || {};
    cur = Math.min(Math.max(s.cur || 0, 0), DATA.rows.length - 1);
    groupFilter = !!s.groupFilter;
  }
} catch (e) {}   // private windows and some file:// setups block this

function persist() {
  try {
    localStorage.setItem(KEY,
      JSON.stringify({decided, reasons, cur, groupFilter}));
  } catch (e) {}
}

const keyOf   = i => DATA.rows[i]._key;
const stateOf = i => decided[keyOf(i)] === undefined ? 'undecided'
                   : (decided[keyOf(i)] ? 'keep' : 'drop');
const groupOf = i => DATA.rows[i]._g;

// Per-category tallies. Recomputed on every render rather than kept as
// a running total, so a decision undone anywhere can never leave the
// sidebar disagreeing with the actual state.
function groupStats() {
  return DATA.groups.map(g => {
    let keep = 0, un = 0;
    for (const i of g.idx) {
      const s = stateOf(i);
      if (s === 'keep') keep++; else if (s === 'undecided') un++;
    }
    return {keep: keep, un: un, total: g.idx.length};
  });
}

// ---- rendering ----
function render() {
  const r = DATA.rows[cur];
  const st = stateOf(cur);

  const frame = $('#frame');
  frame.className = 'frame ' + (st === 'keep' ? 'keep'
                              : st === 'drop' ? '' : 'undecided');
  const img = $('#img');
  if (img.getAttribute('src') !== r._src) img.setAttribute('src', r._src);
  img.alt = r.scene_id || '';

  $('#pos').textContent = (cur + 1) + ' / ' + DATA.rows.length;

  // the running tally for THIS category, next to the picture, because
  // that is the number that stops a category taking over the set
  const gs = groupStats()[groupOf(cur)];
  const gname = DATA.groups[groupOf(cur)].name;
  const overTarget = TARGET > 0 && gs.keep > TARGET;
  $('#meta').innerHTML = DATA.meta
    .filter(f => r[f])
    .map(f => '<span>' + f + ' <b>' + esc(r[f]) + '</b></span>').join('')
    + '<span class="state ' + st + '">' + st.toUpperCase() + '</span>'
    + '<span' + (overTarget ? ' style="color:var(--warn)"' : '') + '>'
    + esc(gname) + ' <b>' + gs.keep + ' kept</b> of ' + gs.total
    + (TARGET > 0 ? ' · target ' + TARGET : '') + '</span>';

  const kb = $('#keepBtn');
  kb.classList.toggle('on', st === 'keep');
  kb.textContent = st === 'keep' ? 'Kept  ✓' : 'Keep';

  $('#reason').value = reasons[keyOf(cur)] || '';
  counts();
  sidebar();
  strip();
  // fetch a few ahead so paging forward does not stall
  for (let k = 1; k <= PRELOAD; k++) {
    const n = DATA.rows[cur + k];
    if (n) { const p = new Image(); p.src = n._src; }
  }
}

function counts() {
  let keep = 0, drop = 0, un = 0;
  for (let i = 0; i < DATA.rows.length; i++) {
    const s = stateOf(i);
    if (s === 'keep') keep++; else if (s === 'drop') drop++; else un++;
  }
  $('#counts').innerHTML =
    'kept <b>' + keep + '</b> · discarded <b>' + drop +
    '</b> · undecided <b>' + un + '</b>';
  return {keep, drop, un};
}

function sidebar() {
  const st = groupStats(), cg = groupOf(cur);
  let out = '';
  DATA.groups.forEach((g, i) => {
    const s = st[i], cls = ['grp'];
    if (i === cg) cls.push('current');
    if (TARGET > 0 && s.keep > TARGET) cls.push('over');
    else if (TARGET > 0 && s.keep === TARGET) cls.push('at');
    if (s.un === 0) cls.push('done');
    out += '<div class="' + cls.join(' ') + '" data-g="' + i + '" ' +
           'title="' + esc(g.name) + '">' +
           '<span class="gname">' + esc(g.name) + '</span>' +
           '<span class="gcount">' + s.keep + '/' + s.total +
           '</span></div>';
  });
  $('#groups').innerHTML = out;
  const c = $('#groups .current');
  if (c) c.scrollIntoView({block: 'nearest'});
  $('#filterBox').checked = groupFilter;
}

function strip() {
  // with the filter on, the strip shows only the current category, so
  // what you see along the bottom is what the arrows will walk through
  const el = $('#strip');
  const pool = groupFilter ? DATA.groups[groupOf(cur)].idx
                           : DATA.rows.map((_, i) => i);
  const at = pool.indexOf(cur);
  const lo = Math.max(0, at - 12), hi = Math.min(pool.length, at + 24);
  let out = '';
  for (let p = lo; p < hi; p++) {
    const i = pool[p], r = DATA.rows[i], s = stateOf(i);
    out += '<span class="thumb ' + (s === 'drop' ? '' : s) +
           (i === cur ? ' current' : '') + '" data-i="' + i + '">' +
           '<img loading="lazy" src="' + esc(r._src) + '"></span>';
  }
  el.innerHTML = out;
  const c = el.querySelector('.current');
  if (c) c.scrollIntoView({block: 'nearest', inline: 'center'});
}

function cellHTML(i) {
  const r = DATA.rows[i], s = stateOf(i);
  return '<div class="cell ' + (s === 'drop' ? '' : s) +
         '" data-i="' + i + '"><img loading="lazy" src="' +
         esc(r._src) + '"><div class="tag">' +
         esc(r.scene_id || '') + '</div></div>';
}

function openGrid() {
  // grouped with a heading per category, so the overview answers the
  // question you actually have: how many did I take from each?
  const st = groupStats();
  let out = '';
  DATA.groups.forEach((g, gi) => {
    const s = st[gi];
    const over = TARGET > 0 && s.keep > TARGET;
    out += '<div class="ghead' + (over ? ' over' : '') + '">' +
           esc(g.name) + '<span>' + s.keep + ' kept of ' + s.total +
           (TARGET > 0 ? ' · target ' + TARGET : '') +
           (over ? ' — over target' : '') + '</span></div>' +
           '<div class="cells">' + g.idx.map(cellHTML).join('') +
           '</div>';
  });
  $('#gridBody').innerHTML = out;
  $('#grid').classList.add('open');
}

// ---- actions ----
function go(d) {
  saveReason();
  if (groupFilter) {
    const g = DATA.groups[groupOf(cur)];
    const p = g.idx.indexOf(cur);
    cur = g.idx[Math.max(0, Math.min(g.idx.length - 1, p + d))];
  } else {
    cur = Math.max(0, Math.min(DATA.rows.length - 1, cur + d));
  }
  persist(); render();
}
function jump(i) { saveReason(); cur = i; persist(); render(); }

// Land on the first scene of the category you have not decided yet,
// rather than its first scene, so returning to a category resumes
// where you left off instead of making you page past settled ones.
function gotoGroup(gi) {
  if (gi < 0 || gi >= DATA.groups.length) return;
  saveReason();
  const idx = DATA.groups[gi].idx;
  if (!idx.length) return;
  const un = idx.find(i => decided[keyOf(i)] === undefined);
  cur = (un === undefined) ? idx[0] : un;
  persist(); render();
}
function stepGroup(d) { gotoGroup(groupOf(cur) + d); }
function toggleFilter() { groupFilter = !groupFilter; persist(); render(); }
function toggleKeep() {
  const k = keyOf(cur);
  // pressing Keep again undoes it, back to undecided
  if (decided[k] === true) delete decided[k]; else decided[k] = true;
  persist(); render();
}
function markDrop() { decided[keyOf(cur)] = false; persist(); render(); }
function saveReason() {
  const k = keyOf(cur), v = $('#reason').value.trim();
  if (v) reasons[k] = v; else delete reasons[k];
}
function nextUndecided() {
  const pool = groupFilter ? DATA.groups[groupOf(cur)].idx
                           : DATA.rows.map((_, i) => i);
  const at = pool.indexOf(cur);
  for (let k = 1; k <= pool.length; k++) {
    const i = pool[(at + k) % pool.length];
    if (decided[keyOf(i)] === undefined) { jump(i); return; }
  }
  alert(groupFilter
    ? 'Every scene in this category has been decided. Press ] for the '
      + 'next category.'
    : 'Every scene has been decided.');
}

// ---- saving ----
function esc(s) {
  return String(s == null ? '' : s)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}
function csvCell(v) {
  v = (v == null) ? '' : String(v);
  return /[",\n\r]/.test(v) ? '"' + v.replace(/"/g, '""') + '"' : v;
}

function buildCSV() {
  saveReason();
  const cols = DATA.cols.slice();
  if (!cols.includes('keep')) cols.push('keep');
  if (!cols.includes('reason')) cols.push('reason');
  // start from every original worksheet row, so rows whose image was
  // missing keep whatever they already had
  const all = DATA.all.map(r => Object.assign({}, r));
  DATA.rows.forEach((r, i) => {
    const t = all[r._i];
    if (!t) return;
    t.keep = (decided[r._key] === true) ? '1' : '0';
    t.reason = reasons[r._key] || '';
  });
  const lines = [cols.map(csvCell).join(',')];
  for (const r of all) lines.push(cols.map(c => csvCell(r[c])).join(','));
  return lines.join('\n') + '\n';
}

function save() {
  const c = counts();
  if (c.un > 0 && !confirm(
      c.un + ' scene' + (c.un === 1 ? ' is' : 's are') +
      ' still undecided and will be saved as DISCARDED (keep = 0).' +
      '\n\nSave anyway?')) return;
  const blob = new Blob([buildCSV()], {type: 'text/csv;charset=utf-8'});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = DATA.outName;
  document.body.appendChild(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 2000);
}

function resetAll() {
  const c = counts();
  if (!confirm(
      'Start from scratch?\n\nThis clears all ' + (c.keep + c.drop) +
      ' decisions on this page and cannot be undone.' +
      (c.keep ? '\n\nSave first if you want to keep the ' + c.keep +
                ' you have marked.' : ''))) return;
  decided = {}; reasons = {}; cur = 0;
  try { localStorage.removeItem(KEY); } catch (e) {}
  persist(); render();
}

function loadDecisions(file) {
  const fr = new FileReader();
  fr.onload = () => {
    const lines = fr.result.split(/\r?\n/).filter(x => x.length);
    if (!lines.length) return;
    // minimal CSV reader: enough for the file this page writes
    const parse = line => {
      const out = []; let cell = '', q = false;
      for (let i = 0; i < line.length; i++) {
        const ch = line[i];
        if (q) {
          if (ch === '"' && line[i + 1] === '"') { cell += '"'; i++; }
          else if (ch === '"') q = false;
          else cell += ch;
        } else if (ch === '"') q = true;
        else if (ch === ',') { out.push(cell); cell = ''; }
        else cell += ch;
      }
      out.push(cell); return out;
    };
    const head = parse(lines[0]);
    const iKeep = head.indexOf('keep');
    const iId = head.indexOf('scene_id');
    const iReason = head.indexOf('reason');
    if (iKeep < 0 || iId < 0) { alert('That file has no keep/scene_id columns.'); return; }
    const known = {};
    DATA.rows.forEach(r => { known[r.scene_id] = r._key; });
    let n = 0;
    for (let L = 1; L < lines.length; L++) {
      const c = parse(lines[L]);
      const k = known[c[iId]];
      if (k === undefined) continue;
      const v = (c[iKeep] || '').trim();
      if (v === '1') { decided[k] = true; n++; }
      else if (v === '0') { decided[k] = false; n++; }
      if (iReason >= 0 && c[iReason]) reasons[k] = c[iReason];
    }
    persist(); render();
    alert('Loaded ' + n + ' decisions.');
  };
  fr.readAsText(file);
}

// ---- wiring ----
document.addEventListener('DOMContentLoaded', () => {
  $('#prev').onclick = () => go(-1);
  $('#next').onclick = () => go(1);
  $('#keepBtn').onclick = toggleKeep;
  $('#dropBtn').onclick = markDrop;
  $('#saveBtn').onclick = save;
  $('#gridBtn').onclick = openGrid;
  $('#resetBtn').onclick = resetAll;
  $('#undecidedBtn').onclick = nextUndecided;
  $('#closeGrid').onclick = () => $('#grid').classList.remove('open');
  $('#loadInput').onchange = e => {
    if (e.target.files[0]) loadDecisions(e.target.files[0]);
    e.target.value = '';
  };
  $('#strip').onclick = e => {
    const t = e.target.closest('.thumb');
    if (t) jump(+t.dataset.i);
  };
  $('#gridBody').onclick = e => {
    const t = e.target.closest('.cell');
    if (t) { jump(+t.dataset.i); $('#grid').classList.remove('open'); }
  };
  $('#groups').onclick = e => {
    const t = e.target.closest('.grp');
    if (t) gotoGroup(+t.dataset.g);
  };
  $('#filterBox').onchange = toggleFilter;
  $('#reason').onblur = () => { saveReason(); persist(); };

  document.addEventListener('keydown', e => {
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') {
      if (e.key === 'Escape') e.target.blur();
      return;
    }
    const k = e.key.toLowerCase();
    if (e.key === 'ArrowRight') { e.preventDefault(); go(1); }
    else if (e.key === 'ArrowLeft') { e.preventDefault(); go(-1); }
    else if (k === 'k' || e.key === ' ') { e.preventDefault(); toggleKeep(); }
    else if (k === 'x') { e.preventDefault(); markDrop(); }
    else if (k === 'g') { e.preventDefault();
      $('#grid').classList.contains('open')
        ? $('#grid').classList.remove('open') : openGrid(); }
    else if (k === 's') { e.preventDefault(); save(); }
    else if (k === 'u') { e.preventDefault(); nextUndecided(); }
    else if (e.key === '[') { e.preventDefault(); stepGroup(-1); }
    else if (e.key === ']') { e.preventDefault(); stepGroup(1); }
    else if (k === 'f') { e.preventDefault(); toggleFilter(); }
    else if (e.key === 'Escape') $('#grid').classList.remove('open');
  });

  render();
});
"""


def storage_id(rows):
    """Name the browser saves this page's decisions under.

    Hashed over the SORTED scene keys, so it tracks which scenes are in
    the worksheet and not what order they are shown in: regenerating the
    page, or changing GROUP_ORDER or the target, keeps your decisions,
    while a different shortlist starts clean.
    """
    if STORAGE_SCOPE != "auto":
        return str(STORAGE_SCOPE)
    sig = hashlib.sha1(
        "\n".join(sorted(r["_key"] for r in rows)).encode("utf-8")
    ).hexdigest()[:10]
    return f"{os.path.basename(os.path.abspath(BASE_DIR))}:{sig}"


def build_html(rows, cols, all_rows, groups):
    data = {
        "groups": groups,
        "id": storage_id(rows),
        "outName": OUTPUT_CSV_NAME,
        "meta": [f for f in META_FIELDS],
        "cols": cols,
        "rows": [{**{k: v for k, v in r.items() if not k.startswith("_")},
                  "_src": r["_src"], "_i": r["_i"], "_key": r["_key"],
                  "_g": r["_g"]} for r in rows],
        "all": [dict(r) for r in all_rows],
    }
    # A category or caption containing "</script>" would otherwise close
    # the script block early and break the page. Escaping the slash is
    # invisible to JSON.parse but no longer matches the HTML parser's
    # end-tag rule.
    data_json = json.dumps(data).replace("</", "<\\/")
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(TITLE)}</title>
<style>{CSS}</style>
</head>
<body>

<header>
  <h1>{html.escape(TITLE)}</h1>
  <span class="counts" id="pos"></span>
  <span class="counts" id="counts"></span>
  <span class="spacer"></span>
  <button id="undecidedBtn">Next undecided <kbd>U</kbd></button>
  <button id="gridBtn">Overview <kbd>G</kbd></button>
  <label class="hint" style="cursor:pointer">
    Load decisions
    <input id="loadInput" type="file" accept=".csv" hidden>
  </label>
  <button id="resetBtn">Reset</button>
  <button id="saveBtn" class="primary">Save CSV <kbd>S</kbd></button>
</header>

<main>
  <aside>
    <div class="asidehead">
      <span>categories · kept / total</span>
      <label style="cursor:pointer; display:flex; gap:5px;
                    align-items:center">
        <input id="filterBox" type="checkbox"> this one only <kbd>F</kbd>
      </label>
    </div>
    <div id="groups"></div>
  </aside>

  <div class="stage">
    <div class="frame undecided" id="frame"><img id="img" alt=""></div>
    <div class="meta" id="meta"></div>
    <div class="controls">
      <button id="prev">&larr; Previous</button>
      <button id="keepBtn" class="keep">Keep</button>
      <button id="dropBtn">Discard <kbd>X</kbd></button>
      <button id="next">Next &rarr;</button>
      <input id="reason" class="reason" placeholder="reason (optional)">
    </div>
    <div class="hint">
      <kbd>&larr;</kbd> <kbd>&rarr;</kbd> move &nbsp;·&nbsp;
      <kbd>[</kbd> <kbd>]</kbd> category &nbsp;·&nbsp;
      <kbd>K</kbd> or <kbd>space</kbd> keep &nbsp;·&nbsp;
      <kbd>X</kbd> discard &nbsp;·&nbsp;
      anything left undecided is saved as discarded
    </div>
  </div>
</main>

<footer><div id="strip"></div></footer>

<div class="grid" id="grid">
  <div class="bar">
    <b>Overview by category</b>
    <span class="hint">green = keep · grey = discarded · dashed =
      undecided · click a scene to jump to it</span>
    <span class="spacer"></span>
    <button id="closeGrid">Close <kbd>Esc</kbd></button>
  </div>
  <div id="gridBody"></div>
</div>

<script>
const DATA = {data_json};
const PRELOAD = {int(PRELOAD_AHEAD)};
const TARGET = {int(PER_CATEGORY_TARGET)};
{JS}
</script>
</body>
</html>
"""


def main():
    print("=" * 60)
    print("MAKE SCREENING PAGE")
    print("=" * 60)
    rows, cols, all_rows = load_rows()
    rows, groups = build_groups(rows)
    page = build_html(rows, cols, all_rows, groups)
    os.makedirs(os.path.dirname(os.path.abspath(OUTPUT_HTML)),
                exist_ok=True)
    with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
        f.write(page)
    size = os.path.getsize(OUTPUT_HTML) / 1024
    print(f"\n  wrote {OUTPUT_HTML}  ({size:.0f} kB, {len(rows)} scenes)")
    print(f"  decisions stored under '{storage_id(rows)}' -- a different "
          "shortlist")
    print("  gets its own, so this page opens with nothing selected.")
    print("\n  Open it in a browser. Decisions are kept as you go, so "
          "you can close")
    print("  the page and come back. Press Save to download "
          f"{OUTPUT_CSV_NAME},")
    print(f"  then move it next to {WORKSHEET} for step 3.")


if __name__ == "__main__":
    main()
