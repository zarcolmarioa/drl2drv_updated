"""
v4_captions_by_axis.py  (v4 axes: the Stage C captions, axis by axis)

Writes one self-contained HTML document (opens in any browser, works
offline) with the captions of every axis selected for CLIP, the sentence
CLIP encodes for each caption (the primary template, and optionally the
robustness variants), and for every caption the phrases behind it, each in its context sentence with the phrase
highlighted, its document, whether it is first-person, and its sense class.
The texts dropped in Stage C are listed under each axis, with their context
and the reason. Nothing is changed in the pipeline files.

Inputs
------
  STAGEV_DIR/stageA_verified.csv        contexts
  STAGEB_DIR/stageB_phrase_map.csv      phrase -> text, document, first
                                        person, sense class
  STAGEB_DIR/stageB_clip_selection.csv  axes selected for CLIP, definitions
  STAGEC_DIR/stageC_captions.csv        text -> caption, how decided
  STAGEC_DIR/stageC_axes.csv            support after Stage C

Output
------
  OUTPUT_FILE (default axes/v4/stageC/v4_captions_by_axis.html)

Usage
-----
    conda activate drl2drv
    python v4_captions_by_axis.py
"""

import html
import os
import re
import time
import unicodedata

import pandas as pd


# ============================================================
# SETTINGS
# ============================================================

STAGEV_DIR  = "axes/v4/stageV"
STAGEB_DIR  = "axes/v4/stageB"
STAGEC_DIR  = "axes/v4/stageC"
OUTPUT_FILE = "axes/v4/stageC/v4_captions_by_axis.html"
# axes shown but marked as dropped after Stage C (axis_id: reason)
DROPPED_AXES = {
    "A15": "Dropped after Stage C (5 October 2026): its vision texts all come "
           "from one document, the book's quotation of a mescaline account.",
}
SHOW_DROPPED_TEXTS = True     # list the texts dropped in Stage C under each axis
# the sentence CLIP encodes for each caption (step 05: the primary template;
# the others are the robustness variants, as for v3 in axes_v3.yaml)
TEMPLATES = {"scene": "a scene that looks {}",
             "photo": "a photo that looks {}",
             "place": "a place that looks {}",
             "world": "the world looks {}"}
PRIMARY_TEMPLATE = "scene"
SHOW_TEMPLATE_VARIANTS = True # also show the photo / place / world sentences


# ============================================================
# Helpers
# ============================================================

E = html.escape
CLASS_LABEL = {"vision": "vision", "not_specific": "not specific",
               "disputed": "disputed", "mixed": "mixed"}


def clean(t):
    t = unicodedata.normalize("NFKC", str(t))
    return " ".join(t.replace("’", "'").replace("‘", "'").split())


def highlight(quote, context):
    """The context with the quote in <mark>; the quote alone if it is not
    found in the context."""
    q, c = clean(quote), clean(context)
    ws = re.findall(r"\w+", q.lower())
    if c and ws:
        m = re.search(r"\W+".join(re.escape(w) for w in ws), c, flags=re.I)
        if m:
            return (E(c[:m.start()]) + "<mark>" + E(c[m.start():m.end()]) + "</mark>"
                    + E(c[m.end():]))
    return "<mark>" + E(q) + "</mark>" + (f' <span class="nf">(context: {E(c)})</span>'
                                          if c else "")


def sentence(template, caption):
    """The template with the caption in bold, as HTML."""
    pre, _, post = template.partition("{}")
    return f"{E(pre)}<b>{E(caption)}</b>{E(post)}"


def drop_reason(how):
    if "not_scene" in how:
        return "not a scene"
    if "outside" in how:
        return "outside the definition"
    if "cannot" in how:
        return "no caption possible"
    if how == "open":
        return "not yet decided"
    return "dropped"


CSS = """
:root{
 /* one reading column; sticky axis index on wide screens */
 --bg:#f5f6f4;--surface:#ffffff;--ink:#1d2321;--muted:#5d6763;--rule:#d9dedb;--accent:#2f6f62;--accent-soft:#e3efeb;--mark:#fbe7a8;
 --vision:#2d5f9a;--vision-soft:#e4ecf6;--ns:#5d6763;--ns-soft:#eceeed;--mixed:#8a5a12;--mixed-soft:#f5ead8;--warn:#9a3b2c;--warn-soft:#f6e4e0;
 --display:"Newsreader",Georgia,"Times New Roman",serif;--body:"IBM Plex Sans",system-ui,-apple-system,"Segoe UI",sans-serif;--mono:"IBM Plex Mono",ui-monospace,Menlo,Consolas,monospace}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#141816;--surface:#1b201e;--ink:#e6ebe8;--muted:#9aa5a0;--rule:#2e3633;--accent:#7cc2b1;--accent-soft:#1f302b;--mark:#5a4a14;
 --vision:#8db7ea;--vision-soft:#1c2a3b;--ns:#a3ada8;--ns-soft:#262c2a;--mixed:#e0b46a;--mixed-soft:#3a2e1b;--warn:#f0a090;--warn-soft:#3b221d;color-scheme:dark}}
:root[data-theme="dark"]{--bg:#141816;--surface:#1b201e;--ink:#e6ebe8;--muted:#9aa5a0;--rule:#2e3633;--accent:#7cc2b1;--accent-soft:#1f302b;--mark:#5a4a14;
 --vision:#8db7ea;--vision-soft:#1c2a3b;--ns:#a3ada8;--ns-soft:#262c2a;--mixed:#e0b46a;--mixed-soft:#3a2e1b;--warn:#f0a090;--warn-soft:#3b221d;color-scheme:dark}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 var(--body)}
.wrap{max-width:1180px;margin:0 auto;padding-inline:20px;padding-block:28px 64px;display:grid;grid-template-columns:220px minmax(0,1fr);gap:36px}
@media (max-width:880px){.wrap{grid-template-columns:minmax(0,1fr)}nav.index{position:static!important}}
h1{font:600 2.1rem/1.15 var(--display);margin:0 0 6px;text-wrap:balance}
.lede{color:var(--muted);max-width:70ch;margin:0 0 16px}
nav.index{position:sticky;top:16px;align-self:start;display:flex;flex-direction:column;gap:2px}
nav.index a{color:var(--ink);text-decoration:none;padding:5px 8px;border-radius:6px;font-size:.9rem}
nav.index a b{font:500 .78rem var(--mono);color:var(--accent);margin-right:6px}
nav.index a:hover,nav.index a:focus-visible{background:var(--accent-soft);outline:none}
nav.index a.struck{color:var(--muted);text-decoration:line-through}
.filters{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin:0 0 8px}
.filters span{font-size:.78rem;color:var(--muted);text-transform:uppercase;letter-spacing:.06em;margin-right:4px}
.filters button{font:inherit;font-size:.85rem;border:1px solid var(--rule);background:var(--surface);color:var(--ink);padding:4px 12px;border-radius:999px;cursor:pointer}
.filters button[aria-pressed="true"]{background:var(--accent);border-color:var(--accent);color:var(--bg)}
.filters button:focus-visible,summary:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
main{min-width:0;display:flex;flex-direction:column;gap:46px}
section.axis header{border-top:2px solid var(--ink);padding-top:12px;margin-bottom:14px}
section.dropped-axis header{border-top-color:var(--warn)}
h2{font:600 1.45rem/1.2 var(--display);margin:0 0 4px;display:flex;gap:10px;align-items:baseline;flex-wrap:wrap}
.aid{font:500 .85rem var(--mono);color:var(--accent)}
.def{margin:0 0 10px;color:var(--muted);max-width:72ch}
.warn{background:var(--warn-soft);color:var(--warn);padding:8px 12px;border-radius:6px;margin:0 0 10px;max-width:72ch}
.stats{display:flex;flex-wrap:wrap;gap:6px 26px;margin:0}
.stats div{display:flex;flex-direction:column}
.stats dt{font-size:.72rem;text-transform:uppercase;letter-spacing:.06em;color:var(--muted)}
.stats dd{margin:0;font:500 1rem var(--mono);font-variant-numeric:tabular-nums}
ol.caps{list-style:none;margin:0;padding:0;display:flex;flex-direction:column;gap:10px}
li.cap{background:var(--surface);border:1px solid var(--rule);border-radius:8px;padding:10px 14px;min-width:0}
li.cap.dim{opacity:.25}
.capline{display:flex;flex-wrap:wrap;gap:6px 12px;align-items:baseline}
.captext{font-weight:500;font-size:1.02rem}
.sentence{font-size:1.02rem;color:var(--muted)}
.sentence b{color:var(--ink);font-weight:500}
.variants{margin:4px 0 0;font-size:.8rem;color:var(--muted);display:flex;flex-wrap:wrap;gap:2px 16px}
.variants span::before{content:attr(data-t) ": ";font:.7rem var(--mono);text-transform:uppercase;letter-spacing:.05em}
.count{font:.75rem var(--mono);color:var(--muted)}
ul.phr{list-style:none;margin:8px 0 0;padding:0;display:flex;flex-direction:column;gap:6px}
ul.phr li{font-size:.88rem;color:var(--muted);border-left:2px solid var(--rule);padding-left:10px;overflow-wrap:anywhere}
ul.phr li .ctx{color:var(--ink)}
mark{background:var(--mark);color:var(--ink);padding:0 2px;border-radius:2px}
.meta{font:.72rem var(--mono);color:var(--muted);margin-left:6px;white-space:nowrap}
.nf{font-style:italic}
.chip{display:inline-block;font-size:.72rem;padding:1px 8px;border-radius:999px;white-space:nowrap}
.c-vision{background:var(--vision-soft);color:var(--vision)}.c-not_specific,.c-disputed{background:var(--ns-soft);color:var(--ns)}.c-mixed{background:var(--mixed-soft);color:var(--mixed)}
details{margin-top:12px}summary{cursor:pointer;color:var(--muted);font-size:.9rem}
.why{font:.72rem var(--mono);color:var(--warn);margin-right:6px}
@media print{nav.index,.filters{display:none}.wrap{display:block}li.cap{break-inside:avoid}details{display:block}}
"""

JS = """
const btns=[...document.querySelectorAll('.filters button')];
btns.forEach(b=>b.addEventListener('click',()=>{
  btns.forEach(x=>x.setAttribute('aria-pressed',x===b?'true':'false'));
  const f=b.dataset.f;
  document.querySelectorAll('li.cap').forEach(li=>{
    li.classList.toggle('dim',f!=='all'&&!li.dataset.cls.split(' ').includes(f));});
}));
"""


# ============================================================
# Main
# ============================================================

def main():
    t0 = time.time()
    pool = pd.read_csv(os.path.join(STAGEV_DIR, "stageA_verified.csv"), dtype=str).fillna("")
    ctx = dict(zip(pool.phrase_id, pool.context))
    pm = pd.read_csv(os.path.join(STAGEB_DIR, "stageB_phrase_map.csv"), dtype=str).fillna("")
    pm = pm[pm.included.str.lower() == "true"]
    sel = pd.read_csv(os.path.join(STAGEB_DIR, "stageB_clip_selection.csv"), dtype=str).fillna("")
    sel = sel[sel.clip_use.str.lower() == "yes"].set_index("axis_id")
    caps = pd.read_csv(os.path.join(STAGEC_DIR, "stageC_captions.csv"), dtype=str).fillna("")
    axs = pd.read_csv(os.path.join(STAGEC_DIR, "stageC_axes.csv"), dtype=str).fillna("") \
        .set_index("axis_id")
    phr = {t: g for t, g in pm.groupby("text_id")}

    def phrase_items(text_id):
        out = []
        g = phr.get(text_id)
        if g is None:
            return out
        for p in g.itertuples():
            fp = "first person" if p.first_person == "yes" else "not first person"
            out.append(f'<li><span class="ctx">“{highlight(p.quote, ctx.get(p.phrase_id, ""))}”</span>'
                       f'<span class="meta">{E(p.doc_id)} · {fp} · '
                       f'{CLASS_LABEL.get(p.sense_class, p.sense_class)}</span></li>')
        return out

    nav, sections, n_texts, n_caps = [], [], 0, set()
    for a in sel.index:
        g = caps[caps.axis_id == a]
        kept = g[g.caption != ""].copy()
        kept["key"] = kept.caption.str.lower().str.strip()
        dropped = g[g.caption == ""]
        if a not in DROPPED_AXES:
            n_texts += len(kept)
            n_caps |= {(a, k) for k in kept.key}
        items = []
        for k, gg in sorted(kept.groupby("key"), key=lambda kv: (-len(kv[1]), kv[0])):
            classes = sorted(set(gg.text_class))
            lis = [x for t in gg.text_id for x in phrase_items(t)]
            chips = "".join(f'<span class="chip c-{c}">{CLASS_LABEL.get(c, c)}</span>'
                            for c in classes)
            cap = gg.caption.iloc[0]
            variants = ""
            if SHOW_TEMPLATE_VARIANTS:
                variants = '<div class="variants">' + "".join(
                    f'<span data-t="{E(n)}">{sentence(t, cap)}</span>'
                    for n, t in TEMPLATES.items() if n != PRIMARY_TEMPLATE) + "</div>"
            items.append(
                f'<li class="cap" data-cls="{" ".join(classes)}"><div class="capline">'
                f'<span class="sentence">{sentence(TEMPLATES[PRIMARY_TEMPLATE], cap)}</span>'
                f'<span class="count">{len(gg)} text{"s" if len(gg) > 1 else ""}, '
                f'{len(lis)} phrase{"s" if len(lis) != 1 else ""}</span>{chips}</div>'
                f'{variants}<ul class="phr">{"".join(lis)}</ul></li>')
        drop_html = ""
        if SHOW_DROPPED_TEXTS:
            dl = []
            for x in dropped.itertuples():
                lis = phrase_items(x.text_id)
                dl.append(f'<li class="cap"><div class="capline"><span class="why">'
                          f'{drop_reason(x.how)}</span><span>“{E(x.text)}”</span></div>'
                          f'<ul class="phr">{"".join(lis)}</ul></li>')
            drop_html = (f'<details><summary>Dropped in Stage C ({len(dropped)})</summary>'
                         f'<ol class="caps" style="margin-top:8px">{"".join(dl) or "<li>none</li>"}'
                         f'</ol></details>')
        s = axs.loc[a] if a in axs.index else None

        def flag(v):
            return "✓" if str(v) == "True" else "–"
        stats = ""
        if s is not None:
            stats = (f'<dl class="stats"><div><dt>texts</dt><dd>{len(kept)}</dd></div>'
                     f'<div><dt>distinct captions</dt><dd>{kept.key.nunique()}</dd></div>'
                     f'<div><dt>documents</dt><dd>{s.all_docs}</dd></div>'
                     f'<div><dt>first-person phrases</dt><dd>{s.all_first_person}</dd></div>'
                     f'<div><dt>kept: all / vision / not specific</dt><dd>{flag(s.all_kept)} / '
                     f'{flag(s.vision_kept)} / {flag(s.not_specific_kept)}</dd></div></dl>')
        warn = f'<p class="warn">{E(DROPPED_AXES[a])}</p>' if a in DROPPED_AXES else ""
        nav.append(f'<a href="#{a}" class="{"struck" if a in DROPPED_AXES else ""}">'
                   f'<b>{a}</b>{E(sel.loc[a, "name_stageC"])}</a>')
        sections.append(
            f'<section id="{a}" class="axis{" dropped-axis" if a in DROPPED_AXES else ""}">'
            f'<header><h2><span class="aid">{a}</span>{E(sel.loc[a, "name_stageC"])}</h2>'
            f'<p class="def">{E(sel.loc[a, "definition_stageC"])}</p>{warn}{stats}</header>'
            f'<ol class="caps">{"".join(items)}</ol>{drop_html}</section>')

    kept_axes = [a for a in sel.index if a not in DROPPED_AXES]
    doc = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>v4 Captions by Axis</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500&family=Newsreader:opsz,wght@6..72,600&display=swap">
<style>{CSS}</style></head><body>
<div class="wrap">
<nav class="index" aria-label="Axes">{"".join(nav)}</nav>
<main>
<div><h1>v4 captions by axis</h1>
<p class="lede">The Stage C captions of the axes selected for CLIP, generated
{time.strftime('%d %B %Y, %H:%M')}. Each entry shows the sentence CLIP encodes, with the
caption in bold (primary template: “{E(TEMPLATES[PRIMARY_TEMPLATE].format('…'))}”{"; below it the robustness templates" if SHOW_TEMPLATE_VARIANTS else ""});
identical captions from different texts are one entry (one anchor in Stage D).
Under each caption: every phrase behind it, in its context sentence with the
phrase highlighted, its document, whether it is first-person, and its sense class.
{len(kept_axes)} axes ({", ".join(kept_axes)}): {n_texts} texts,
{len(n_caps)} distinct captions.</p>
<div class="filters" role="group" aria-label="Highlight by sense class"><span>Highlight</span>
<button data-f="all" aria-pressed="true">all</button>
<button data-f="vision" aria-pressed="false">vision</button>
<button data-f="not_specific" aria-pressed="false">not specific</button>
<button data-f="mixed" aria-pressed="false">mixed</button></div></div>
{"".join(sections)}
</main></div>
<script>{JS}</script>
</body></html>
"""
    os.makedirs(os.path.dirname(OUTPUT_FILE) or ".", exist_ok=True)
    open(OUTPUT_FILE, "w", encoding="utf-8").write(doc)
    print(f"Wrote {OUTPUT_FILE}: {len(sel)} axes, {n_texts} texts "
          f"({time.time() - t0:.1f} s). Open it in a browser.")


if __name__ == "__main__":
    main()
