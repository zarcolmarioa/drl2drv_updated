"""
03_encode_images.py  (pipeline step 03)

Encodes every image in image_index.csv (step 02) with un2CLIP and
saves, row-aligned with `image_id`:

  * the 1024-d pooled image representation BEFORE the projection
    (pooler_output = post-layernorm CLS token of the final layer), and
  * the 768-d L2-normalised image embedding AFTER un2CLIP's own
    projection, i.e. the vector the text axes are compared with.

Only the final, text-aligned layer is encoded. Axes and projections
are built in steps 05-06; low-level features in step 04.

What changed from the old Script A (new_encode_and_save.py)
-----------------------------------------------------------
  * FIX A1 (critical). un2CLIP fine-tunes the whole image encoder,
    INCLUDING the final image projection `proj`. The official loader
    (github.com/LiYinqi/un2CLIP) loads it with
        model.visual.load_state_dict(state_dict)
    The old key remapping discarded `proj` and kept the standard
    OpenAI projection (relative difference 3.4e-3; embeddings differed
    at cosine ~0.9997). Here `proj` is loaded, and the start-up checks
    verify it.
  * FIX A2. Only the final layer. Layers 3 and 18 were not in the
    text-aligned space.
  * The 1024-d pre-projection vectors are saved too, so a future change
    of projection never requires re-encoding. They also allow the
    magnitude-based control (MA-CLIP, review section 6).
  * FIX A7. Outputs are written by image_id into pre-allocated arrays.
    Resuming skips images already marked done, and refuses to resume if
    the image index has changed (fingerprint check). A final check
    re-encodes random images and compares them with the stored rows, so
    a misalignment cannot go unnoticed.
  * FIX A8. Unreadable images get NaN rows and status -1, never a blank
    placeholder embedding.
  * Faster: images are decoded and preprocessed in parallel worker
    processes; only the model runs on the GPU.
  * The text encoder is not loaded (axes are step 05).

Preprocessing (PREPROCESS setting)
----------------------------------
"openai" (default) reproduces the official OpenAI/un2CLIP preprocessing
exactly: torchvision Resize(336, bicubic) on the PIL image, CenterCrop
336, convert to RGB, ToTensor, CLIP normalisation -- the same code as
clip._transform, which is how CLIP and un2CLIP were trained and
evaluated.

"hf" uses HuggingFace's CLIPImageProcessor, as the old pipeline did.
Recent transformers versions run it on torchvision tensors rather than
PIL images; the bicubic resize then differs slightly, and the test run
showed embeddings at cosine 0.9977 from the official ones on the same
images -- a larger deviation than the projection error being fixed.
Kept only for comparison with the old results.

Start-up check 3b compares the chosen preprocessing with the official
one and should give cosine 1.000000 with "openai".

Start-up checks (the run stops if a critical one fails)
-------------------------------------------------------
  1. WEIGHTS     every vision-tower weight in the checkpoint was loaded;
                 nothing unexpected; nothing missing except buffers.
  2. PROJECTION  the model's projection equals the checkpoint's `proj`.
  3. OFFICIAL    (if the OpenAI `clip` package is installed) on
                 N_CHECK_IMAGES images, embeddings match the official
                 un2CLIP loader:
                   same pixels             cosine >= SAME_PIXELS_MIN_COS
                   each own preprocessing  cosine >= OWN_PREPROC_MIN_COS
                 Install with:
                   pip install ftfy regex
                   pip install git+https://github.com/openai/CLIP.git
                 Without it, check 3 is skipped with a warning (checks
                 1-2 still guarantee the projection fix). With
                 PREPROCESS = "openai", 3b is expected to be 1.000000.

End-of-run check
----------------
  4. ALIGNMENT   FINAL_RECHECK_N random images are re-encoded one by one
                 and compared with their stored rows (cosine >= 0.99999).

Inputs
------
  INDEX_CSV            data/selection/image_index.csv from step 02
  UN2CLIP_CKPT_PATH    openai_vit_l_14_336.ckpt (un2CLIP)
  CLIP_MODEL_NAME      openai/clip-vit-large-patch14-336 (HuggingFace)

Outputs (under OUTPUT_DIR)
--------------------------
  pooled_1024.npy      (N, 1024) POOLED_DTYPE   pre-projection vectors
  embeddings_768.npy   (N, 768)  float32        L2-normalised, un2CLIP proj
  status.npy           (N,)      int8           0 not done, 1 ok, -1 unreadable
  image_ids.npy        (N,)      int64          image_id of each row
  run_meta.json        settings, model/checkpoint info, index fingerprint,
                       check results (also used to validate a resume)
  failed_images.csv    image_id and path of unreadable images (if any)
  encode_report.txt    human-readable summary

Row i of every array is image_id i of image_index.csv.

Resume
------
Just run the script again. It opens the existing arrays, skips rows with
status != 0, and continues. It refuses to resume if image_index.csv or
the key settings changed; delete OUTPUT_DIR (or choose a new one) to
start over. Progress is saved every FLUSH_EVERY_BATCHES batches, so an
interruption loses at most a few thousand images.

Usage
-----
    Edit SETTINGS, then:
        conda activate drl2drv
        python 03_encode_images.py

Recommended: a dry run first with TEST_LIMIT = 2000 and a separate
OUTPUT_DIR (a minute or two), then the full run.
"""

import hashlib
import json
import os
import time
import warnings

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import CLIPImageProcessor, CLIPModel


# ============================================================
# SETTINGS
# ============================================================

# --- Inputs ---
INDEX_CSV           = "data/selection/image_index.csv"
CLIP_MODEL_NAME     = "openai/clip-vit-large-patch14-336"
UN2CLIP_CKPT_PATH   = "./pretrained_models/openai_vit_l_14_336.ckpt"

# Name used by the official `clip` package for the same base model
# (start-up check 3 only).
OPENAI_MODEL_NAME   = "ViT-L/14@336px"

# --- Output ---
OUTPUT_DIR          = "data/embeddings"

# --- Preprocessing ---
# "openai"  identical to the official OpenAI / un2CLIP preprocessing
#           (recommended; see the docstring).
# "hf"      HuggingFace CLIPImageProcessor, as in the old pipeline.
PREPROCESS          = "openai"

# --- Dry run ---
# Encode only the first TEST_LIMIT images of the index (None = all).
# Use a separate OUTPUT_DIR for dry runs.
TEST_LIMIT          = None

# --- Throughput ---
# BATCH_SIZE    images per forward pass. 64 fits comfortably in 16 GB in
#               fp32 for ViT-L/14-336.
# NUM_WORKERS   processes decoding and preprocessing images. Decoding the
#               full-resolution Places images is usually the bottleneck,
#               so set this close to the number of CPU cores (leave one or
#               two free).
BATCH_SIZE          = 64
NUM_WORKERS         = 8
PREFETCH_FACTOR     = 4

# --- Numerics ---
# Full fp32 without TF32, as in the old pipeline, so results are
# comparable with the validation work. (TF32 would speed up the first
# convolution slightly but change the numbers in the 3rd-4th digit.)
ALLOW_TF32          = False

# Storage type of the 1024-d pre-projection vectors. float16 halves the
# file (~2.3 GB for 1.1M images); recomputing a projection from float16
# changes embeddings by < 1e-5 in cosine.
POOLED_DTYPE        = "float16"

# --- Progress saving ---
FLUSH_EVERY_BATCHES = 50

# --- Checks ---
N_CHECK_IMAGES      = 16
SAME_PIXELS_MIN_COS = 0.9999    # same input tensor into both models
OWN_PREPROC_MIN_COS = 0.99999   # each pipeline with its own preprocessing
FINAL_RECHECK_N     = 64
FINAL_RECHECK_MIN   = 0.99999
RANDOM_SEED         = 42

# Compute the SHA-256 of the checkpoint for the record (a few seconds).
HASH_CHECKPOINT     = True

DEVICE              = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# Model loading (un2CLIP on HuggingFace CLIP)
# ============================================================

def load_un2clip_state_dict(path):
    """The checkpoint is the complete OpenAI-format visual tower
    (model.visual.state_dict()), possibly wrapped or prefixed."""
    raw = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(raw, dict) and "state_dict" in raw:
        raw = raw["state_dict"]
    elif isinstance(raw, dict) and "model" in raw:
        raw = raw["model"]
    out = {}
    for k, v in raw.items():
        if k.startswith("visual."):
            k = k[len("visual."):]
        out[k] = v
    return out


def remap_openai_to_hf(sd):
    """OpenAI visual-tower keys -> HuggingFace CLIPVisionTransformer keys.

    Returns (hf_state_dict, proj, unused_keys). `proj` (d_model x 768)
    is returned separately because in HuggingFace it lives outside the
    vision tower, as CLIPModel.visual_projection (weight = proj.T).
    d_model is read from the checkpoint, not assumed.
    """
    d_model = sd["class_embedding"].shape[0]
    hf, unused = {}, []

    def block(rest, v):
        n, sub = rest.split(".", 1)
        p = f"encoder.layers.{n}."
        if sub in ("attn.in_proj_weight", "attn.in_proj_bias"):
            q, k, vv = v.split(d_model, dim=0)      # rows ordered q, k, v
            kind = "weight" if sub.endswith("weight") else "bias"
            return {p + f"self_attn.q_proj.{kind}": q,
                    p + f"self_attn.k_proj.{kind}": k,
                    p + f"self_attn.v_proj.{kind}": vv}
        for a, b in (("attn.out_proj.", "self_attn.out_proj."),
                     ("ln_1.", "layer_norm1."), ("ln_2.", "layer_norm2."),
                     ("mlp.c_fc.", "mlp.fc1."), ("mlp.c_proj.", "mlp.fc2.")):
            if sub.startswith(a):
                return {p + b + sub[len(a):]: v}
        return None

    for k, v in sd.items():
        if k == "proj":
            continue
        if k.startswith("transformer.resblocks."):
            r = block(k[len("transformer.resblocks."):], v)
        elif k.startswith("resblocks."):
            r = block(k[len("resblocks."):], v)
        elif k == "class_embedding":
            r = {"embeddings.class_embedding": v}
        elif k == "positional_embedding":
            r = {"embeddings.position_embedding.weight": v}
        elif k == "conv1.weight":
            r = {"embeddings.patch_embedding.weight": v}
        elif k.startswith("ln_pre."):
            r = {"pre_layrnorm." + k[len("ln_pre."):]: v}
        elif k.startswith("ln_post."):
            r = {"post_layernorm." + k[len("ln_post."):]: v}
        else:
            r = None
        if r is None:
            unused.append(k)
        else:
            hf.update(r)
    if "proj" not in sd:
        raise KeyError("un2CLIP checkpoint has no 'proj' key.")
    return hf, sd["proj"], unused


def load_model():
    """HuggingFace CLIP with the complete un2CLIP visual tower.

    Returns (model, processor, raw_state_dict, check_results).
    """
    print(f"  base model : {CLIP_MODEL_NAME}")
    model = CLIPModel.from_pretrained(CLIP_MODEL_NAME, use_safetensors=True)
    processor = CLIPImageProcessor.from_pretrained(CLIP_MODEL_NAME)
    # The text tower is not needed in this step.
    model.text_model = None

    print(f"  un2CLIP    : {UN2CLIP_CKPT_PATH}")
    if not os.path.exists(UN2CLIP_CKPT_PATH):
        raise FileNotFoundError(UN2CLIP_CKPT_PATH)
    sd = load_un2clip_state_dict(UN2CLIP_CKPT_PATH)
    hf_sd, proj, unused = remap_openai_to_hf(sd)

    # --- Check 1: every weight loaded ---
    missing, unexpected = model.vision_model.load_state_dict(hf_sd,
                                                             strict=False)
    buffers = {n for n, _ in model.vision_model.named_buffers()}
    missing_w = [k for k in missing if k not in buffers]
    ok1 = not missing_w and not unexpected and not unused
    print(f"  check 1 (weights): {len(hf_sd)} tensors loaded; "
          f"missing weights {len(missing_w)}, unexpected "
          f"{len(unexpected)}, unused checkpoint keys {len(unused)} "
          f"-> {'PASS' if ok1 else 'FAIL'}")
    if not ok1:
        raise RuntimeError(f"Weight loading incomplete.\n  missing: "
                           f"{missing_w[:5]}\n  unexpected: "
                           f"{unexpected[:5]}\n  unused: {unused[:5]}")

    # --- The fix: use un2CLIP's projection ---
    with torch.no_grad():
        if tuple(proj.T.shape) != tuple(model.visual_projection.weight.shape):
            raise RuntimeError(
                f"proj shape {tuple(proj.shape)} does not fit "
                f"visual_projection {tuple(model.visual_projection.weight.shape)}")
        std_w = model.visual_projection.weight.detach().clone()
        model.visual_projection.weight.copy_(proj.T)

    # --- Check 2: projection in place ---
    w = model.visual_projection.weight.detach()
    ok2 = torch.equal(w, proj.T.to(w.dtype))
    changed = (proj.T.float() - std_w.float()).abs().max().item()
    print(f"  check 2 (projection): un2CLIP proj applied "
          f"(differs from the standard one by up to {changed:.2e}) "
          f"-> {'PASS' if ok2 else 'FAIL'}")
    if not ok2:
        raise RuntimeError("visual_projection does not equal un2CLIP proj.")

    model = model.to(DEVICE).eval()
    checks = {"weights_loaded": len(hf_sd), "missing_weights": missing_w,
              "unexpected": list(unexpected), "unused_ckpt_keys": unused,
              "proj_applied": ok2,
              "proj_max_diff_vs_standard": changed}
    return model, processor, sd, checks


@torch.inference_mode()
def embed(model, pixels):
    """(pooled_1024, embedding_768_normalised) for a pixel batch."""
    pooled = model.vision_model(pixel_values=pixels).pooler_output
    emb = model.visual_projection(pooled)
    emb = emb / emb.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    return pooled, emb


# ============================================================
# Start-up check 3: against the official un2CLIP loader
# ============================================================

def official_check(model, preprocess, sd, idx):
    try:
        import clip
    except ImportError:
        print("  check 3 (official loader): SKIPPED -- OpenAI 'clip' "
              "package not installed.")
        return {"official_check": "skipped"}

    ref, ref_pre = clip.load(OPENAI_MODEL_NAME, device="cpu", jit=False)
    ref.visual.load_state_dict(sd)                  # strict, as un2CLIP does
    ref = ref.float().to(DEVICE).eval()

    imgs = []
    for p in idx["path"]:
        try:
            with Image.open(p) as im:
                imgs.append(im.convert("RGB").copy())
        except Exception:
            continue
        if len(imgs) >= N_CHECK_IMAGES:
            break

    def cos(a, b):
        a = a.float() / a.float().norm(dim=-1, keepdim=True)
        b = b.float() / b.float().norm(dim=-1, keepdim=True)
        return (a * b).sum(-1).cpu().numpy()

    with torch.inference_mode():
        px_ref = torch.stack([ref_pre(im) for im in imgs]).to(DEVICE)
        e_ref = ref.encode_image(px_ref)
        _, e_same = embed(model, px_ref)
        px_own = torch.stack([preprocess(im) for im in imgs]).to(DEVICE)
        _, e_own = embed(model, px_own)
    c_same, c_own = cos(e_ref, e_same), cos(e_ref, e_own)
    ok_same = c_same.min() >= SAME_PIXELS_MIN_COS
    ok_own = c_own.min() >= OWN_PREPROC_MIN_COS
    print(f"  check 3a (official, same pixels): min cos "
          f"{c_same.min():.6f} -> {'PASS' if ok_same else 'FAIL'}")
    print(f"  check 3b (official vs '{PREPROCESS}' preprocessing): min cos "
          f"{c_own.min():.6f} -> {'PASS' if ok_own else 'WARN'}")
    del ref
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    if not ok_same:
        raise RuntimeError("Embeddings differ from the official un2CLIP "
                           "loader on identical pixels. Stopping.")
    if not ok_own:
        warnings.warn("Preprocessing differs from the official one more "
                      "than expected; inspect before relying on results.")
    return {"official_check": "run",
            "same_pixels_min_cos": float(c_same.min()),
            "own_preproc_min_cos": float(c_own.min())}


# ============================================================
# Preprocessing
# ============================================================

def _to_rgb(im):
    return im.convert("RGB")


class OpenAIPreprocess:
    """Exactly clip._transform(n_px): Resize (bicubic, on the PIL image)
    -> CenterCrop -> RGB -> ToTensor -> Normalize. A class, not a
    closure, so worker processes can pickle it."""

    def __init__(self, n_px, mean, std):
        try:
            from torchvision.transforms import (CenterCrop, Compose,
                                                InterpolationMode,
                                                Normalize, Resize,
                                                ToTensor)
        except ImportError as e:
            raise ImportError("PREPROCESS = 'openai' needs torchvision "
                              "(pip install torchvision).") from e
        self.t = Compose([
            Resize(n_px, interpolation=InterpolationMode.BICUBIC),
            CenterCrop(n_px),
            _to_rgb,
            ToTensor(),
            Normalize(tuple(mean), tuple(std)),
        ])

    def __call__(self, im):
        return self.t(im)


class HFPreprocess:
    """HuggingFace CLIPImageProcessor, as in the old pipeline."""

    def __init__(self, processor):
        self.processor = processor

    def __call__(self, im):
        return self.processor(images=im.convert("RGB"),
                              return_tensors="pt")["pixel_values"][0]


def make_preprocess(processor):
    n_px = processor.crop_size["height"]
    if PREPROCESS == "openai":
        return OpenAIPreprocess(n_px, processor.image_mean,
                                processor.image_std), n_px
    if PREPROCESS == "hf":
        return HFPreprocess(processor), n_px
    raise ValueError(f"Unknown PREPROCESS: {PREPROCESS!r}")


# ============================================================
# Data
# ============================================================

class ImageSet(Dataset):
    """Decodes and preprocesses one image; never raises."""

    def __init__(self, ids, paths, preprocess, size):
        self.ids, self.paths = ids, paths
        self.preprocess = preprocess
        self.size = size

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        try:
            with Image.open(self.paths[i]) as im:
                px = self.preprocess(im)
            ok = True
        except Exception:
            px = torch.zeros(3, self.size, self.size)
            ok = False
        return int(self.ids[i]), px, ok


def collate(batch):
    ids, px, ok = zip(*batch)
    return (torch.tensor(ids, dtype=torch.long), torch.stack(px),
            torch.tensor(ok, dtype=torch.bool))


# ============================================================
# Output arrays, fingerprint, resume
# ============================================================

def fingerprint(idx):
    h = hashlib.sha1()
    for k, p in zip(idx["image_key"], idx["path"]):
        h.update(f"{k}\t{p}\n".encode("utf-8"))
    return h.hexdigest()


def sha256(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def save_status(status):
    tmp = os.path.join(OUTPUT_DIR, "status.tmp.npy")
    np.save(tmp, status)
    os.replace(tmp, os.path.join(OUTPUT_DIR, "status.npy"))


def open_outputs(n, d_pool, d_emb, meta):
    """Create or reopen the output arrays. Returns (pooled, emb, status,
    resumed)."""
    p_pool = os.path.join(OUTPUT_DIR, "pooled_1024.npy")
    p_emb = os.path.join(OUTPUT_DIR, "embeddings_768.npy")
    p_stat = os.path.join(OUTPUT_DIR, "status.npy")
    p_meta = os.path.join(OUTPUT_DIR, "run_meta.json")
    fmt = np.lib.format

    if os.path.exists(p_meta) and os.path.exists(p_stat):
        with open(p_meta) as f:
            old = json.load(f)
        keys = ["index_fingerprint", "n_images", "clip_model_name",
                "un2clip_ckpt_path", "pooled_dtype", "allow_tf32",
                "preprocessing"]
        diffs = [k for k in keys if old.get(k) != meta.get(k)]
        if diffs:
            raise RuntimeError(
                f"Existing run in {OUTPUT_DIR} has different {diffs}. "
                "Refusing to resume; use a new OUTPUT_DIR or delete it.")
        pooled = fmt.open_memmap(p_pool, mode="r+")
        emb = fmt.open_memmap(p_emb, mode="r+")
        status = np.load(p_stat)
        return pooled, emb, status, True

    pooled = fmt.open_memmap(p_pool, mode="w+", dtype=POOLED_DTYPE,
                             shape=(n, d_pool))
    emb = fmt.open_memmap(p_emb, mode="w+", dtype=np.float32,
                          shape=(n, d_emb))
    status = np.zeros(n, dtype=np.int8)
    save_status(status)
    return pooled, emb, status, False


# ============================================================
# Main
# ============================================================

def main():
    t0 = time.time()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = ALLOW_TF32
    torch.backends.cudnn.allow_tf32 = ALLOW_TF32
    warnings.filterwarnings("ignore", category=Image.DecompressionBombWarning)

    print("=" * 70)
    print("03_encode_images.py")
    print("=" * 70)
    print(f"Index   : {INDEX_CSV}")
    print(f"Output  : {OUTPUT_DIR}")
    print(f"Device  : {DEVICE}")

    # --- Index ---
    idx = pd.read_csv(INDEX_CSV)
    if TEST_LIMIT is not None:
        idx = idx.iloc[:TEST_LIMIT].copy()
    if not (idx["image_id"].to_numpy() == np.arange(len(idx))).all():
        raise RuntimeError("image_id must be 0..N-1 in row order "
                           "(as written by step 02).")
    n = len(idx)
    print(f"Images  : {n:,}" + (f"  (TEST_LIMIT)" if TEST_LIMIT else ""))

    # --- Model and start-up checks ---
    print("\nLoading model and running start-up checks...")
    model, processor, sd, checks = load_model()
    preprocess, size = make_preprocess(processor)
    checks.update(official_check(model, preprocess, sd, idx))
    d_pool = model.visual_projection.weight.shape[1]
    d_emb = model.visual_projection.weight.shape[0]

    meta = {
        "script": "03_encode_images.py",
        "index_csv": INDEX_CSV,
        "index_fingerprint": fingerprint(idx),
        "n_images": n,
        "clip_model_name": CLIP_MODEL_NAME,
        "un2clip_ckpt_path": UN2CLIP_CKPT_PATH,
        "un2clip_ckpt_sha256": (sha256(UN2CLIP_CKPT_PATH)
                                if HASH_CHECKPOINT else None),
        "pooled_dtype": POOLED_DTYPE,
        "allow_tf32": ALLOW_TF32,
        "layer": "final (pooler_output, post-layernorm CLS)",
        "preprocessing": {"mode": PREPROCESS,
                          "size": size,
                          "mean": list(processor.image_mean),
                          "std": list(processor.image_std)},
        "dims": {"pooled": d_pool, "embedding": d_emb},
        "torch": torch.__version__,
        "checks": checks,
    }

    pooled, emb, status, resumed = open_outputs(n, d_pool, d_emb, meta)
    if resumed:
        print(f"\nResuming: {int((status != 0).sum()):,} of {n:,} "
              "already done.")
        with open(os.path.join(OUTPUT_DIR, "run_meta.json")) as f:
            meta["started"] = json.load(f).get("started")
    else:
        meta["started"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(os.path.join(OUTPUT_DIR, "run_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    np.save(os.path.join(OUTPUT_DIR, "image_ids.npy"),
            idx["image_id"].to_numpy(np.int64))

    # --- Encode remaining images ---
    todo = np.where(status == 0)[0]
    print(f"To encode: {len(todo):,}\n")
    t_enc = time.time()
    if len(todo):
        ds = ImageSet(todo, idx["path"].to_numpy()[todo], preprocess, size)
        dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS, collate_fn=collate,
                        pin_memory=(DEVICE == "cuda"),
                        persistent_workers=NUM_WORKERS > 0,
                        prefetch_factor=(PREFETCH_FACTOR if NUM_WORKERS > 0
                                         else None))
        n_batches = 0
        try:
            for ids, px, ok in tqdm(dl, desc="encoding", unit="batch"):
                ids = ids.numpy()
                ok = ok.numpy()
                p, e = embed(model, px.to(DEVICE, non_blocking=True))
                p = p.float().cpu().numpy()
                e = e.float().cpu().numpy()
                p[~ok] = np.nan
                e[~ok] = np.nan
                pooled[ids] = p.astype(POOLED_DTYPE)
                emb[ids] = e
                status[ids] = np.where(ok, 1, -1).astype(np.int8)
                n_batches += 1
                if n_batches % FLUSH_EVERY_BATCHES == 0:
                    pooled.flush()
                    emb.flush()
                    save_status(status)
        finally:
            # Also runs on Ctrl-C or an error: keep what is done.
            pooled.flush()
            emb.flush()
            save_status(status)
    t_enc = time.time() - t_enc
    rate = len(todo) / t_enc if t_enc > 0 and len(todo) else 0.0

    # --- Check 4: alignment ---
    print("\nFinal check: re-encoding random images one by one...")
    rng = np.random.default_rng(RANDOM_SEED)
    ok_rows = np.where(status == 1)[0]
    pick = rng.choice(ok_rows, size=min(FINAL_RECHECK_N, len(ok_rows)),
                      replace=False)
    cs = []
    for i in pick:
        with Image.open(idx["path"].iat[i]) as im:
            px = preprocess(im).unsqueeze(0).to(DEVICE)
        _, e = embed(model, px)
        cs.append(float(e[0].float().cpu().numpy() @ emb[i]))
    cs = np.array(cs)
    ok4 = cs.min() >= FINAL_RECHECK_MIN
    print(f"  check 4 (alignment): {len(cs)} images, min cos "
          f"{cs.min():.7f} -> {'PASS' if ok4 else 'FAIL'}")

    # --- Failed images ---
    failed = np.where(status == -1)[0]
    if len(failed):
        idx.iloc[failed][["image_id", "path"]].to_csv(
            os.path.join(OUTPUT_DIR, "failed_images.csv"), index=False)

    # --- Report ---
    done = int((status != 0).sum())
    norms = np.linalg.norm(emb[ok_rows[:10000]], axis=1) if len(ok_rows) \
        else np.array([np.nan])
    meta.update({
        "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_ok": int((status == 1).sum()), "n_failed": int(len(failed)),
        "n_not_done": int((status == 0).sum()),
        "images_per_second_this_run": round(rate, 1),
        "final_recheck_min_cos": float(cs.min()),
        "final_recheck_pass": bool(ok4),
    })
    with open(os.path.join(OUTPUT_DIR, "run_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    lines = [
        "03_encode_images.py -- summary",
        "=" * 60,
        f"Started / finished : {meta['started']} / {meta['finished']}",
        f"Index              : {INDEX_CSV}  ({n:,} images)",
        f"Model              : {CLIP_MODEL_NAME} + un2CLIP "
        f"({os.path.basename(UN2CLIP_CKPT_PATH)})",
        f"Layer              : final (pooler_output), text-aligned",
        f"Precision          : fp32, TF32 {'on' if ALLOW_TF32 else 'off'}",
        f"Preprocessing      : {PREPROCESS}",
        "",
        "Start-up checks",
        f"  1 weights loaded     : PASS ({checks['weights_loaded']} tensors)",
        f"  2 un2CLIP projection : PASS (max diff vs standard "
        f"{checks['proj_max_diff_vs_standard']:.2e})",
        ("  3 official loader    : skipped (clip package not installed)"
         if checks.get("official_check") == "skipped" else
         f"  3 official loader    : same pixels min cos "
         f"{checks['same_pixels_min_cos']:.6f}; {PREPROCESS} preprocessing "
         f"{checks['own_preproc_min_cos']:.6f}"),
        "",
        "Encoding",
        f"  done           : {done:,} / {n:,}",
        f"  ok             : {meta['n_ok']:,}",
        f"  unreadable     : {meta['n_failed']:,}"
        + ("  (failed_images.csv)" if len(failed) else ""),
        f"  this run       : {len(todo):,} images in {t_enc / 3600:.2f} h "
        f"({rate:.1f} img/s)",
        f"  embedding norm : {np.nanmin(norms):.6f} - {np.nanmax(norms):.6f}"
        " (should be 1)",
        "",
        f"End-of-run check 4 (alignment): min cos {cs.min():.7f} -> "
        f"{'PASS' if ok4 else 'FAIL'}",
        "",
        "Files",
        f"  pooled_1024.npy     ({n}, {d_pool}) {POOLED_DTYPE}",
        f"  embeddings_768.npy  ({n}, {d_emb}) float32",
        f"  status.npy, image_ids.npy, run_meta.json",
        "",
        f"Total time: {(time.time() - t0) / 3600:.2f} h",
    ]
    if meta["n_not_done"]:
        lines.insert(-1, f"NOTE: {meta['n_not_done']:,} images not done "
                         "yet; run the script again to continue.")
    report = "\n".join(lines)
    with open(os.path.join(OUTPUT_DIR, "encode_report.txt"), "w") as f:
        f.write(report + "\n")
    print("\n" + report)
    if not ok4:
        raise RuntimeError("Alignment check failed: stored rows do not "
                           "match re-encoded images.")


if __name__ == "__main__":
    main()
