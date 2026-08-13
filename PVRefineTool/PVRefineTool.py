"""
PVRefineTool
=============================================

Runs patch-based inference with a fine-tuned MedSAM model (built on top of
Segment Anything, ViT-B). For each input image / mask pair, the mask is
optionally degraded (to simulate coarse or bounding-box annotations) and
then refined by the model through two rounds of prompt-encoder / mask-decoder
passes. Large images are split into 1024x1024 patches, processed
independently, and stitched back together.

Outputs, per image:
    <name>_refined.png  - model-refined segmentation mask (0/255)
    <name>_gt.png        - normalized ground-truth mask (0/255)

A per-module timing summary (image encoder / decoder passes / post-process)
is printed at the end and written to `inference_timing.csv` in the output
directory.

Usage
-----
    python PVRefineTool.py \
        --data_path /path/to/images_and_labels \
        --seg_path /path/to/output_dir \
        --checkpoint /path/to/medsam_model_best.pth \
        --base_sam_checkpoint /path/to/sam_vit_b.pth \
        --device cuda:0

Expected input layout
----------------------
`--data_path` should contain paired files:
    <name>.bmp
    <name>_label.bmp

See README.md for details on obtaining the base SAM checkpoint and the
expected format of your own fine-tuned MedSAM weights.
"""

import argparse
import csv
import os
import time
from collections import defaultdict

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from skimage import io
from segment_anything import sam_model_registry


# =====================================================================
# 1. MedSAM model definition (with iterative mask-logit refinement)
# =====================================================================
class MedSAM(nn.Module):
    """Wraps a frozen SAM image encoder / prompt encoder with a
    (typically fine-tuned) mask decoder, and performs two refinement
    passes: the first pass conditions on the input mask prompt, the
    second re-conditions on the first pass's output logits."""

    def __init__(self, image_encoder, mask_decoder, prompt_encoder):
        super().__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder

        for param in self.image_encoder.parameters():
            param.requires_grad = False
        for param in self.prompt_encoder.parameters():
            param.requires_grad = False

    def forward(self, image, mask_logits, timing=None):
        """
        Args:
            image: (B, 3, H, W) float tensor.
            mask_logits: (B, 1, 256, 256) float tensor of mask-prompt
                logits, or None to run without a mask prompt.
            timing: optional dict[str, list[float]] to accumulate
                per-stage latency (ms). Only active on CUDA tensors.
        """

        def _sync_time():
            torch.cuda.synchronize()
            return time.perf_counter()

        use_timer = timing is not None and image.is_cuda

        # ---- Image encoder ----
        if use_timer:
            t0 = _sync_time()
        image_embedding = self.image_encoder(image)
        if use_timer:
            t1 = _sync_time()
            timing["image_encoder"].append((t1 - t0) * 1000)

        # ---- Refinement pass 1 ----
        if use_timer:
            t0 = _sync_time()
        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=None, boxes=None, masks=mask_logits,
        )
        low_res_masks, _ = self.mask_decoder(
            image_embeddings=image_embedding,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )
        if use_timer:
            t1 = _sync_time()
            timing["prompt_decoder_pass1"].append((t1 - t0) * 1000)

        # ---- Refinement pass 2 (re-condition on pass-1 output) ----
        if use_timer:
            t0 = _sync_time()
        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=None, boxes=None, masks=low_res_masks
        )
        refined_masks_logits, _ = self.mask_decoder(
            image_embeddings=image_embedding,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )
        if use_timer:
            t1 = _sync_time()
            timing["prompt_decoder_pass2"].append((t1 - t0) * 1000)

        # ---- Upsample to input resolution ----
        if use_timer:
            t0 = _sync_time()
        ori_res_masks = F.interpolate(
            refined_masks_logits, size=(image.shape[2], image.shape[3]),
            mode="bilinear", align_corners=False,
        )
        if use_timer:
            t1 = _sync_time()
            timing["postprocess"].append((t1 - t0) * 1000)
            timing["total"].append(
                timing["image_encoder"][-1] + timing["prompt_decoder_pass1"][-1] +
                timing["prompt_decoder_pass2"][-1] + timing["postprocess"][-1]
            )

        return ori_res_masks


# =====================================================================
# 2. Image tiling / stitching helpers
# =====================================================================
def pad_and_split_image(img, patch_size=1024):
    """Zero-pads an image (or mask) up to a multiple of `patch_size` and
    splits it into a row-major list of square patches."""
    H, W = img.shape[:2]
    pad_h = (patch_size - H % patch_size) % patch_size
    pad_w = (patch_size - W % patch_size) % patch_size

    if img.ndim == 3:
        padded_img = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant", constant_values=0)
    else:
        padded_img = np.pad(img, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=0)

    H_pad, W_pad = padded_img.shape[:2]
    img_patches = []

    for i in range(0, H_pad, patch_size):
        for j in range(0, W_pad, patch_size):
            if padded_img.ndim == 3:
                img_patch = padded_img[i:i + patch_size, j:j + patch_size, :]
            else:
                img_patch = padded_img[i:i + patch_size, j:j + patch_size]
            img_patches.append(img_patch)

    return img_patches, H, W


def stitch_patches(patches, original_H, original_W, patch_size=1024):
    """Reassembles row-major patches into a single image and crops back
    to the original (pre-padding) size."""
    rows = (original_H + patch_size - 1) // patch_size
    cols = (original_W + patch_size - 1) // patch_size

    stitched = np.zeros((rows * patch_size, cols * patch_size), dtype=np.uint8)

    for idx, patch in enumerate(patches):
        i = (idx // cols) * patch_size
        j = (idx % cols) * patch_size
        stitched[i:i + patch_size, j:j + patch_size] = patch

    return stitched[:original_H, :original_W]


def normalize_mask_to_binary(mask, out_value=255):
    """
    Normalizes a mask of unknown/inconsistent encoding into a clean
    binary image (0 / out_value). Handles:
      - already-binary 0/1
      - two-level grayscale (e.g. 0/255, 0/128)
      - noisy near-binary masks (anti-aliased edges, stray values)
      - float-valued masks (0-1 or 0-255 range)
    """
    mask = np.asarray(mask)
    mask_f = mask.astype(np.float32)
    unique_vals = np.unique(mask_f)

    if set(unique_vals.tolist()) <= {0.0, 1.0}:
        binary = (mask_f > 0).astype(np.uint8)
    elif len(unique_vals) <= 2:
        thresh = unique_vals.min()
        binary = (mask_f > thresh).astype(np.uint8)
    else:
        mask_uint8 = cv2.normalize(mask_f, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        _, binary_uint8 = cv2.threshold(mask_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        binary = (binary_uint8 > 0).astype(np.uint8)

    return (binary * out_value).astype(np.uint8)


def degrade_mask(mask, mode="gt", severity=15):
    """
    Simulates various qualities of input mask prompt, useful for
    evaluating robustness to annotation quality.

    mode:
      'gt'     - use the mask as-is (baseline)
      'coarse' - erode/dilate to simulate coarse or misaligned annotation
      'bbox'   - collapse to the bounding box only (discard fine boundary)
      'none'   - provide no mask prompt at all
    """
    if mode == "gt":
        return mask
    if mode == "none":
        return None
    if mode == "bbox":
        ys, xs = np.where(mask > 0)
        bbox_mask = np.zeros_like(mask)
        if len(xs) > 0:
            x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
            bbox_mask[y0:y1 + 1, x0:x1 + 1] = 255
        return bbox_mask
    if mode == "coarse":
        kernel = np.ones((severity, severity), np.uint8)
        op = cv2.erode if np.random.rand() < 0.5 else cv2.dilate
        return op(mask, kernel, iterations=1)
    raise ValueError(f"Unknown mask_mode: {mode}")


# =====================================================================
# 3. Core inference: process one image's patches
# =====================================================================
def process_and_save_patches(img_patches, mask_patches, medsam_model, device,
                              patch_size=1024, mask_mode="gt", severity=15,
                              timing=None, global_counter=None, global_warmup=5):
    """
    Runs inference on a list of aligned image/mask patches.

    timing: dict(list) accumulating per-stage latency in ms; pass None to disable.
    global_counter: dict like {"count": 0}, shared across images, used to
        skip the first `global_warmup` patches from timing statistics.
    global_warmup: number of initial patches (across the whole run) to
        exclude from timing stats (GPU warm-up).
    """
    results = []

    for img_patch, mask_patch in zip(img_patches, mask_patches):
        # --- Prepare image patch ---
        img_patch_resized = cv2.resize(
            img_patch.astype(np.float32), (patch_size, patch_size), interpolation=cv2.INTER_CUBIC
        )
        img_min, img_max = img_patch_resized.min(), img_patch_resized.max()
        if img_max > img_min:
            img_patch_resized = (img_patch_resized - img_min) / (img_max - img_min)

        img_tensor = torch.tensor(img_patch_resized).float().permute(2, 0, 1).unsqueeze(0).to(device)

        # --- Prepare mask patch / prompt ---
        mask = np.array(mask_patch)
        if mask.ndim == 3:
            mask = mask[:, :, 0]

        degraded_mask = degrade_mask(mask, mode=mask_mode, severity=severity)

        if degraded_mask is None:
            prompt_tensor = None
        else:
            mask_256 = np.array(Image.fromarray(degraded_mask).resize((256, 256), Image.NEAREST))
            mask_logits_np = np.ones((256, 256), dtype=np.float32) * (-10.0)
            mask_logits_np[mask_256 > 0] = 10.0
            prompt_tensor = torch.tensor(mask_logits_np).unsqueeze(0).unsqueeze(0).to(device)

        # --- Decide whether to record timing for this patch (skip warm-up) ---
        if global_counter is not None:
            do_timing = (timing is not None) and (global_counter["count"] >= global_warmup)
            global_counter["count"] += 1
        else:
            do_timing = timing is not None

        current_timing = timing if do_timing else None

        # --- Inference ---
        with torch.no_grad():
            medsam_seg_logits = medsam_model(img_tensor, prompt_tensor, timing=current_timing)
            medsam_seg_prob = torch.sigmoid(medsam_seg_logits)

        # --- Post-process ---
        medsam_seg = F.interpolate(
            medsam_seg_prob, size=(patch_size, patch_size),
            mode="bilinear", align_corners=False,
        )
        medsam_seg_np = medsam_seg.squeeze().cpu().numpy()
        pred_mask_uint8 = (medsam_seg_np > 0.5).astype(np.uint8) * 255
        results.append(pred_mask_uint8)

    return results


# =====================================================================
# 4. CLI / main
# =====================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Patch-based inference with iterative-refinement MedSAM."
    )
    parser.add_argument(
        "-i", "--data_path", type=str, required=True,
        help="Folder containing paired images (<name>.bmp) and labels (<name>_label.bmp)."
    )
    parser.add_argument(
        "-o", "--seg_path", type=str, required=True,
        help="Folder to save refined segmentation outputs and the timing CSV."
    )
    parser.add_argument(
        "-chk", "--checkpoint", type=str, required=True,
        help="Path to the fine-tuned MedSAM checkpoint (.pth)."
    )
    parser.add_argument(
        "--base_sam_checkpoint", type=str, required=True,
        help="Path to the base SAM ViT-B checkpoint (sam_vit_b_*.pth), "
             "required to build the model architecture."
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0",
        help="Device to run on, e.g. 'cuda:0' or 'cpu'."
    )
    parser.add_argument(
        "--mask_mode", type=str, default="gt",
        choices=["gt", "coarse", "bbox", "none"],
        help="Quality of the input mask prompt: gt=original, coarse=degraded, "
             "bbox=bounding box only, none=no mask prompt."
    )
    parser.add_argument(
        "--severity", type=int, default=15,
        help="Erosion/dilation kernel size for 'coarse' mask_mode (larger = more degraded)."
    )
    parser.add_argument(
        "--warmup_patches", type=int, default=5,
        help="Number of initial patches to exclude from timing statistics (GPU warm-up)."
    )

    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.seg_path, exist_ok=True)
    device = args.device

    print(">>> Loading model...")
    sam_model = sam_model_registry["vit_b"](checkpoint=args.base_sam_checkpoint)
    medsam_model = MedSAM(
        image_encoder=sam_model.image_encoder,
        mask_decoder=sam_model.mask_decoder,
        prompt_encoder=sam_model.prompt_encoder,
    ).to(device)

    medsam_ckpt = torch.load(args.checkpoint, map_location=device)
    if "model" in medsam_ckpt:
        medsam_model.load_state_dict(medsam_ckpt["model"])
    else:
        medsam_model.load_state_dict(medsam_ckpt)

    medsam_model.eval()
    print(">>> Model loaded.")

    timing_stats = defaultdict(list)
    global_patch_counter = {"count": 0}

    all_files = os.listdir(args.data_path)
    image_files = [
        f for f in all_files
        if f.lower().endswith(".bmp") and not f.lower().endswith("_label.bmp")
    ]
    print(f">>> Found {len(image_files)} image(s) to process.")

    for image_file in image_files:
        prefix = os.path.splitext(image_file)[0]
        label_file = f"{prefix}_label.bmp"
        img_path = os.path.join(args.data_path, image_file)
        label_path = os.path.join(args.data_path, label_file)

        if not os.path.exists(label_path):
            print(f"[WARN] Matching label not found: {label_file} - skipping.")
            continue

        print(f"Processing: {image_file} ...")

        img_np = io.imread(img_path)
        if len(img_np.shape) == 2:
            img_np = np.repeat(img_np[:, :, None], 3, axis=-1)

        mask_np = io.imread(label_path)
        if len(mask_np.shape) == 3:
            mask_np = mask_np[:, :, 0]

        if img_np.shape[:2] != mask_np.shape[:2]:
            print(f"[ERROR] Size mismatch! image {img_np.shape[:2]} vs label {mask_np.shape[:2]}. Skipping.")
            continue

        # Normalize the raw mask to a clean 0/255 binary image before any further processing.
        mask_np = normalize_mask_to_binary(mask_np, out_value=255)

        img_patches, original_H, original_W = pad_and_split_image(img_np)
        mask_patches, _, _ = pad_and_split_image(mask_np)

        segmentation_results = process_and_save_patches(
            img_patches, mask_patches, medsam_model, device,
            mask_mode=args.mask_mode, severity=args.severity,
            timing=timing_stats,
            global_counter=global_patch_counter,
            global_warmup=args.warmup_patches,
        )

        stitched_segmentation = stitch_patches(segmentation_results, original_H, original_W)

        refined_filename = f"{prefix}_refined.png"
        gt_filename = f"{prefix}_gt.png"

        refined_path = os.path.join(args.seg_path, refined_filename)
        gt_path = os.path.join(args.seg_path, gt_filename)

        Image.fromarray(stitched_segmentation).save(refined_path)
        Image.fromarray(mask_np).save(gt_path)

        print(f"   -> saved refined result: {refined_filename}")
        print(f"   -> saved normalized ground-truth: {gt_filename}")

    # ---- Timing summary ----
    print("\n" + "=" * 50)
    print(f">>> Inference timing summary (ms/patch, first {args.warmup_patches} warm-up patches excluded)")
    print(f">>> Total patches processed: {global_patch_counter['count']}, "
          f"used for stats: {max(0, global_patch_counter['count'] - args.warmup_patches)}")
    print("=" * 50)

    summary_rows = []
    for key in ["image_encoder", "prompt_decoder_pass1", "prompt_decoder_pass2", "postprocess", "total"]:
        vals = np.array(timing_stats[key])
        if len(vals) == 0:
            print(f"{key:25s}: no data (module not timed, or not enough patches)")
            continue
        mean_ms, std_ms = vals.mean(), vals.std()
        fps_str = f"   FPS={1000.0 / mean_ms:.2f}" if key == "total" else ""
        print(f"{key:25s}: {mean_ms:8.2f} +/- {std_ms:6.2f} ms  (n={len(vals)}){fps_str}")
        summary_rows.append([key, mean_ms, std_ms, len(vals)])

    csv_path = os.path.join(args.seg_path, "inference_timing.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["module", "mean_ms", "std_ms", "n_samples"])
        writer.writerows(summary_rows)
    print(f"\n>>> Timing summary saved to: {csv_path}")
    print(">>> Done.")


if __name__ == "__main__":
    main()