"""
train_colpali.py — Fine-tune ColPali-3B (PaliGemma backbone) with MatryoshkaMaxSimLoss.

No compression head — raw 2048-dim token embeddings, truncated at [256, 512, 1024, 2048].
Asymmetric encoding: queries use text tokens; documents use image-patch tokens.

Usage:
    python src/train/train_colpali.py --config configs/train_config_colpali.yaml
"""

import sys
from pathlib import Path

_THIS = Path(__file__).resolve().parent
_SRC  = _THIS.parent
_ROOT = _SRC.parent
sys.path = [p for p in sys.path if Path(p).resolve() != _THIS]
sys.path.insert(0, str(_SRC))
sys.path.insert(0, str(_ROOT))

import argparse
import io
import math

import bitsandbytes as bnb
import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image as PILImage
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from scripts.colpali_paligemma_embedding import ColPaliForEmbedding
from train.colpali_collator import ColPaliCollator
from train.dataset import ViDoReDataset
from train.loss import MatryoshkaMaxSimLoss
from train.teacher_attention import (
    TeacherAttentionCache,
    attention_alignment_loss,
    ensure_cache_prompt_mode,
    extract_query_patch_scores_from_similarity,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

class DotDict(dict):
    def __getattr__(self, key):
        try:
            val = self[key]
            return DotDict(val) if isinstance(val, dict) else val
        except KeyError:
            raise AttributeError(key)


def load_config(path: str) -> DotDict:
    with open(path) as f:
        return DotDict(yaml.safe_load(f))


def move_to_device(d: dict, device: torch.device) -> dict:
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in d.items()}


# ── Evaluation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_eval(model, collator, eval_df: pd.DataFrame, device: torch.device, dims, temperature: float) -> dict:
    model.eval()

    def _img_key(row):
        fn = row.get("image_filename")
        if fn and str(fn).strip():
            return str(fn)
        img = row["image"]
        if isinstance(img, dict) and img.get("path"):
            return img["path"]
        return str(row.name)

    filenames     = [_img_key(row) for _, row in eval_df.iterrows()]
    unique_fnames = list(dict.fromkeys(filenames))
    fname_to_idx  = {fn: i for i, fn in enumerate(unique_fnames)}

    fname_to_img = {}
    for (_, row), fn in zip(eval_df.iterrows(), filenames):
        if fn not in fname_to_img:
            img_data = row["image"]
            if isinstance(img_data, dict):
                fname_to_img[fn] = PILImage.open(io.BytesIO(img_data["bytes"])).convert("RGB")
            elif isinstance(img_data, bytes):
                fname_to_img[fn] = PILImage.open(io.BytesIO(img_data)).convert("RGB")

    unique_imgs = [fname_to_img[fn] for fn in unique_fnames]
    img_batch   = cfg_eval_img_batch_size if hasattr(run_eval, "_img_bs") else 4

    # Encode documents
    doc_embs, doc_masks_list = [], []
    for i in range(0, len(unique_imgs), img_batch):
        batch_imgs = unique_imgs[i: i + img_batch]
        fake_batch = [{"query": "", "image": img} for img in batch_imgs]
        inputs = collator(fake_batch)
        d_in   = move_to_device(inputs["doc_inputs"], device)
        d_mask = inputs["doc_token_mask"].to(device)
        out = model(**d_in)
        doc_embs.append(out.last_hidden_state)
        doc_masks_list.append(d_mask)

    doc_embs  = torch.cat(doc_embs,  dim=0)
    doc_masks = torch.cat(doc_masks_list, dim=0)

    # Encode queries
    queries = eval_df["query"].tolist()
    q_embs, q_masks_list = [], []
    q_batch = 16
    for i in range(0, len(queries), q_batch):
        batch_q = queries[i: i + q_batch]
        fake_batch = [{"query": q, "image": PILImage.new("RGB", (224, 224))} for q in batch_q]
        inputs = collator(fake_batch)
        q_in   = {k: v.to(device) for k, v in inputs["query_inputs"].items() if isinstance(v, torch.Tensor)}
        q_mask = inputs["query_token_mask"].to(device)
        out = model(**q_in)
        q_embs.append(out.last_hidden_state)
        q_masks_list.append(q_mask)

    q_embs  = torch.cat(q_embs,  dim=0)
    q_masks = torch.cat(q_masks_list, dim=0)

    # Score using the largest dim (full 2048)
    from train.loss import MaxSimLoss
    q_norm = torch.nn.functional.normalize(q_embs.float(), p=2, dim=-1) * q_masks.unsqueeze(-1).float()
    d_norm = torch.nn.functional.normalize(doc_embs.float(), p=2, dim=-1) * doc_masks.unsqueeze(-1).float()
    scores = MaxSimLoss._compute_scores(q_norm, d_norm).cpu()  # [Q, D]

    # nDCG@5
    gt_idxs = [fname_to_idx[fn] for fn in filenames]
    ks = [5]
    ndcg_vals = []
    for qi, gt in enumerate(gt_idxs):
        row_scores = scores[qi]
        topk = torch.argsort(row_scores, descending=True)[: ks[0]].tolist()
        dcg  = sum((1.0 / math.log2(rank + 2)) for rank, idx in enumerate(topk) if idx == gt)
        ndcg_vals.append(min(dcg, 1.0))  # IDCG = 1.0 (single relevant)

    return {"ndcg@5": float(np.mean(ndcg_vals))}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     default="configs/train_config_colpali.yaml")
    parser.add_argument("--checkpoint", default=None, help="LoRA checkpoint to resume from")
    args = parser.parse_args()

    cfg    = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device  : {device}")
    print(f"Model   : {cfg.model.name_or_path}")

    # ── Model ──────────────────────────────────────────────────────────────
    model = ColPaliForEmbedding.from_pretrained(
        cfg.model.name_or_path,
        torch_dtype=torch.bfloat16,
    ).to(device)

    # ── Gradient checkpointing (must be before PEFT wrap) ──────────────────
    if cfg.training.gradient_checkpointing:
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable()

    # ── LoRA ───────────────────────────────────────────────────────────────
    lora_cfg = LoraConfig(
        r=cfg.lora.r,
        lora_alpha=cfg.lora.lora_alpha,
        lora_dropout=cfg.lora.lora_dropout,
        bias=cfg.lora.bias,
        target_modules=list(cfg.lora.target_modules),
        task_type="FEATURE_EXTRACTION",
    )

    if args.checkpoint:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.checkpoint, is_trainable=True)
        print(f"Loaded LoRA from: {args.checkpoint}")
    else:
        model = get_peft_model(model, lora_cfg)

    model.print_trainable_parameters()

    # ── Collator + Dataset ─────────────────────────────────────────────────
    collator = ColPaliCollator(
        processor_path=cfg.model.name_or_path,
        max_query_len=cfg.training.get("max_query_len", 50),
    )

    dataset = ViDoReDataset(
        data_path=cfg.data.train_data_path,
        split="train",
        num_shards=cfg.data.num_shards,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.training.per_device_train_batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=cfg.training.dataloader_num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # ── Eval data ──────────────────────────────────────────────────────────
    eval_df = None
    try:
        eval_path = cfg.data.eval_data_path
        eval_file = next(Path(eval_path).glob(f"{cfg.data.eval_split}-*.parquet"))
        eval_df   = pd.read_parquet(eval_file)
        print(f"Eval data: {eval_file}  ({len(eval_df)} rows)")
    except Exception as e:
        print(f"[WARN] Eval data not loaded: {e}")

    teacher_attn_cache_prior = None
    teacher_attn_cache_query = None
    agree_lambda_prior = float(cfg.loss.get("agree_lambda_prior", cfg.loss.get("agree_lambda", 0.0)))
    agree_lambda_query = float(cfg.loss.get("agree_lambda_query", 0.0))
    agree_loss_type = str(cfg.loss.get("agree_loss_type", "kl"))
    agree_student_score_mode = str(cfg.loss.get("agree_student_score_mode", "softmax_sum"))

    attn_cache_path_prior = cfg.data.get("attn_cache_path_prior", cfg.data.get("attn_cache_path", None))
    attn_cache_path_query = cfg.data.get("attn_cache_path_query", None)
    if attn_cache_path_prior:
        teacher_attn_cache_prior = TeacherAttentionCache(attn_cache_path_prior)
        ensure_cache_prompt_mode(teacher_attn_cache_prior.metadata, "image_only")
        print(f"Teacher prior cache: {attn_cache_path_prior}")
    else:
        agree_lambda_prior = 0.0

    if attn_cache_path_query:
        teacher_attn_cache_query = TeacherAttentionCache(attn_cache_path_query)
        ensure_cache_prompt_mode(teacher_attn_cache_query.metadata, "query_image")
        print(f"Teacher query cache: {attn_cache_path_query}")
    else:
        agree_lambda_query = 0.0

    # ── Loss ───────────────────────────────────────────────────────────────
    dims        = list(cfg.loss.get("dims", [256, 512, 1024, 2048]))
    temperature = float(cfg.loss.get("temperature", 1.0))
    loss_fn     = MatryoshkaMaxSimLoss(dims=dims, temperature=temperature)
    print(f"Loss    : MatryoshkaMaxSimLoss  dims={dims}  temp={temperature}")
    if teacher_attn_cache_prior is not None:
        print(f"Align   : prior teacher ({agree_loss_type}, lambda={agree_lambda_prior})")
    if teacher_attn_cache_query is not None:
        print(f"Align   : query teacher ({agree_loss_type}, lambda={agree_lambda_query})")

    # ── Optimizer & Scheduler ──────────────────────────────────────────────
    trainable   = [p for p in model.parameters() if p.requires_grad]
    optimizer   = bnb.optim.PagedAdamW8bit(trainable, lr=cfg.training.learning_rate)
    grad_accum  = cfg.training.gradient_accumulation_steps
    steps_per_epoch = len(dataloader) // grad_accum
    total_steps     = steps_per_epoch * cfg.training.num_epochs
    warmup_steps    = math.ceil(total_steps * cfg.training.warmup_ratio)
    scheduler       = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    print(f"Samples : {len(dataset)}")
    print(f"Steps/epoch: {steps_per_epoch}  |  Total: {total_steps}  |  Warmup: {warmup_steps}")
    print(f"Eff. batch : {cfg.training.per_device_train_batch_size * grad_accum}")

    output_dir = Path(cfg.training.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Resume ─────────────────────────────────────────────────────────────
    import re
    start_step = 0
    if args.checkpoint:
        m = re.search(r"checkpoint-(\d+)", str(args.checkpoint))
        if m:
            start_step = int(m.group(1))
            for _ in range(start_step):
                scheduler.step()
            print(f"Resuming from global_step={start_step}")

    global_step = start_step
    optimizer.zero_grad()

    # ── Training loop ──────────────────────────────────────────────────────
    model.train()
    for epoch in range(cfg.training.num_epochs):
        accum_loss  = 0.0
        window_loss = 0.0
        progress = tqdm(total=steps_per_epoch,
                        desc=f"Epoch {epoch + 1}/{cfg.training.num_epochs}")

        for step, batch in enumerate(dataloader):
            query_inputs = move_to_device(batch["query_inputs"], device)
            doc_inputs   = move_to_device(batch["doc_inputs"],   device)
            q_mask       = batch["query_token_mask"].to(device)
            d_mask       = batch["doc_token_mask"].to(device)
            sample_ids   = batch.get("sample_ids", [])
            queries      = batch.get("queries", [])

            q_out = model(**query_inputs)
            d_out = model(**doc_inputs)

            q_emb = q_out.last_hidden_state   # [B, Nq, 2048]
            d_emb = d_out.last_hidden_state   # [B, Nd, 2048]

            loss = loss_fn(q_emb, d_emb, q_mask, d_mask)

            align_query_value = None
            align_prior_value = None
            matched_query = 0
            matched_prior = 0
            if (
                sample_ids
                and queries
                and (teacher_attn_cache_prior is not None or teacher_attn_cache_query is not None)
            ):
                student_scores = extract_query_patch_scores_from_similarity(
                    q_emb=q_emb,
                    d_emb=d_emb,
                    q_input_ids=query_inputs["input_ids"],
                    q_attention_mask=query_inputs["attention_mask"],
                    d_input_ids=doc_inputs["input_ids"],
                    d_attention_mask=doc_inputs["attention_mask"],
                    queries=queries,
                    tokenizer=collator.processor.tokenizer,
                    image_token_id=collator.image_token_id,
                    mode=agree_student_score_mode,
                )

                if teacher_attn_cache_query is not None and agree_lambda_query > 0:
                    teacher_query_scores = teacher_attn_cache_query.get_many(sample_ids)
                    align_query_value, matched_query = attention_alignment_loss(
                        student_scores=student_scores,
                        teacher_scores=teacher_query_scores,
                        loss_type=agree_loss_type,
                        teacher_grids=teacher_attn_cache_query.get_many_grids(sample_ids),
                    )
                    align_query_value = align_query_value.to(loss.device)
                    if matched_query > 0:
                        loss = loss + agree_lambda_query * align_query_value

                if teacher_attn_cache_prior is not None and agree_lambda_prior > 0:
                    teacher_prior_scores = teacher_attn_cache_prior.get_many(sample_ids)
                    align_prior_value, matched_prior = attention_alignment_loss(
                        student_scores=student_scores,
                        teacher_scores=teacher_prior_scores,
                        loss_type=agree_loss_type,
                        teacher_grids=teacher_attn_cache_prior.get_many_grids(sample_ids),
                    )
                    align_prior_value = align_prior_value.to(loss.device)
                    if matched_prior > 0:
                        loss = loss + agree_lambda_prior * align_prior_value

            (loss / grad_accum).backward()

            accum_loss  += loss.item()
            window_loss += loss.item()

            if (step + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable, cfg.training.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                progress.update(1)

                postfix = {"loss": f"{window_loss / grad_accum:.4f}", "lr": f"{scheduler.get_last_lr()[0]:.2e}"}
                if align_query_value is not None and matched_query > 0:
                    postfix["agree_q"] = f"{align_query_value.item():.4f}"
                if align_prior_value is not None and matched_prior > 0:
                    postfix["agree_p"] = f"{align_prior_value.item():.4f}"
                progress.set_postfix(**postfix)
                window_loss = 0.0

                if global_step % cfg.training.logging_steps == 0:
                    avg_loss = accum_loss / (cfg.training.logging_steps * grad_accum)
                    accum_loss = 0.0
                    extra = ""
                    if align_query_value is not None and matched_query > 0:
                        extra += f"  agree_q={align_query_value.item():.4f}  matched_q={matched_query}"
                    if align_prior_value is not None and matched_prior > 0:
                        extra += f"  agree_p={align_prior_value.item():.4f}  matched_p={matched_prior}"
                    print(f"\n[step {global_step:>6}] loss={avg_loss:.4f}  lr={scheduler.get_last_lr()[0]:.2e}{extra}")

                if global_step % cfg.training.save_steps == 0:
                    ckpt = output_dir / f"checkpoint-{global_step}"
                    model.save_pretrained(ckpt)
                    print(f"  → checkpoint saved: {ckpt}")

        progress.close()

        print(f"Epoch {epoch + 1} complete.")

        if eval_df is not None:
            metrics = run_eval(model, collator, eval_df, device, dims, temperature)
            print(f"  Eval  — nDCG@5: {metrics['ndcg@5']:.4f}")

    # ── Save final ─────────────────────────────────────────────────────────
    final_dir = output_dir / "final"
    model.save_pretrained(final_dir)
    print(f"\nFinal adapter saved → {final_dir}")


if __name__ == "__main__":
    main()
