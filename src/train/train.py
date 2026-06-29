"""
train.py — ColQwen3.5 LoRA training for ColPali-style document retrieval.

Usage (run from project root):
    python src/train/train.py --config configs/train_config.yaml
"""

import sys
from pathlib import Path

# ── sys.path fix ──────────────────────────────────────────────────────────────
# When Python runs `python src/train/train.py`, it prepends `src/train/` to
# sys.path automatically.  This makes `from train.X import Y` resolve to
# `train.py` (this file) instead of the `train/` package → self-import loop.
# Fix: strip the script dir and add `src/` so packages resolve correctly.
_THIS = Path(__file__).resolve().parent   # graduation_thesis/src/train/
_SRC  = _THIS.parent                      # graduation_thesis/src/
_ROOT = _SRC.parent                       # graduation_thesis/  (for scripts.*)
sys.path = [p for p in sys.path if Path(p).resolve() != _THIS]
sys.path.insert(0, str(_SRC))
sys.path.insert(0, str(_ROOT))
# ─────────────────────────────────────────────────────────────────────────────

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
from transformers import get_cosine_schedule_with_warmup

from scripts.colqwen3_5_embedding import ColQwen3_5ForEmbedding, ColQwen3_5Embedder
from train.collator import ColPaliCollator
from train.dataset import ViDoReDataset, LlamaIndexDataset, TripletDataset
from train.loss import InfoNCELoss, MaxSimLoss, MatryoshkaMaxSimLoss, AugmentedMaxSimLoss
from train.sampler import CollisionAwareSampler
from train.teacher_attention import (
    TeacherAttentionCache,
    attention_alignment_loss,
    ensure_cache_prompt_mode,
    extract_prior_patch_scores,
    extract_query_patch_scores_from_similarity,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

class DotDict(dict):
    """Nested dict with attribute-style access."""
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
def run_eval(model, collator, eval_df: pd.DataFrame, device: torch.device, cfg, mode: str = "multivec_mrl", proj_dim=None) -> dict:
    """Evaluate on the test parquet. Returns {ndcg@5, recall@1, recall@5}.

    mode: "dense" | "multivec_proj" | "multivec_mrl"
    """
    model.eval()
    proc = collator.processor

    # ── Build corpus of unique document images ─────────────────────────────
    # Use image_filename column as unique key (always non-null in test parquet)
    def _img_key(row):
        fn = row.get("image_filename")
        if fn and str(fn).strip():
            return str(fn)
        img = row["image"]
        if isinstance(img, dict) and img.get("path"):
            return img["path"]
        return str(row.name)  # last-resort fallback

    filenames     = [_img_key(row) for _, row in eval_df.iterrows()]
    unique_fnames = list(dict.fromkeys(filenames))
    fname_to_idx  = {fn: i for i, fn in enumerate(unique_fnames)}

    fname_to_img: dict = {}
    for (_, row), fn in zip(eval_df.iterrows(), filenames):
        if fn not in fname_to_img:
            img_data = row["image"]
            if isinstance(img_data, dict):
                fname_to_img[fn] = PILImage.open(io.BytesIO(img_data["bytes"])).convert("RGB")
            elif isinstance(img_data, bytes):
                fname_to_img[fn] = PILImage.open(io.BytesIO(img_data)).convert("RGB")
            else:
                fname_to_img[fn] = img_data.convert("RGB")
    unique_images = [fname_to_img[fn] for fn in unique_fnames]

    # ── Build query list with ground-truth corpus indices ──────────────────
    valid_queries: list  = []
    valid_rel_idxs: list = []
    for (_, row), fn in zip(eval_df.iterrows(), filenames):
        q = str(row.get("query", "")).strip()
        if q.lower() == "none" or len(q) < 5:
            continue
        valid_queries.append(q)
        valid_rel_idxs.append(fname_to_idx[fn])

    print(f"  [eval] corpus={len(unique_images)} docs  queries={len(valid_queries)}")

    # ── Encode documents ───────────────────────────────────────────────────
    img_batch_size = cfg.data.eval_img_batch
    doc_embs: list = []

    for i in tqdm(range(0, len(unique_images), img_batch_size), desc="  eval-docs", leave=False):
        imgs    = unique_images[i : i + img_batch_size]
        d_convs = [collator._make_doc_conv(img) for img in imgs]
        d_texts = proc.apply_chat_template(d_convs, add_generation_prompt=True, tokenize=False)

        doc_inputs = move_to_device(dict(proc(
            text=d_texts,
            images=imgs,
            padding=True,
            truncation=True,
            max_length=cfg.data.max_length,
            return_tensors="pt",
        )), device)

        out  = model(**doc_inputs)
        mask = doc_inputs["attention_mask"].bool()
        h    = out.hidden_states          # keep in model dtype (bf16)

        if mode == "dense":
            pooled = ColQwen3_5Embedder._pooling_last(h, mask)
            emb = torch.nn.functional.normalize(model.linear_head(pooled).float(), p=2, dim=-1)
        elif mode == "multivec_proj":
            emb = torch.nn.functional.normalize(model.linear_head(h).float(), p=2, dim=-1)
            emb = emb * mask.unsqueeze(-1).float()
        else:  # multivec_mrl
            emb = torch.nn.functional.normalize(h.float(), p=2, dim=-1)
            emb = emb * mask.unsqueeze(-1).float()
        doc_embs.append(emb)  # keep on GPU

    # ── Encode queries ─────────────────────────────────────────────────────
    q_batch_size = cfg.data.eval_q_batch
    qry_embs: list = []

    for i in tqdm(range(0, len(valid_queries), q_batch_size), desc="  eval-qrys", leave=False):
        q_batch  = valid_queries[i : i + q_batch_size]
        q_convs  = [collator._make_query_conv(q) for q in q_batch]
        q_texts  = proc.apply_chat_template(q_convs, add_generation_prompt=True, tokenize=False)
        q_inputs = move_to_device(dict(proc(
            text=q_texts, padding=True, truncation=True,
            max_length=cfg.data.max_length, return_tensors="pt",
        )), device)

        out  = model(**q_inputs)
        mask = q_inputs["attention_mask"].bool()
        h    = out.hidden_states          # keep in model dtype (bf16)

        if mode == "dense":
            pooled = ColQwen3_5Embedder._pooling_last(h, mask)
            emb = torch.nn.functional.normalize(model.linear_head(pooled).float(), p=2, dim=-1)
        elif mode == "multivec_proj":
            emb = torch.nn.functional.normalize(model.linear_head(h).float(), p=2, dim=-1)
            emb = emb * mask.unsqueeze(-1).float()
        else:  # multivec_mrl
            emb = torch.nn.functional.normalize(h.float(), p=2, dim=-1)
            emb = emb * mask.unsqueeze(-1).float()
        qry_embs.append(emb)  # keep on GPU

    # ── Score matrix [N_q, N_d] ───────────────────────────────────────────
    n_q    = len(valid_queries)
    n_d    = len(unique_images)
    scores = np.zeros((n_q, n_d), dtype=np.float32)

    if mode == "dense":
        # Dense: stack all embeddings → dot product
        q_all = torch.cat(qry_embs, dim=0)  # [N_q, proj_dim]
        d_all = torch.cat(doc_embs, dim=0)  # [N_d, proj_dim]
        scores = (q_all @ d_all.T).cpu().numpy()
    else:
        # Multi-vector: MaxSim
        q_offset = 0
        for q_block in tqdm(qry_embs, desc="  eval-maxsim", leave=False):
            bq       = q_block.shape[0]
            d_offset = 0
            for d_block in doc_embs:
                bd  = d_block.shape[0]
                sim = torch.einsum("bqd,cnd->bcqn", q_block, d_block)
                ms  = sim.max(dim=-1).values.sum(dim=-1)
                scores[q_offset : q_offset + bq, d_offset : d_offset + bd] = ms.detach().cpu().numpy()
                d_offset += bd
            q_offset += bq

    # ── Metrics ────────────────────────────────────────────────────────────
    ndcg5, r1, r5 = [], [], []
    for q_idx, rel_idx in enumerate(valid_rel_idxs):
        s    = scores[q_idx]
        top5 = np.argsort(s)[::-1][:5]
        rank = np.where(top5 == rel_idx)[0]
        ndcg5.append((1.0 / np.log2(int(rank[0]) + 2)) / (1.0 / np.log2(2)) if len(rank) > 0 else 0.0)
        r1.append(1.0 if rel_idx == np.argsort(s)[::-1][0] else 0.0)
        r5.append(1.0 if rel_idx in top5 else 0.0)

    result = {
        "ndcg@5":   round(float(np.mean(ndcg5)), 4),
        "recall@1": round(float(np.mean(r1)),    4),
        "recall@5": round(float(np.mean(r5)),    4),
    }
    print(f"  [eval] nDCG@5={result['ndcg@5']:.4f}  Recall@1={result['recall@1']:.4f}  Recall@5={result['recall@5']:.4f}")

    model.train()
    return result


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_config.yaml")
    parser.add_argument("--checkpoint", default=None, help="Path to LoRA checkpoint to resume training or for evaluation only")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device  : {device}")
    print(f"Model   : {cfg.model.name_or_path}")

    # ── Model ─────────────────────────────────────────────────────────────
    mode     = cfg.model.get("mode", "multivec_mrl")
    proj_dim = cfg.model.get("projection_dim", None) if mode != "multivec_mrl" else None

    if mode not in ("dense", "multivec_proj", "multivec_mrl"):
        raise ValueError(f"model.mode must be 'dense' | 'multivec_proj' | 'multivec_mrl', got {mode!r}")
    if mode in ("dense", "multivec_proj") and proj_dim is None:
        raise ValueError(f"model.mode={mode!r} requires model.projection_dim in config")

    print(f"Mode    : {mode}  (proj_dim={proj_dim})")

    if proj_dim:
        from transformers import AutoConfig
        base_cfg = AutoConfig.from_pretrained(cfg.model.name_or_path)
        base_cfg.embedding_dim = proj_dim
        model = ColQwen3_5ForEmbedding.from_pretrained(
            cfg.model.name_or_path,
            config=base_cfg,
            torch_dtype=torch.bfloat16,
        ).to(device)
        print(f"  Projection head: hidden_size → {proj_dim}")
    else:
        model = ColQwen3_5ForEmbedding.from_pretrained(
            cfg.model.name_or_path,
            torch_dtype=torch.bfloat16,
        ).to(device)

    # ── LoRA ──────────────────────────────────────────────────────────────
    lora_cfg = LoraConfig(
        r=cfg.lora.r,
        lora_alpha=cfg.lora.lora_alpha,
        lora_dropout=cfg.lora.lora_dropout,
        bias=cfg.lora.bias,
        target_modules=list(cfg.lora.target_modules),
        task_type="FEATURE_EXTRACTION",
        modules_to_save=["linear_head"] if mode in ("dense", "multivec_proj") else None,
    )

    if args.checkpoint:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.checkpoint, is_trainable=True)
        print(f"Applying LoRA adapters (r={cfg.lora.r}, alpha={cfg.lora.lora_alpha}) ...")
        print(f"  Loaded LoRA weights from: {args.checkpoint}")
    else:
        print(f"Applying LoRA adapters (r={cfg.lora.r}, alpha={cfg.lora.lora_alpha}) ...")
        model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # gradient_checkpointing + LoRA requires this
    if cfg.training.gradient_checkpointing:
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable()

    print(f"\n[OK] Model ready. Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    model.train()

    teacher_attn_cache_prior = None
    teacher_attn_cache_query = None
    agree_lambda_prior = float(cfg.loss.get("agree_lambda_prior", cfg.loss.get("agree_lambda", 0.0)))
    agree_lambda_query = float(cfg.loss.get("agree_lambda_query", 0.0))
    agree_loss_type = str(cfg.loss.get("agree_loss_type", "kl"))
    agree_student_score_mode = str(cfg.loss.get("agree_student_score_mode", "softmax_sum"))
    doc_instruction = str(cfg.data.get("doc_instruction", cfg.data.get("default_instruction", "Represent the user's input.")))
    query_instruction = str(cfg.data.get("query_instruction", cfg.data.get("default_instruction", "Represent the user's input.")))
    use_query_system_instruction = bool(cfg.data.get("use_query_system_instruction", True))
    use_doc_system_instruction = bool(cfg.data.get("use_doc_system_instruction", True))
    attn_cache_path_prior = cfg.data.get("attn_cache_path_prior", cfg.data.get("attn_cache_path", None))
    attn_cache_path_query = cfg.data.get("attn_cache_path_query", None)

    if attn_cache_path_prior:
        teacher_attn_cache_prior = TeacherAttentionCache(attn_cache_path_prior)
        ensure_cache_prompt_mode(teacher_attn_cache_prior.metadata, "image_only")
        doc_instruction = str(teacher_attn_cache_prior.metadata.get("instruction", doc_instruction))
        print(
            f"Teacher prior cache: {attn_cache_path_prior}  "
            f"(source_mode={teacher_attn_cache_prior.source_mode}, instruction={doc_instruction!r})"
        )
    else:
        agree_lambda_prior = 0.0

    if attn_cache_path_query:
        teacher_attn_cache_query = TeacherAttentionCache(attn_cache_path_query)
        ensure_cache_prompt_mode(teacher_attn_cache_query.metadata, "query_image")
        query_instruction = str(teacher_attn_cache_query.metadata.get("instruction", query_instruction))
        print(
            f"Teacher query cache: {attn_cache_path_query}  "
            f"(source_mode={teacher_attn_cache_query.source_mode}, instruction={query_instruction!r})"
        )
    else:
        agree_lambda_query = 0.0

    # ── Collator ───────────────────────────────────────────────────────────
    collator = ColPaliCollator(
        model_path=cfg.model.name_or_path,
        max_length=cfg.data.max_length,
        min_pixels=cfg.data.min_pixels,
        max_pixels=cfg.data.max_pixels,
        query_instruction=query_instruction,
        doc_instruction=doc_instruction,
        use_query_system_instruction=use_query_system_instruction,
        use_doc_system_instruction=use_doc_system_instruction,
    )
    prior_instruction_token_ids = None
    if teacher_attn_cache_prior is not None and teacher_attn_cache_prior.source_mode == "instruction":
        prior_instruction_token_ids = collator.processor.tokenizer.encode(
            doc_instruction,
            add_special_tokens=False,
        )

    # ── Dataset & DataLoader ───────────────────────────────────────────────
    dataset_type = cfg.data.get("dataset_type", "vidore")

    if dataset_type == "llamaindex":
        languages    = list(cfg.data.get("languages", ["en"]))
        hard_neg_k   = int(cfg.data.get("hard_neg_k", 3))
        preload_all  = bool(cfg.data.get("preload_all", False))
        max_shards   = int(cfg.data.get("max_shards", 12))
        dataset = LlamaIndexDataset(
            data_root=cfg.data.train_data_path,
            languages=languages,
            hard_neg_k=hard_neg_k,
            preload_all=preload_all,
            max_shards=max_shards,
        )
        # Lazy cache is not fork-safe → force num_workers=0
        num_workers = cfg.training.dataloader_num_workers if preload_all else 0
        if not preload_all and cfg.training.dataloader_num_workers > 0:
            print(f"[WARN] preload_all=False → forcing num_workers=0 (ShardLRUCache is not fork-safe)")
        sampler = CollisionAwareSampler(
            dataset,
            batch_size=cfg.training.per_device_train_batch_size,
            buffer_size=int(cfg.data.get("collision_buffer_size", 1000)),
            retry_limit=int(cfg.data.get("collision_retry_limit", 50)),
            drop_last=True,
        )
        dataloader = DataLoader(
            dataset,
            batch_sampler=sampler,
            collate_fn=collator,
            num_workers=num_workers,
            pin_memory=True,
        )
        print(f"Dataset : LlamaIndex {languages}  hard_neg_k={hard_neg_k}  "
              f"preload_all={preload_all}  num_workers={num_workers}  samples={len(dataset)}")
    elif dataset_type == "triplet":
        hard_neg_k  = int(cfg.data.get("hard_neg_k", 3))
        num_workers = cfg.training.dataloader_num_workers
        dataset = TripletDataset(
            data_path=cfg.data.train_data_path,
            hard_neg_k=hard_neg_k,
        )
        # Polars collect() → full dataset in RAM → fork-safe → any num_workers
        dataloader = DataLoader(
            dataset,
            batch_size=cfg.training.per_device_train_batch_size,
            shuffle=True,
            collate_fn=collator,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        )
        print(f"Dataset : Triplet  hard_neg_k={hard_neg_k}  "
              f"num_workers={num_workers}  samples={len(dataset)}")
    else:
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
        print(f"Dataset : ViDoRe  samples={len(dataset)}")

    # ── Eval dataset (test parquet) ────────────────────────────────────────
    eval_parquet_path = Path(cfg.data.eval_data_path) / f"{cfg.data.eval_split}-00000-of-00001.parquet"
    if eval_parquet_path.exists():
        eval_df = pd.read_parquet(eval_parquet_path)
        print(f"Eval data : {eval_parquet_path}  ({len(eval_df)} rows)")
    else:
        eval_df = None
        print(f"[WARN] Eval parquet not found: {eval_parquet_path} — skipping eval")

    # ── Loss — determined by mode and dataset_type ────────────────────────
    temperature    = float(cfg.loss.temperature)
    use_hard_negs  = dataset_type in ("llamaindex", "triplet")   # both supply hard_neg_images

    if mode == "dense":
        loss_fn = InfoNCELoss(temperature=temperature)
        print(f"Loss    : InfoNCELoss (temp={temperature})")
    elif mode == "multivec_proj":
        if use_hard_negs:
            loss_fn = AugmentedMaxSimLoss(temperature=temperature)
            print(f"Loss    : AugmentedMaxSimLoss (temp={temperature})")
        else:
            loss_fn = MaxSimLoss(temperature=temperature)
            print(f"Loss    : MaxSimLoss (temp={temperature})")
    elif mode == "multivec_mrl":
        dims    = list(cfg.loss.get("dims", [128, 256, 512, 1024]))
        weights = cfg.loss.get("weights", None)
        weights = list(weights) if weights is not None else None
        if use_hard_negs:
            # loss_fn = MatryoshkaAugmentedMaxSimLoss(dims=dims, temperature=temperature, weights=weights)
            loss_fn = AugmentedMaxSimLoss(temperature=temperature)
            print(f"Loss    : AugmentedMaxSimLoss (temp={temperature})")
            # print(f"Loss    : MatryoshkaAugmentedMaxSimLoss  dims={dims}  (temp={temperature})")
        else:
            loss_fn = MatryoshkaMaxSimLoss(dims=dims, temperature=temperature, weights=weights)
            print(f"Loss    : MatryoshkaMaxSimLoss  dims={dims}")
    else:
        raise ValueError(f"Unknown mode: {mode!r}")

    if teacher_attn_cache_prior is not None:
        if agree_lambda_prior > 0:
            print(f"Align   : prior teacher ({agree_loss_type}, lambda={agree_lambda_prior})")
        else:
            print("[WARN] prior teacher cache provided but agree_lambda_prior <= 0, so prior alignment is disabled")
    if teacher_attn_cache_query is not None:
        if agree_lambda_query > 0:
            print(f"Align   : query teacher ({agree_loss_type}, lambda={agree_lambda_query})")
        else:
            print("[WARN] query teacher cache provided but agree_lambda_query <= 0, so query alignment is disabled")

    # ── Optimizer & Scheduler ─────────────────────────────────────────────
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = bnb.optim.PagedAdamW8bit(trainable, lr=cfg.training.learning_rate)

    grad_accum = cfg.training.gradient_accumulation_steps
    steps_per_epoch = len(dataloader) // grad_accum
    total_steps = steps_per_epoch * cfg.training.num_epochs
    warmup_steps = math.ceil(total_steps * cfg.training.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    print(f"Samples : {len(dataset)}")
    print(f"Steps/epoch: {steps_per_epoch}  |  Total: {total_steps}  |  Warmup: {warmup_steps}")
    print(f"Eff. batch : {cfg.training.per_device_train_batch_size * grad_accum}")

    # ── Output dir ────────────────────────────────────────────────────────
    output_dir = Path(cfg.training.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Training loop ─────────────────────────────────────────────────────
    # Restore global_step from checkpoint path (e.g. "checkpoint-3000" → 3000)
    start_step = 0
    if args.checkpoint:
        import re
        m = re.search(r"checkpoint-(\d+)", str(args.checkpoint))
        if m:
            start_step = int(m.group(1))
            print(f"  Fast-forwarding scheduler to step {start_step} ...")
            for _ in range(start_step):
                scheduler.step()
            print(f"  Resuming from global_step={start_step}")

    global_step = start_step
    optimizer.zero_grad()

    for epoch in range(cfg.training.num_epochs):
        accum_loss = 0.0
        progress = tqdm(enumerate(dataloader), total=len(dataloader),
                        desc=f"Epoch {epoch + 1}/{cfg.training.num_epochs}")

        for step, batch in progress:
            query_inputs = move_to_device(batch["query_inputs"], device)
            doc_inputs   = move_to_device(batch["doc_inputs"],   device)
            pos_count    = batch.get("pos_count", cfg.training.per_device_train_batch_size)
            sample_ids   = batch.get("sample_ids", [])
            queries      = batch.get("queries", [])

            need_prior_attn = teacher_attn_cache_prior is not None and agree_lambda_prior > 0
            q_out = model(**query_inputs)
            d_out = model(**doc_inputs, output_attentions=need_prior_attn)

            q_emb  = q_out.hidden_states           # [B, Nq, D]
            d_emb  = d_out.hidden_states           # [B + n_neg, Nd, D]
            q_mask = q_out.attention_mask.bool()   # [B, Nq]
            d_mask = d_out.attention_mask.bool()   # [B + n_neg, Nd]

            if mode == "dense":
                q_vec = model.linear_head(ColQwen3_5Embedder._pooling_last(q_emb, q_mask))
                d_vec = model.linear_head(ColQwen3_5Embedder._pooling_last(d_emb, d_mask))
                loss = loss_fn(q_vec, d_vec)
            elif mode == "multivec_proj":
                q_proj = model.linear_head(q_emb)
                d_proj = model.linear_head(d_emb)
                if use_hard_negs:
                    loss = loss_fn(q_proj, d_proj, q_mask, d_mask, pos_count=pos_count)
                else:
                    loss = loss_fn(q_proj, d_proj, q_mask, d_mask)
            else:  # multivec_mrl
                if use_hard_negs:
                    loss = loss_fn(q_emb, d_emb, q_mask, d_mask, pos_count=pos_count)
                else:
                    loss = loss_fn(q_emb, d_emb, q_mask, d_mask)

            align_prior_value = None
            align_query_value = None
            matched_prior = 0
            matched_query = 0
            positive_sample_ids = list(sample_ids[:pos_count])
            if (
                positive_sample_ids
                and any(sample_id is not None for sample_id in positive_sample_ids)
                and queries
                and len(queries) >= pos_count
                and (teacher_attn_cache_prior is not None or teacher_attn_cache_query is not None)
            ):
                student_grids = None
                if "image_grid_thw" in doc_inputs:
                    student_grids = list(doc_inputs["image_grid_thw"][:pos_count])

                if teacher_attn_cache_query is not None and agree_lambda_query > 0:
                    student_query_scores = extract_query_patch_scores_from_similarity(
                        q_emb=q_emb[:pos_count],
                        d_emb=d_emb[:pos_count],
                        q_input_ids=query_inputs["input_ids"][:pos_count],
                        q_attention_mask=query_inputs["attention_mask"][:pos_count],
                        d_input_ids=doc_inputs["input_ids"][:pos_count],
                        d_attention_mask=doc_inputs["attention_mask"][:pos_count],
                        queries=queries[:pos_count],
                        tokenizer=collator.processor.tokenizer,
                        image_token_id=collator.image_token_id,
                        mode=agree_student_score_mode,
                    )
                    teacher_query_scores = teacher_attn_cache_query.get_many(positive_sample_ids)
                    align_query_value, matched_query = attention_alignment_loss(
                        student_scores=student_query_scores,
                        teacher_scores=teacher_query_scores,
                        loss_type=agree_loss_type,
                        student_grids=student_grids,
                        teacher_grids=teacher_attn_cache_query.get_many_grids(positive_sample_ids),
                    )
                    align_query_value = align_query_value.to(loss.device)
                    if matched_query > 0:
                        loss = loss + agree_lambda_query * align_query_value

                if teacher_attn_cache_prior is not None and agree_lambda_prior > 0:
                    student_prior_scores = extract_prior_patch_scores(
                        attentions=d_out.attentions,
                        input_ids=doc_inputs["input_ids"][:pos_count],
                        attention_mask=doc_inputs["attention_mask"][:pos_count],
                        image_token_id=collator.image_token_id,
                        source_mode=teacher_attn_cache_prior.source_mode,
                        instruction_token_ids=prior_instruction_token_ids,
                    )
                    teacher_prior_scores = teacher_attn_cache_prior.get_many(positive_sample_ids)
                    align_prior_value, matched_prior = attention_alignment_loss(
                        student_scores=student_prior_scores,
                        teacher_scores=teacher_prior_scores,
                        loss_type=agree_loss_type,
                        student_grids=student_grids,
                        teacher_grids=teacher_attn_cache_prior.get_many_grids(positive_sample_ids),
                    )
                    align_prior_value = align_prior_value.to(loss.device)
                    if matched_prior > 0:
                        loss = loss + agree_lambda_prior * align_prior_value

            (loss / grad_accum).backward()
            accum_loss += loss.item()

            # Optimizer step every grad_accum micro-batches
            if (step + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                avg_loss = accum_loss / grad_accum
                accum_loss = 0.0
                postfix = {"loss": f"{avg_loss:.4f}", "lr": f"{scheduler.get_last_lr()[0]:.1e}"}
                if align_query_value is not None and matched_query > 0:
                    postfix["agree_q"] = f"{align_query_value.item():.4f}"
                    postfix["mq"] = matched_query
                if align_prior_value is not None and matched_prior > 0:
                    postfix["agree_p"] = f"{align_prior_value.item():.4f}"
                    postfix["mp"] = matched_prior
                progress.set_postfix(**postfix)

                if global_step % cfg.training.logging_steps == 0:
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

        print(f"Epoch {epoch + 1} complete.")
        # if eval_df is not None:
        #     print(f"Running eval after epoch {epoch + 1} ...")
        #     run_eval(model, collator, eval_df, device, cfg,
        #              mode=mode, proj_dim=proj_dim)

    # ── Save final adapters ───────────────────────────────────────────────
    final = output_dir / "final"
    model.save_pretrained(final)
    print(f"\nTraining done. LoRA adapters → {final}")


if __name__ == "__main__":
    main()
