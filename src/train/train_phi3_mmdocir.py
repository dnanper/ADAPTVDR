"""Train Phi3-Vision with thesis-style late-interaction hard-negative loss."""

import argparse
import sys
from pathlib import Path

_THIS = Path(__file__).resolve().parent
_SRC = _THIS.parent
_ROOT = _SRC.parent
sys.path = [p for p in sys.path if Path(p).resolve() != _THIS]
sys.path.insert(0, str(_SRC))
sys.path.insert(0, str(_ROOT))

import torch
import yaml
from peft import LoraConfig, PeftModel, get_peft_model
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

from scripts.colphi3_embedding import ColPhi3ForEmbedding
from train.dataset import TripletDataset
from train.loss import AugmentedMaxSimLoss, MatryoshkaAugmentedMaxSimLoss
from train.phi3_collator import Phi3MMDocIRCollator
from train.teacher_attention import (
    TeacherAttentionCache,
    attention_alignment_loss,
    ensure_cache_prompt_mode,
    extract_prior_patch_scores,
    extract_query_patch_scores_from_similarity,
)


class DotDict(dict):
    def __getattr__(self, key):
        try:
            value = self[key]
        except KeyError:
            raise AttributeError(key)
        return DotDict(value) if isinstance(value, dict) else value


def load_config(path: str) -> DotDict:
    with open(path, encoding="utf-8") as f:
        return DotDict(yaml.safe_load(f))


def move_to_device(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def make_optimizer(params, lr: float):
    try:
        import bitsandbytes as bnb

        return bnb.optim.PagedAdamW8bit(params, lr=lr)
    except Exception:
        return torch.optim.AdamW(params, lr=lr)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_config_phi3_mmdocir.yaml")
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if cfg.model.get("torch_dtype", "bfloat16") == "bfloat16" else torch.float16

    model = ColPhi3ForEmbedding(
        cfg.model.name_or_path,
        projection_dim=int(cfg.model.get("projection_dim", 128)),
        torch_dtype=dtype,
    ).to(device)

    lora_cfg = LoraConfig(
        r=cfg.lora.r,
        lora_alpha=cfg.lora.lora_alpha,
        lora_dropout=cfg.lora.lora_dropout,
        bias=cfg.lora.bias,
        target_modules=list(cfg.lora.target_modules),
        task_type="FEATURE_EXTRACTION",
        modules_to_save=["projection"],
    )
    model = PeftModel.from_pretrained(model, args.checkpoint, is_trainable=True) if args.checkpoint else get_peft_model(model, lora_cfg)
    if cfg.training.gradient_checkpointing:
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable()
    model.train()

    collator = Phi3MMDocIRCollator(
        model_path=cfg.model.name_or_path,
        max_length=cfg.data.max_length,
        image_size=cfg.data.get("image_size", 1344),
        min_pixels=int(cfg.data.get("min_pixels", 4096)),
        max_pixels=int(cfg.data.get("max_pixels", 1048576)),
        query_instruction=str(cfg.data.get("query_instruction", "Represent the user's input.")),
        doc_instruction=str(cfg.data.get("doc_instruction", "Represent the user's input.")),
    )
    prior_instruction_token_ids = collator.processor.tokenizer.encode(
        collator.doc_instruction,
        add_special_tokens=False,
    )
    dataset = TripletDataset(cfg.data.train_data_path, hard_neg_k=int(cfg.data.get("hard_neg_k", 1)))
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.training.per_device_train_batch_size,
        shuffle=True,
        num_workers=cfg.training.dataloader_num_workers,
        collate_fn=collator,
        pin_memory=True,
    )

    dims = list(cfg.loss.get("dims", [128]))
    temperature = float(cfg.loss.get("temperature", 1.0))
    if bool(cfg.loss.get("use_matryoshka", True)):
        loss_fn = MatryoshkaAugmentedMaxSimLoss(dims=dims, temperature=temperature)
    else:
        loss_fn = AugmentedMaxSimLoss(temperature=temperature)

    teacher_prior = None
    teacher_query = None
    agree_lambda_prior = float(cfg.loss.get("agree_lambda_prior", 0.0))
    agree_lambda_query = float(cfg.loss.get("agree_lambda_query", 0.0))
    if cfg.data.get("attn_cache_path_prior"):
        teacher_prior = TeacherAttentionCache(cfg.data.attn_cache_path_prior)
        ensure_cache_prompt_mode(teacher_prior.metadata, "image_only")
    else:
        agree_lambda_prior = 0.0
    if cfg.data.get("attn_cache_path_query"):
        teacher_query = TeacherAttentionCache(cfg.data.attn_cache_path_query)
        ensure_cache_prompt_mode(teacher_query.metadata, "query_image")
    else:
        agree_lambda_query = 0.0

    optimizer = make_optimizer((p for p in model.parameters() if p.requires_grad), cfg.training.learning_rate)
    grad_accum = int(cfg.training.gradient_accumulation_steps)
    total_steps = max(1, (len(dataloader) * cfg.training.num_epochs) // grad_accum)
    warmup_steps = int(total_steps * float(cfg.training.get("warmup_ratio", 0.0)))
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    output_dir = Path(cfg.training.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0
    optimizer.zero_grad()
    for epoch in range(cfg.training.num_epochs):
        progress = tqdm(dataloader, desc=f"epoch {epoch + 1}/{cfg.training.num_epochs}")
        for step, batch in enumerate(progress):
            query_inputs = move_to_device(batch["query_inputs"], device)
            doc_inputs = move_to_device(batch["doc_inputs"], device)
            pos_count = int(batch["pos_count"])
            sample_ids = list(batch.get("sample_ids", []))[:pos_count]
            queries = list(batch.get("queries", []))[:pos_count]

            need_attn = teacher_prior is not None and agree_lambda_prior > 0
            q_out = model(**query_inputs)
            d_out = model(**doc_inputs, output_attentions=need_attn)
            q_emb, d_emb = q_out.hidden_states, d_out.hidden_states
            q_mask = batch.get("query_token_mask", q_out.attention_mask.bool()).to(device)
            d_mask = batch.get("doc_token_mask", d_out.attention_mask.bool()).to(device)
            loss = loss_fn(q_emb, d_emb, q_mask, d_mask, pos_count=pos_count)

            if sample_ids and any(s is not None for s in sample_ids):
                # Phi3 processor does not expose Qwen-style image_grid_thw; leave grid
                # resizing off until a real Phi patch-grid mapper is added.
                student_grids = None
                if teacher_query is not None and agree_lambda_query > 0:
                    student_scores = extract_query_patch_scores_from_similarity(
                        q_emb=q_emb[:pos_count],
                        d_emb=d_emb[:pos_count],
                        q_input_ids=query_inputs["input_ids"][:pos_count],
                        q_attention_mask=query_inputs["attention_mask"][:pos_count],
                        d_input_ids=doc_inputs["input_ids"][:pos_count],
                        d_attention_mask=doc_inputs["attention_mask"][:pos_count],
                        queries=queries,
                        tokenizer=collator.processor.tokenizer,
                        image_token_id=collator.image_token_id,
                        mode=str(cfg.loss.get("agree_student_score_mode", "softmax_sum")),
                    )
                    align, matched = attention_alignment_loss(
                        student_scores,
                        teacher_query.get_many(sample_ids),
                        loss_type=str(cfg.loss.get("agree_loss_type", "kl")),
                        student_grids=student_grids,
                        teacher_grids=teacher_query.get_many_grids(sample_ids),
                    )
                    if matched:
                        loss = loss + agree_lambda_query * align.to(loss.device)

                if teacher_prior is not None and agree_lambda_prior > 0:
                    student_scores = extract_prior_patch_scores(
                        attentions=d_out.attentions,
                        input_ids=doc_inputs["input_ids"][:pos_count],
                        attention_mask=doc_inputs["attention_mask"][:pos_count],
                        image_token_id=collator.image_token_id,
                        source_mode=teacher_prior.source_mode,
                        instruction_token_ids=prior_instruction_token_ids,
                    )
                    align, matched = attention_alignment_loss(
                        student_scores,
                        teacher_prior.get_many(sample_ids),
                        loss_type=str(cfg.loss.get("agree_loss_type", "kl")),
                        student_grids=student_grids,
                        teacher_grids=teacher_prior.get_many_grids(sample_ids),
                    )
                    if matched:
                        loss = loss + agree_lambda_prior * align.to(loss.device)

            (loss / grad_accum).backward()
            if (step + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                progress.set_postfix(loss=f"{loss.item():.4f}", lr=f"{scheduler.get_last_lr()[0]:.1e}")
                if global_step % cfg.training.save_steps == 0:
                    model.save_pretrained(output_dir / f"checkpoint-{global_step}")

    model.save_pretrained(output_dir / "final")
    print(f"saved: {output_dir / 'final'}")


if __name__ == "__main__":
    main()
