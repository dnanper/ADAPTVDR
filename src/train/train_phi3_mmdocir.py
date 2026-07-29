"""Train Phi3-Vision with thesis-style late-interaction hard-negative loss."""

import argparse
import csv
import json
import sys
import time
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

from scripts.colphi3_embedding import ColPhi3ForEmbedding, ensure_phi3_img_projection_bias
from train.batch_utils import clone_tensor_inputs, forward_model_in_chunks, move_to_device, slice_batch
from train.dataset import TripletDataset
from train.loss import AugmentedMaxSimLoss, MatryoshkaAugmentedMaxSimLoss
from train.phi3_collator import Phi3MMDocIRCollator
from train.schedule_utils import make_scheduler
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


def make_optimizer(params, lr: float):
    try:
        import bitsandbytes as bnb

        return bnb.optim.PagedAdamW8bit(params, lr=lr)
    except Exception:
        return torch.optim.AdamW(params, lr=lr)


def require_finite(name: str, value: torch.Tensor) -> None:
    if not torch.isfinite(value).all():
        raise FloatingPointError(f"{name} contains non-finite values")


class TrainMetricLogger:
    def __init__(self, output_dir: Path):
        self.csv_path = output_dir / "train_metrics.csv"
        self.jsonl_path = output_dir / "train_metrics.jsonl"
        self.fields = [
            "time",
            "epoch",
            "global_step",
            "micro_step",
            "lr",
            "loss",
            "retrieval_loss",
            "agree_query",
            "agree_prior",
            "matched_query",
            "matched_prior",
        ]
        self.csv_file = self.csv_path.open("a", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.csv_file, fieldnames=self.fields)
        if self.csv_path.stat().st_size == 0:
            self.writer.writeheader()
            self.csv_file.flush()
        self.jsonl_file = self.jsonl_path.open("a", encoding="utf-8")

    def log(self, row: dict) -> None:
        record = {field: row.get(field) for field in self.fields}
        self.writer.writerow(record)
        self.csv_file.flush()
        self.jsonl_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.jsonl_file.flush()

    def close(self) -> None:
        self.csv_file.close()
        self.jsonl_file.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_config_phi3_mmdocir.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--resume-global-step", type=int, default=0)
    parser.add_argument("--skip-micro-steps", type=int, default=0)
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
        modules_to_save=["linear_head"],
    )
    model = PeftModel.from_pretrained(model, args.checkpoint, is_trainable=True) if args.checkpoint else get_peft_model(model, lora_cfg)
    ensure_phi3_img_projection_bias(model)
    if cfg.training.gradient_checkpointing:
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable()
    model.train()

    teacher_prior = None
    teacher_query = None
    agree_lambda_prior = float(cfg.loss.get("agree_lambda_prior", 0.0))
    agree_lambda_query = float(cfg.loss.get("agree_lambda_query", 0.0))
    doc_instruction = str(cfg.data.get("doc_instruction", "Represent the user's input."))
    query_instruction = str(cfg.data.get("query_instruction", "Represent the user's input."))
    if cfg.data.get("attn_cache_path_prior"):
        teacher_prior = TeacherAttentionCache(cfg.data.attn_cache_path_prior)
        ensure_cache_prompt_mode(teacher_prior.metadata, "image_only")
        doc_instruction = str(teacher_prior.metadata.get("instruction", doc_instruction))
        print(
            f"Teacher prior cache: {cfg.data.attn_cache_path_prior}  "
            f"(source_mode={teacher_prior.source_mode}, instruction={doc_instruction!r})"
        )
    else:
        agree_lambda_prior = 0.0
    if cfg.data.get("attn_cache_path_query"):
        teacher_query = TeacherAttentionCache(cfg.data.attn_cache_path_query)
        ensure_cache_prompt_mode(teacher_query.metadata, "query_image")
        query_instruction = str(teacher_query.metadata.get("instruction", query_instruction))
        print(
            f"Teacher query cache: {cfg.data.attn_cache_path_query}  "
            f"(source_mode={teacher_query.source_mode}, instruction={query_instruction!r})"
        )
    else:
        agree_lambda_query = 0.0
    if teacher_prior is not None and agree_lambda_prior <= 0:
        print("[WARN] prior teacher cache provided but agree_lambda_prior <= 0, so prior alignment is disabled")

    collator = Phi3MMDocIRCollator(
        model_path=cfg.model.name_or_path,
        max_length=cfg.data.max_length,
        image_size=cfg.data.get("image_size", 1344),
        min_pixels=int(cfg.data.get("min_pixels", 4096)),
        max_pixels=int(cfg.data.get("max_pixels", 1048576)),
        query_instruction=query_instruction,
        doc_instruction=doc_instruction,
    )
    prior_instruction_token_ids = None
    if teacher_prior is not None and teacher_prior.source_mode == "instruction":
        prior_instruction_token_ids = collator.processor.tokenizer.encode(
            doc_instruction,
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
    prior_microbatch_size = max(1, int(cfg.loss.get("prior_microbatch_size", 1)))
    if bool(cfg.loss.get("use_matryoshka", True)):
        loss_fn = MatryoshkaAugmentedMaxSimLoss(dims=dims, temperature=temperature)
    else:
        loss_fn = AugmentedMaxSimLoss(temperature=temperature)

    optimizer = make_optimizer((p for p in model.parameters() if p.requires_grad), cfg.training.learning_rate)
    grad_accum = int(cfg.training.gradient_accumulation_steps)
    doc_microbatch_size = max(1, int(cfg.training.get("doc_microbatch_size", 1)))
    total_steps = max(1, (len(dataloader) * cfg.training.num_epochs) // grad_accum)
    warmup_steps = int(total_steps * float(cfg.training.get("warmup_ratio", 0.0)))
    if args.resume_global_step < 0 or args.skip_micro_steps < 0:
        parser.error("resume steps must be non-negative")
    if args.skip_micro_steps % grad_accum:
        parser.error("skip-micro-steps must align with gradient-accumulation-steps")
    if args.resume_global_step != args.skip_micro_steps // grad_accum:
        parser.error("resume-global-step must equal skip-micro-steps / gradient-accumulation-steps")
    if args.resume_global_step > total_steps or args.skip_micro_steps > len(dataloader):
        parser.error("resume position exceeds the configured training run")
    scheduler = make_scheduler(optimizer, warmup_steps, total_steps, args.resume_global_step)
    output_dir = Path(cfg.training.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metric_logger = TrainMetricLogger(output_dir)
    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()
    print(f"Metrics : {metric_logger.csv_path}")
    print(f"Metrics : {metric_logger.jsonl_path}")

    global_step = args.resume_global_step
    if global_step:
        print(f"Resuming adapter at global_step={global_step}; skipping {args.skip_micro_steps} micro-steps")
    optimizer.zero_grad()
    try:
        for epoch in range(cfg.training.num_epochs):
            progress = tqdm(dataloader, desc=f"epoch {epoch + 1}/{cfg.training.num_epochs}")
            for step, batch in enumerate(progress):
                if step < args.skip_micro_steps:
                    continue
                query_inputs = move_to_device(batch["query_inputs"], device)
                doc_inputs = batch["doc_inputs"]
                positive_doc_inputs = batch.get("positive_doc_inputs", batch["doc_inputs"])
                pos_count = int(batch["pos_count"])
                sample_ids = list(batch.get("sample_ids", []))[:pos_count]
                queries = list(batch.get("queries", []))[:pos_count]

                q_out = model(**clone_tensor_inputs(query_inputs))
                d_out = forward_model_in_chunks(
                    model,
                    doc_inputs,
                    device,
                    microbatch_size=doc_microbatch_size,
                    output_attentions=False,
                )
                q_emb, d_emb = q_out.hidden_states, d_out.hidden_states
                require_finite("query embeddings", q_emb)
                require_finite("document embeddings", d_emb)
                q_mask = batch.get("query_token_mask", q_out.attention_mask.bool()).to(device)
                d_mask = batch.get("doc_token_mask", d_out.attention_mask.bool()).to(device)
                doc_input_ids = doc_inputs["input_ids"].to(device)
                doc_attention_mask = doc_inputs["attention_mask"].to(device)
                retrieval_loss = loss_fn(q_emb, d_emb, q_mask, d_mask, pos_count=pos_count)
                require_finite("retrieval loss", retrieval_loss)
                loss = retrieval_loss
                agree_query_value = None
                agree_prior_value = None
                matched_query = 0
                matched_prior = 0

                if sample_ids and any(s is not None for s in sample_ids):
                    student_grids = None
                    if "doc_image_grid_thw" in batch:
                        student_grids = list(batch["doc_image_grid_thw"][:pos_count])
                    positive_student_grids = student_grids
                    if "positive_doc_image_grid_thw" in batch:
                        positive_student_grids = list(batch["positive_doc_image_grid_thw"][:pos_count])
                    if teacher_query is not None and agree_lambda_query > 0:
                        student_scores = extract_query_patch_scores_from_similarity(
                            q_emb=q_emb[:pos_count],
                            d_emb=d_emb[:pos_count],
                            q_input_ids=query_inputs["input_ids"][:pos_count],
                            q_attention_mask=query_inputs["attention_mask"][:pos_count],
                            d_input_ids=doc_input_ids[:pos_count],
                            d_attention_mask=doc_attention_mask[:pos_count],
                            queries=queries,
                            tokenizer=collator.processor.tokenizer,
                            image_token_id=collator.image_token_id,
                            mode=str(cfg.loss.get("agree_student_score_mode", "softmax_sum")),
                            d_image_mask=d_mask[:pos_count],
                        )
                        align, matched_query = attention_alignment_loss(
                            student_scores,
                            teacher_query.get_many(sample_ids),
                            loss_type=str(cfg.loss.get("agree_loss_type", "kl")),
                            student_grids=student_grids,
                            teacher_grids=teacher_query.get_many_grids(sample_ids),
                            allow_1d_resize=bool(cfg.loss.get("allow_1d_teacher_resize", False)),
                        )
                        if matched_query:
                            agree_query_value = align.to(loss.device)
                            loss = loss + agree_lambda_query * agree_query_value
                            require_finite("query alignment loss", agree_query_value)

                    if teacher_prior is not None and agree_lambda_prior > 0:
                        student_scores = []
                        for start in range(0, pos_count, prior_microbatch_size):
                            end = min(pos_count, start + prior_microbatch_size)
                            prior_inputs = move_to_device(slice_batch(positive_doc_inputs, start, end), device)
                            prior_mask = batch.get("positive_doc_token_mask", batch["doc_token_mask"][:pos_count])[start:end].to(device)
                            prior_out = model(**clone_tensor_inputs(prior_inputs), output_attentions=True)
                            student_scores.extend(
                                extract_prior_patch_scores(
                                    attentions=prior_out.attentions,
                                    input_ids=prior_inputs["input_ids"],
                                    attention_mask=prior_inputs["attention_mask"],
                                    image_token_id=collator.image_token_id,
                                    source_mode=teacher_prior.source_mode,
                                    instruction_token_ids=prior_instruction_token_ids,
                                    image_token_mask=prior_mask,
                                )
                            )
                        align, matched_prior = attention_alignment_loss(
                            student_scores,
                            teacher_prior.get_many(sample_ids),
                            loss_type=str(cfg.loss.get("agree_loss_type", "kl")),
                            student_grids=positive_student_grids,
                            teacher_grids=teacher_prior.get_many_grids(sample_ids),
                            allow_1d_resize=bool(cfg.loss.get("allow_1d_teacher_resize", False)),
                        )
                        if matched_prior:
                            agree_prior_value = align.to(loss.device)
                            loss = loss + agree_lambda_prior * agree_prior_value
                            require_finite("prior alignment loss", agree_prior_value)

                require_finite("total loss", loss)

                (loss / grad_accum).backward()
                if (step + 1) % grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1
                    lr = scheduler.get_last_lr()[0]
                    row = {
                        "time": time.time(),
                        "epoch": epoch + 1,
                        "global_step": global_step,
                        "micro_step": step + 1,
                        "lr": lr,
                        "loss": float(loss.detach().cpu()),
                        "retrieval_loss": float(retrieval_loss.detach().cpu()),
                        "agree_query": float(agree_query_value.detach().cpu()) if agree_query_value is not None else None,
                        "agree_prior": float(agree_prior_value.detach().cpu()) if agree_prior_value is not None else None,
                        "matched_query": matched_query,
                        "matched_prior": matched_prior,
                    }
                    metric_logger.log(row)
                    progress.set_postfix(
                        loss=f"{row['loss']:.4f}",
                        ret=f"{row['retrieval_loss']:.4f}",
                        aq="-" if row["agree_query"] is None else f"{row['agree_query']:.4f}",
                        ap="-" if row["agree_prior"] is None else f"{row['agree_prior']:.4f}",
                        mq=matched_query,
                        mp=matched_prior,
                        lr=f"{lr:.1e}",
                    )
                    if global_step % cfg.training.save_steps == 0:
                        model.save_pretrained(output_dir / f"checkpoint-{global_step}")

        model.save_pretrained(output_dir / "final")
        print(f"saved: {output_dir / 'final'}")
    finally:
        metric_logger.close()


if __name__ == "__main__":
    main()
