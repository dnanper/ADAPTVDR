from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from PIL import Image

from colpali_engine.models.paligemma.colpali.modeling_colpali import ColPali
from colpali_engine.models.paligemma.colpali.processing_colpali import ColPaliProcessor

from scripts.adaptive_pruning import AdaptivePruner, IMAGE_TOKEN_ID_COLPALI, PruningStats
from scripts.precompute_teacher_attn import find_subsequence_positions


_DUMMY_IMAGE = Image.new("RGB", (448, 448), color=(255, 255, 255))
DEFAULT_PRUNE_QUERY = "Represent this document for retrieval."


@dataclass
class ColPaliEmbedderOutput:
    embeddings:     torch.FloatTensor          # [B, N, 128]
    attention_mask: torch.Tensor               # [B, N]
    attentions:     Optional[tuple] = None     # tuple([B, H, N, N]) per layer, if requested
    input_ids:      Optional[torch.Tensor] = None  # [B, N]


class ColPaliEmbedder:
    """Embedder for ColPali (PaliGemma-3B + Linear 2048->128 projection)."""

    def __init__(self, model_name_or_path: str, embed_dim: Optional[int] = None, **kwargs):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device    = device
        self.embed_dim = embed_dim

        self.model = ColPali.from_pretrained(
            model_name_or_path, torch_dtype=torch.bfloat16, device_map=str(device), **kwargs
        )
        self.model.eval()
        self.processor = ColPaliProcessor.from_pretrained(model_name_or_path)

    @torch.no_grad()
    def _forward(
        self,
        inputs: Dict[str, torch.Tensor],
        output_attentions: bool = False,
    ) -> ColPaliEmbedderOutput:
        inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                  for k, v in inputs.items()}
        # self.model        = ColPali (wraps PaliGemma via composition)
        # self.model.model  = PaliGemmaForConditionalGeneration (has hidden_states & attentions)
        # ColPali.forward() pops output_hidden_states and returns plain tensor → bypass it
        pv = inputs.get("pixel_values")
        outputs = self.model.model(
            input_ids=inputs.get("input_ids"),
            attention_mask=inputs.get("attention_mask"),
            pixel_values=pv.to(dtype=self.model.dtype) if pv is not None else None,
            output_hidden_states=True,
            output_attentions=output_attentions,
            return_dict=True,
        )
        last_hidden = outputs.hidden_states[-1]                        # [B, N, 2048] bfloat16
        emb  = self.model.custom_text_proj(last_hidden)               # [B, N, 128]  bfloat16
        emb  = F.normalize(emb.float(), p=2, dim=-1).to(last_hidden.dtype)  # norm fp32, store bf16
        mask = inputs["attention_mask"]
        emb  = emb * mask.unsqueeze(-1).to(emb.dtype)

        if self.embed_dim is not None and emb.shape[-1] > self.embed_dim:
            emb = F.normalize(emb[..., : self.embed_dim], p=2, dim=-1)
            emb = emb * mask.unsqueeze(-1).to(emb.dtype)

        return ColPaliEmbedderOutput(
            embeddings=emb,
            attention_mask=mask,
            attentions=outputs.attentions if output_attentions else None,
            input_ids=inputs.get("input_ids"),
        )

    def embed_images_pruned(
        self,
        images: List[Union[str, Image.Image]],
        pruner: Optional[AdaptivePruner] = None,
        r_min: float = 0.3,
        r_max: float = 0.99,
        query_text: Optional[Union[str, List[str]]] = None,
        score_mode: str = "softmax_sum",
    ) -> Tuple[List[torch.Tensor], PruningStats]:
        """Encode images and prune patch embeddings with adaptive entropy pruning.

        Requires attn_implementation="eager" at model load (flash_attn doesn't
        return attention weights).

        Args:
            images:  List of PIL images or file paths.
            pruner:  AdaptivePruner instance. If None, one is created with r_min/r_max.
            r_min:   Min keep ratio (used only when pruner=None).
            r_max:   Max keep ratio (used only when pruner=None).
            query_text: Optional query or dummy-query used to score patches from
                        query-document interaction instead of doc self-attention.
                        Pass a string to reuse the same query for all images, or
                        a list[str] of the same length as `images`.
            score_mode: Aggregation over query×patch similarity. One of
                        {"softmax_sum", "sum", "mean"}.
        Returns:
            (pruned_list, stats)
              pruned_list: List[Tensor[k_b, 128]] — variable length per sample
              stats:       PruningStats
        """
        if pruner is None:
            pruner = AdaptivePruner(r_min=r_min, r_max=r_max,
                                    image_token_id=IMAGE_TOKEN_ID_COLPALI)
        if query_text is None:
            batch = self.process_images(images)
            out   = self._forward(batch, output_attentions=True)
            return pruner.prune_doc(
                hidden_states=out.embeddings,
                attentions=out.attentions,
                input_ids=out.input_ids,
                attention_mask=out.attention_mask,
            )

        queries = [query_text] * len(images) if isinstance(query_text, str) else list(query_text)
        if len(queries) != len(images):
            raise ValueError("query_text list must have the same length as images")

        doc_batch = self.process_images(images)
        doc_out = self._forward(doc_batch, output_attentions=False)
        patch_scores = self._compute_query_patch_scores(
            queries=queries,
            query_batch=self.process_queries(queries),
            doc_out=doc_out,
            score_mode=score_mode,
        )
        return pruner.prune_doc_with_patch_scores(
            hidden_states=doc_out.embeddings,
            patch_scores_list=patch_scores,
            input_ids=doc_out.input_ids,
            attention_mask=doc_out.attention_mask,
        )

    def process_images(self, images: List[Union[str, Image.Image]]) -> Dict[str, torch.Tensor]:
        pil_images = [Image.open(img).convert("RGB") if isinstance(img, str)
                      else img.convert("RGB") for img in images]
        return self.processor(
            text=["<image>\n"] * len(pil_images),
            images=pil_images,
            return_tensors="pt",
            padding="longest",
            truncation=False,
        )

    def process_queries(self, queries: List[str]) -> Dict[str, torch.Tensor]:
        return self.processor(
            text=[f"<image>Question: {query}\nAnswer:" for query in queries],
            images=[_DUMMY_IMAGE] * len(queries),
            return_tensors="pt",
            padding="longest",
        )

    def _compute_query_patch_scores(
        self,
        queries: List[str],
        query_batch: Dict[str, torch.Tensor],
        doc_out: ColPaliEmbedderOutput,
        score_mode: str = "softmax_sum",
    ) -> List[torch.Tensor]:
        query_out = self._forward(query_batch, output_attentions=False)
        tokenizer = self.processor.tokenizer
        q_emb = query_out.embeddings.float()
        d_emb = doc_out.embeddings.float()
        q_input_ids = query_out.input_ids
        q_attention_mask = query_out.attention_mask
        d_input_ids = doc_out.input_ids
        d_attention_mask = doc_out.attention_mask

        patch_scores: List[torch.Tensor] = []
        for idx, query in enumerate(queries):
            query_token_ids = tokenizer.encode(str(query).strip(), add_special_tokens=False)
            if not query_token_ids:
                patch_scores.append(torch.zeros(0, device=self.device))
                continue

            valid_q_ids = q_input_ids[idx][q_attention_mask[idx].bool()].tolist()
            try:
                query_positions = torch.tensor(
                    find_subsequence_positions(valid_q_ids, query_token_ids),
                    dtype=torch.long,
                    device=self.device,
                )
            except ValueError:
                query_mask = (
                    (q_input_ids[idx] != IMAGE_TOKEN_ID_COLPALI)
                    & q_attention_mask[idx].bool()
                )
                query_positions = query_mask.nonzero(as_tuple=False).squeeze(-1)

            img_positions = (
                (d_input_ids[idx] == IMAGE_TOKEN_ID_COLPALI)
                & d_attention_mask[idx].bool()
            ).nonzero(as_tuple=False).squeeze(-1)
            if img_positions.numel() == 0:
                patch_scores.append(torch.zeros(0, device=self.device))
                continue

            sim = q_emb[idx][query_positions] @ d_emb[idx][img_positions].T
            if score_mode == "sum":
                patch_scores.append(sim.sum(dim=0))
            elif score_mode == "mean":
                patch_scores.append(sim.mean(dim=0))
            else:
                patch_scores.append(sim.softmax(dim=-1).sum(dim=0))

        return patch_scores

    def embed_images_pruned_with_dummy_query(
        self,
        images: List[Union[str, Image.Image]],
        dummy_query: str = DEFAULT_PRUNE_QUERY,
        pruner: Optional[AdaptivePruner] = None,
        r_min: float = 0.3,
        r_max: float = 0.99,
        score_mode: str = "softmax_sum",
    ) -> Tuple[List[torch.Tensor], PruningStats]:
        """Prune document embeddings using query-document interaction with a fixed dummy query."""
        return self.embed_images_pruned(
            images=images,
            pruner=pruner,
            r_min=r_min,
            r_max=r_max,
            query_text=dummy_query,
            score_mode=score_mode,
        )

    def process(self, inputs: List[Dict[str, Any]], normalize: bool = True, pooling: bool = False) -> tuple:
        if all("image" in x for x in inputs):
            batch = self.process_images([x["image"] for x in inputs])
        elif all("text" in x for x in inputs):
            batch = self.process_queries([x["text"] for x in inputs])
        else:
            raise ValueError("Batch must be entirely all-image or all-text.")

        out = self._forward(batch)
        emb, mask = out.embeddings, out.attention_mask

        if pooling:
            last_idx = mask.shape[1] - mask.flip(dims=[1]).argmax(dim=1) - 1
            emb = emb[torch.arange(emb.shape[0], device=emb.device), last_idx]

        return emb, mask
