from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
import unicodedata
from PIL import Image
import logging

from peft import PeftModel
import torch
import torch.nn.functional as F

from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5PreTrainedModel, Qwen3_5Model
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config

from qwen_vl_utils.vision_process import process_vision_info

from transformers import AutoProcessor
from transformers.modeling_outputs import ModelOutput
from transformers.utils import TransformersKwargs
from transformers.processing_utils import Unpack
from transformers.cache_utils import Cache


MAX_LENGTH = 2048
QUERY_INSTRUCTION = "Represent the following query for searching relevant document pages."
DOC_INSTRUCTION = "Represent this document image to support information retrieval based on its text and layout."
IMAGE_BASE_FACTOR = 16
IMAGE_FACTOR = IMAGE_BASE_FACTOR * 2
MIN_PIXELS = 4 * IMAGE_FACTOR * IMAGE_FACTOR    # 4096
MAX_PIXELS = 1024 * IMAGE_FACTOR * IMAGE_FACTOR  # 1048576
PAD_TOKEN = "<|endoftext|>"


logger = logging.getLogger(__name__)


def _load_multimodal_processor(model_name_or_path: str):
    qwen3_vl_processor_cls = None
    try:
        from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor

        qwen3_vl_processor_cls = Qwen3VLProcessor
    except Exception:
        qwen3_vl_processor_cls = None

    if qwen3_vl_processor_cls is not None:
        try:
            return qwen3_vl_processor_cls.from_pretrained(model_name_or_path, padding_side="right")
        except Exception:
            pass

    processor = AutoProcessor.from_pretrained(
        model_name_or_path,
        padding_side="right",
        trust_remote_code=True,
    )
    if not hasattr(processor, "image_processor") or not hasattr(processor, "tokenizer"):
        raise RuntimeError(
            "Failed to load a multimodal processor for "
            f"{model_name_or_path}. Current transformers build returned {type(processor).__name__} "
            "instead of a vision-language processor. Upgrade transformers to a version that "
            "supports Qwen3/Qwen3.5-VL or use a compatible environment."
        )
    return processor


@dataclass
class ColQwen3_5ForEmbeddingOutput(ModelOutput):
    """Output of ColQwen3_5ForEmbedding.

    Args:
        hidden_states (`torch.FloatTensor`): Last hidden state of the model [B, N, D].
        attention_mask (`torch.Tensor`): Attention mask [B, N].
        attentions (`tuple`, optional): Per-layer attention tensors when
            forward() is called with output_attentions=True.  Each entry is
            [B, H, N, N] for full-attention layers or None for DeltaNet layers.
    """
    hidden_states:  Optional[torch.FloatTensor] = None
    attention_mask: Optional[torch.Tensor]      = None
    attentions:     Optional[tuple]             = None
    

class ColQwen3_5ForEmbedding(Qwen3_5PreTrainedModel):
    _checkpoint_conversion_mapping = {}
    accepts_loss_kwargs = False
    config: Qwen3_5Config

    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen3_5Model(config)
        # Optional dense projection head: D → embedding_dim (e.g. 2048 → 128)
        proj_dim = getattr(config, "embedding_dim", None)
        if proj_dim:
            self.linear_head = torch.nn.Linear(config.text_config.hidden_size, proj_dim, bias=False)
        else:
            self.linear_head = None
        self.post_init()
        # Orthogonal init AFTER post_init() so _init_weights() doesn't overwrite it
        if proj_dim:
            torch.nn.init.orthogonal_(self.linear_head.weight)
        
    def get_input_embeddings(self):
        return self.model.get_input_embeddings()
    
    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)
        
    def get_decoder(self):
        return self.model.get_decoder()
    
    def set_decoder(self, decoder):
        self.model.set_decoder(decoder)
        
    def get_image_features(self, pixel_values: torch.FloatTensor, image_grid_thw: Optional[torch.LongTensor] = None):
        return self.model.get_image_features(pixel_values, image_grid_thw)
    
    @property
    def language_model(self):
        return self.model.language_model
    
    @property
    def vision_model(self):
        return self.model.visual
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        output_attentions: bool = False,
        **kwargs: Unpack[TransformersKwargs],   # type: ignore
    ) -> Union[tuple, ColQwen3_5ForEmbeddingOutput]:
        r"""
        Returns:
            ColQwen3_5ForEmbeddingOutput with fields:
                - `hidden_states` ([B, N, D]): Last hidden state of the model.
                - `attention_mask` ([B, N]): Attention mask.
                - `attentions` (tuple | None): Per-layer attention tensors when
                  output_attentions=True.  GQA layers → [B, H, N, N]; DeltaNet
                  layers (Qwen3.5 hybrid) → None.
        """
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            output_attentions=output_attentions,
            **kwargs,
        )

        return ColQwen3_5ForEmbeddingOutput(
            hidden_states=outputs.last_hidden_state,
            attention_mask=attention_mask,
            attentions=outputs.attentions if output_attentions else None,
        )
        
        
class ColQwen3_5Embedder:
    def __init__(
        self,
        model_name_or_path: str = "Qwen/Qwen3.5-0.8B",
        lora_checkpoint: Optional[str] = None,
        max_length: int = MAX_LENGTH,
        min_pixels: int = MIN_PIXELS,
        max_pixels: int = MAX_PIXELS,
        default_instruction: str = "Represent the user's input.",
        embed_dim: Optional[int] = None,
        **kwargs,
    ):
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.embed_dim = embed_dim
        
        self.default_instruction = default_instruction
    
        self.model = ColQwen3_5ForEmbedding.from_pretrained(
            model_name_or_path, torch_dtype=torch.bfloat16,
        ).to(device)  # type: ignore

        if lora_checkpoint:
            self.model = PeftModel.from_pretrained(self.model, lora_checkpoint)
    
        self.processor = _load_multimodal_processor(model_name_or_path)  # type: ignore
        
        self.model.eval()
        
    @torch.no_grad()
    def forward(self, inputs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        outputs = self.model(**inputs)
        return {
            "embeddings": outputs.hidden_states,
            "attention_mask": inputs.get("attention_mask"),
        }
    
    def truncate_tokens(self, token_ids: List[int], max_length: int) -> List[int]:
        if len(token_ids) <= max_length:
            return token_ids

        special_token_ids = set(self.processor.tokenizer.all_special_ids)
        num_special = sum(1 for token_idx in token_ids if token_idx in special_token_ids)
        num_non_special_to_keep = max_length - num_special

        final_token_ids = []
        non_special_kept_count = 0

        for token_idx in token_ids:
            if token_idx in special_token_ids:
                final_token_ids.append(token_idx)
            elif non_special_kept_count < num_non_special_to_keep:
                final_token_ids.append(token_idx)
                non_special_kept_count += 1
                
        return final_token_ids
    
    def format_model_input(
        self, text: Optional[str] = None,
        image: Optional[Union[str, Image.Image]] = None,
        instruction: Optional[str] = None,
    ) -> List[Dict]:
        
        # Ensure instruction ends with punctuation
        if instruction:
            instruction = instruction.strip()
            if instruction and not unicodedata.category(instruction[-1]).startswith('P'):
                instruction = instruction + '.'

        content = []
        conversation = [
            {"role": "system", "content": [{"type": "text", "text": instruction or self.default_instruction}]},
            {"role": "user", "content": content}
        ]
        
        # Add text, image content to conversation
        if not text and not image:
            content.append({'type': 'text', 'text': "NULL"})
            return conversation
        
        if image:
            image_content = None
            if isinstance(image, Image.Image):
                image_content = image
            elif isinstance(image, str):
                image_content = image if image.startswith(('http', 'oss')) else 'file://' + image
            else:
                raise TypeError(f"Unrecognized image type: {type(image)}")

            # Add image input details to content
            if image_content:
                content.append({
                    'type': 'image', 'image': image_content,
                    "min_pixels": self.min_pixels,
                    "max_pixels": self.max_pixels
                })

        if text:
            content.append({'type': 'text', 'text': text})
            
        return conversation
            
    def _preprocess_inputs(self, conversations: List[List[Dict]]) -> Dict[str, torch.Tensor]:
        text = self.processor.apply_chat_template(
            conversations, add_generation_prompt=True, tokenize=False
        )

        try:
            images, video_inputs, video_kwargs = process_vision_info(
                conversations, image_patch_size=16,
                return_video_metadata=True, return_video_kwargs=True
            )
        
        except Exception as e:
            logger.error(f"Error in processing vision info: {e}")
            images = None
            video_inputs = None
            video_kwargs = {'do_sample_frames': False}
            text = self.processor.apply_chat_template(
                [{'role': 'user', 'content': [{'type': 'text', 'text': 'NULL'}]}], 
                add_generation_prompt=True, tokenize=False
            )

        if video_inputs is not None:
            videos, video_metadata = zip(*video_inputs)
            videos = list(videos)
            video_metadata = list(video_metadata)
        else:
            videos, video_metadata = None, None

        inputs = self.processor(
            text=text, images=images, videos=videos, video_metadata=video_metadata, truncation=True, 
            max_length=self.max_length, padding=True, do_resize=False, return_tensors='pt',
            **video_kwargs
        )
        return inputs

    # Pool the last hidden state by attention mask for embeddings
    @staticmethod
    def _pooling_last(hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        flipped_tensor = attention_mask.flip(dims=[1])
        last_one_positions = flipped_tensor.to(torch.int).argmax(dim=1)
        col = attention_mask.shape[1] - last_one_positions - 1
        row = torch.arange(hidden_state.shape[0], device=hidden_state.device)
        return hidden_state[row, col]
    
    def _truncate_dimensions(self, embeddings: torch.Tensor) -> torch.Tensor:
        # Truncate to embed_dim if specified
        if self.embed_dim is not None and embeddings.shape[-1] > self.embed_dim:
            return embeddings[:, :, :self.embed_dim]
        return embeddings

    # Process inputs to generate normalized embeddings
    def process(self, inputs: List[Dict[str, Any]], normalize: bool = True, pooling: bool = False) -> tuple:
        conversations = [self.format_model_input(
            text=ele.get('text'),
            image=ele.get('image'),
            instruction=ele.get('instruction'),
        ) for ele in inputs]

        processed_inputs = self._preprocess_inputs(conversations)
        processed_inputs = {k: v.to(self.model.device) for k, v in processed_inputs.items()}

        outputs = self.forward(processed_inputs)

        embeddings = outputs['embeddings']
        attention_mask = outputs['attention_mask']

        linear_head = getattr(self.model, 'linear_head', None)

        if pooling:
            embeddings = self._pooling_last(embeddings, attention_mask)
            if linear_head is not None:
                embeddings = linear_head(embeddings.float())
            if normalize:
                embeddings = F.normalize(embeddings, p=2, dim=-1)
            return embeddings, attention_mask

        else:
            if linear_head is not None:
                # per-token projection (multi-vector + projection head)
                embeddings = linear_head(embeddings.float())
            else:
                embeddings = self._truncate_dimensions(embeddings)
            if normalize:
                embeddings = F.normalize(embeddings, p=2, dim=-1)
            return embeddings, attention_mask

    # ── Convenience encode methods ────────────────────────────────────────

    def encode_queries(self, queries: List[str], normalize: bool = True) -> torch.Tensor:
        """Multi-vector query embeddings → (Q, Lq, D)."""
        inputs = [{"text": q, "instruction": QUERY_INSTRUCTION} for q in queries]
        embeddings, _ = self.process(inputs, normalize=normalize, pooling=False)
        return embeddings

    def encode_docs(
        self,
        images: Optional[List] = None,
        texts: Optional[List[str]] = None,
        normalize: bool = True,
    ) -> torch.Tensor:
        """Multi-vector doc embeddings → (N, Ld, D). Pass images or texts."""
        if images is not None:
            inputs = [{"image": img, "instruction": DOC_INSTRUCTION} for img in images]
        elif texts is not None:
            inputs = [{"text": t, "instruction": DOC_INSTRUCTION} for t in texts]
        else:
            raise ValueError("Either images or texts must be provided")
        embeddings, _ = self.process(inputs, normalize=normalize, pooling=False)
        return embeddings

    def encode_queries_dense(self, queries: List[str], normalize: bool = True) -> torch.Tensor:
        """Dense (pooled) query embeddings → (Q, D)."""
        inputs = [{"text": q, "instruction": QUERY_INSTRUCTION} for q in queries]
        embeddings, _ = self.process(inputs, normalize=normalize, pooling=True)
        return embeddings

    def encode_docs_dense(
        self,
        images: Optional[List] = None,
        texts: Optional[List[str]] = None,
        normalize: bool = True,
    ) -> torch.Tensor:
        """Dense (pooled) doc embeddings → (N, D). Pass images or texts."""
        if images is not None:
            inputs = [{"image": img, "instruction": DOC_INSTRUCTION} for img in images]
        elif texts is not None:
            inputs = [{"text": t, "instruction": DOC_INSTRUCTION} for t in texts]
        else:
            raise ValueError("Either images or texts must be provided")
        embeddings, _ = self.process(inputs, normalize=normalize, pooling=True)
        return embeddings
