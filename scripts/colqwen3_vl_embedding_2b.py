from typing import ClassVar, List, Optional, Tuple, Union
from PIL import Image
import importlib
import logging
from abc import ABC, abstractmethod

import torch
from torch import nn

from transformers.models.qwen3_vl import Qwen3VLConfig, Qwen3VLModel
from transformers import Qwen2VLProcessor
from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize

from transformers import BatchEncoding, BatchFeature


logger = logging.getLogger(__name__)


try:
    from fast_plaid import search
except ImportError:
    logger.info(
        "FastPlaid is not installed.If you want to use it:Instal with `pip install --no-deps fast-plaid fastkmeans`"
    )


def get_torch_device(device: str = "auto") -> str:
    """
    Returns the device (string) to be used by PyTorch.

    `device` arg defaults to "auto" which will use:
    - "cuda:0" if available
    - else "mps" if available
    - else "cpu".
    """

    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda:0"
        elif torch.backends.mps.is_available():  # for Apple Silicon
            device = "mps"
        else:
            device = "cpu"
        logger.info(f"Using device: {device}")

    return device


class BaseVisualRetrieverProcessor(ABC):
    """
    Base class for visual retriever processors.
    """

    query_prefix: ClassVar[str] = "Query: "  # Default prefix for queries. Override in subclasses if needed.

    @abstractmethod
    def process_images(
        self,
        images: List[Image.Image],
    ) -> Union[BatchFeature, BatchEncoding]:
        """
        Process a list of images into a format suitable for the model.
        Args:
            images (List[Image.Image]): List of images to process.
        Returns:
            Union[BatchFeature, BatchEncoding]: Processed images.
        """
        pass

    @abstractmethod
    def process_texts(self, texts: List[str]) -> Union[BatchFeature, BatchEncoding]:
        """
        Process a list of texts into a format suitable for the model.

        Args:
            texts: List of input texts.

        Returns:
            Union[BatchFeature, BatchEncoding]: Processed texts.
        """
        pass

    def process_queries(
        self,
        texts: Optional[List[str]] = None,
        queries: Optional[List[str]] = None,
        max_length: int = 50,
        suffix: Optional[str] = None,
    ) -> Union[BatchFeature, BatchEncoding]:
        """
        Process a list of queries into a format suitable for the model.

        Args:
            texts: List of input texts.
            [DEPRECATED] max_length: Maximum length of the text.
            suffix: Suffix to append to each text. If None, the default query augmentation token is used.

        Returns:
            Union[BatchFeature, BatchEncoding]: Processed texts.

        NOTE: This function will be deprecated. Use `process_texts` instead.
        It is kept to maintain back-compatibility with vidore evaluator.
        """

        if texts and queries:
            raise ValueError("Only one of 'texts' or 'queries' should be provided.")
        if queries is not None:
            texts = queries
        elif texts is None:
            raise ValueError("No texts or queries provided.")

        if suffix is None:
            suffix = self.query_augmentation_token * 10

        # Add the query prefix and suffix to each text
        texts = [self.query_prefix + text + suffix for text in texts]

        return self.process_texts(texts=texts)

    @abstractmethod
    def score(
        self,
        qs: Union[torch.Tensor, List[torch.Tensor]],
        ps: Union[torch.Tensor, List[torch.Tensor]],
        device: Optional[Union[str, torch.device]] = None,
        **kwargs,
    ) -> torch.Tensor:
        pass

    @staticmethod
    def score_single_vector(
        qs: Union[torch.Tensor, List[torch.Tensor]],
        ps: Union[torch.Tensor, List[torch.Tensor]],
        device: Optional[Union[str, torch.device]] = None,
    ) -> torch.Tensor:
        """
        Compute the dot product score for the given single-vector query and passage embeddings.
        """
        device = device or get_torch_device("auto")

        if len(qs) == 0:
            raise ValueError("No queries provided")
        if len(ps) == 0:
            raise ValueError("No passages provided")

        if isinstance(qs, list):
            qs = torch.stack(qs).to(device)
            ps = torch.stack(ps).to(device)

        scores = torch.einsum("bd,cd->bc", qs, ps)
        assert scores.shape[0] == len(qs), f"Expected {len(qs)} scores, got {scores.shape[0]}"

        scores = scores.to(torch.float32)
        return scores

    @staticmethod
    def score_multi_vector(
        qs: Union[torch.Tensor, List[torch.Tensor]],
        ps: Union[torch.Tensor, List[torch.Tensor]],
        batch_size: int = 128,
        device: Optional[Union[str, torch.device]] = None,
    ) -> torch.Tensor:
        """
        Compute the late-interaction/MaxSim score (ColBERT-like) for the given multi-vector
        query embeddings (`qs`) and passage embeddings (`ps`). For ColPali, a passage is the
        image of a document page.

        Because the embedding tensors are multi-vector and can thus have different shapes, they
        should be fed as:
        (1) a list of tensors, where the i-th tensor is of shape (sequence_length_i, embedding_dim)
        (2) a single tensor of shape (n_passages, max_sequence_length, embedding_dim) -> usually
            obtained by padding the list of tensors.

        Args:
            qs (`Union[torch.Tensor, List[torch.Tensor]`): Query embeddings.
            ps (`Union[torch.Tensor, List[torch.Tensor]`): Passage embeddings.
            batch_size (`int`, *optional*, defaults to 128): Batch size for computing scores.
            device (`Union[str, torch.device]`, *optional*): Device to use for computation. If not
                provided, uses `get_torch_device("auto")`.

        Returns:
            `torch.Tensor`: A tensor of shape `(n_queries, n_passages)` containing the scores. The score
            tensor is saved on the "cpu" device.
        """
        device = device or get_torch_device("auto")

        if len(qs) == 0:
            raise ValueError("No queries provided")
        if len(ps) == 0:
            raise ValueError("No passages provided")

        scores_list: List[torch.Tensor] = []

        for i in range(0, len(qs), batch_size):
            scores_batch = []
            qs_batch = torch.nn.utils.rnn.pad_sequence(qs[i : i + batch_size], batch_first=True, padding_value=0).to(
                device
            )
            for j in range(0, len(ps), batch_size):
                ps_batch = torch.nn.utils.rnn.pad_sequence(
                    ps[j : j + batch_size], batch_first=True, padding_value=0
                ).to(device)
                scores_batch.append(torch.einsum("bnd,csd->bcns", qs_batch, ps_batch).max(dim=3)[0].sum(dim=2))
            scores_batch = torch.cat(scores_batch, dim=1).cpu()
            scores_list.append(scores_batch)

        scores = torch.cat(scores_list, dim=0)
        assert scores.shape[0] == len(qs), f"Expected {len(qs)} scores, got {scores.shape[0]}"

        scores = scores.to(torch.float32)
        return scores

    @staticmethod
    def get_topk_plaid(
        qs: Union[torch.Tensor, List[torch.Tensor]],
        plaid_index: "search.FastPlaid",
        k: int = 10,
        batch_size: int = 128,
        device: Optional[Union[str, torch.device]] = None,
    ) -> torch.Tensor:
        """
        Experimental: Compute the late-interaction/MaxSim score (ColBERT-like) for the given multi-vector
        query embeddings (`qs`) and passage embeddings endoded in a plaid index. For ColPali, a passage is the
        image of a document page.
        """
        device = device or get_torch_device("auto")

        if len(qs) == 0:
            raise ValueError("No queries provided")

        scores_list: List[torch.Tensor] = []

        for i in range(0, len(qs), batch_size):
            scores_batch = []
            qs_batch = torch.nn.utils.rnn.pad_sequence(qs[i : i + batch_size], batch_first=True, padding_value=0).to(
                device
            )
            # Use the plaid index to get the top-k scores
            scores_batch = plaid_index.search(
                queries_embeddings=qs_batch.to(torch.float32),
                top_k=k,
            )
            scores_list.append(scores_batch)

        return scores_list

    @staticmethod
    def create_plaid_index(
        ps: Union[torch.Tensor, List[torch.Tensor]],
        device: Optional[Union[str, torch.device]] = None,
    ) -> torch.Tensor:
        """
        Experimental: Create a FastPlaid index from the given passage embeddings.
        Args:
            ps (`Union[torch.Tensor, List[torch.Tensor]]`): Passage embeddings. Should be a list of tensors,
                where each tensor is of shape (sequence_length_i, embedding_dim).
            device (`Optional[Union[str, torch.device]]`, *optional*): Device to use for computation. If not
                provided, uses `get_torch_device("auto")`.
        """
        # assert fast_plaid is installed
        if not importlib.util.find_spec("fast_plaid"):
            raise ImportError("FastPlaid is not installed. Please install it with `pip install fast-plaid`.")

        fast_plaid_index = search.FastPlaid(index="index")
        # torch.nn.utils.rnn.pad_sequence(ds, batch_first=True, padding_value=0).to(device)
        device = device or get_torch_device("auto")
        fast_plaid_index.create(documents_embeddings=[d.to(device).to(torch.float32) for d in ps])
        return fast_plaid_index

    @abstractmethod
    def get_n_patches(
        self,
        image_size: Tuple[int, int],
        *args,
        **kwargs,
    ) -> Tuple[int, int]:
        """
        Get the number of patches (n_patches_x, n_patches_y) that will be used to process an
        image of size (height, width) with the given patch size.
        """
        pass


class ColQwen3(Qwen3VLModel):
    """
    ColQwen3 model implementation following ColPali paper architecture.
    
    Key insights from ColPali paper:
    1. Use ALL tokens (text + vision) as multi-vector embeddings
    2. Vision tokens carry visual information for retrieval
    3. Handle DeepStack features and image_grid_thw properly
    4. Use late interaction (MaxSim) for scoring
    
    Inherits from Qwen3VLModel to get proper vision processing with DeepStack.
    """
    
    main_input_name: ClassVar[str] = "doc_input_ids"
    
    def __init__(self, config: Qwen3VLConfig, mask_non_image_embeddings: bool = False):
        super().__init__(config=config)
        
        # Read dim from config if available, otherwise default to 128
        # This allows loading models with different projection dimensions (e.g., 512)
        self.dim = getattr(config, 'dim', 128)
        
        # Qwen3-VL stores hidden_size in text_config, Qwen2.5-VL in root config
        # But both should work with config.hidden_size after super().__init__
        self.custom_text_proj = nn.Linear(self.config.text_config.hidden_size, self.dim)
        self.padding_side = "left"
        self.mask_non_image_embeddings = mask_non_image_embeddings
        # EXACTLY like ColQwen2.5 - just call post_init()
        self.post_init()
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        """
        Load pretrained model.
        
        CRITICAL: Immer config als Qwen3VLConfig laden, auch bei HF Models!
        """
        from transformers import AutoConfig
        from pathlib import Path
        import json
        
        # Check if config is already provided
        if 'config' not in kwargs:
            # Load config manually (lokal ODER HF!)
            try:
                if Path(pretrained_model_name_or_path).is_dir():
                    # Lokaler Pfad
                    config_path = Path(pretrained_model_name_or_path) / "config.json"
                    with open(config_path) as f:
                        config_dict = json.load(f)
                else:
                    # HF Model - download config
                    from huggingface_hub import hf_hub_download
                    config_file = hf_hub_download(pretrained_model_name_or_path, "config.json")
                    with open(config_file) as f:
                        config_dict = json.load(f)
                
                # Force model_type to qwen3_vl für korrektes Parsing
                original_model_type = config_dict.get("model_type")
                config_dict["model_type"] = "qwen3_vl"
                
                # Create Qwen3VLConfig from dict (parsed rope_scaling korrekt!)
                config = Qwen3VLConfig.from_dict(config_dict)
                
                # Restore original model_type
                config.model_type = original_model_type if original_model_type else "colqwen3"
                
                # CRITICAL: Read dim from config.json if present (for upgraded projection models)
                if "dim" in config_dict:
                    config.dim = config_dict["dim"]
                    print(f"   📐 Using custom projection dim from config: {config.dim}")
                
                # Add config to kwargs
                kwargs['config'] = config
            except Exception as e:
                print(f"⚠️ Could not pre-load config: {e}")
                # Fallback to default loading
                pass
        
        # Load with config (if successfully parsed) or default
        return super().from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
    
    def forward(self, *args, **kwargs) -> torch.Tensor:
        """
        Forward pass following ColPali paper architecture.
        
        MUST handle pixel_values EXACTLY like ColQwen2.5!
        """
        
        # CRITICAL FIX: Handle pixel_values EXACTLY like ColQwen2.5
        # The processor pads them, we must unpad them here!
        if "pixel_values" in kwargs:
            offsets = kwargs["image_grid_thw"][:, 1] * kwargs["image_grid_thw"][:, 2]  # (batch_size,)
            kwargs["pixel_values"] = torch.cat(
                [pixel_sequence[:offset] for pixel_sequence, offset in zip(kwargs["pixel_values"], offsets)],
                dim=0,
            )
        
        # Remove arguments that shouldn't go to parent forward
        kwargs.pop("return_dict", True)
        kwargs.pop("output_hidden_states", None)
        kwargs.pop("use_cache", None)
        
        # CRITICAL: Call the parent forward to get full model output with hidden states
        # Qwen3VLModel.forward returns Qwen3VLCausalLMOutputWithPast which has hidden_states
        output = (
            super()
            .forward(*args, **kwargs, use_cache=False, output_hidden_states=True, return_dict=True)
        )
        
        # Extract hidden states from the output object
        last_hidden_states = output.hidden_states[-1] if hasattr(output, 'hidden_states') else output.last_hidden_state
        
        # Project to ColPali dimension
        proj = self.custom_text_proj(last_hidden_states)  # (batch_size, sequence_length, dim)
        
        # L2 normalization - ESSENTIAL for ColBERT-style retrieval!
        proj = proj / proj.norm(dim=-1, keepdim=True)  # EXACTLY like ColQwen2.5 - NO epsilon!
        
        # Apply attention mask AFTER normalization (EXACTLY like ColQwen2.5)
        proj = proj * kwargs["attention_mask"].unsqueeze(-1)  # (batch_size, sequence_length, dim)
        
        # Optionally mask non-image embeddings
        if "pixel_values" in kwargs and self.mask_non_image_embeddings:
            # Qwen3-VL specific image token ID (from documentation)
            image_token_id = getattr(self.config, 'image_token_id', 151655)
            # CRITICAL: Ensure input_ids is on same device (for Multi-GPU)
            input_ids = kwargs["input_ids"].to(proj.device)
            image_mask = (input_ids == image_token_id).unsqueeze(-1)
            proj = proj * image_mask
        
        return proj
    
    @property
    def patch_size(self) -> int:
        """Get patch size - following ColQwen2.5 pattern"""
        # Qwen3-VL might structure this differently
        if hasattr(self, 'visual') and hasattr(self.visual, 'config'):
            return getattr(self.visual.config, 'patch_size', 16)
        return 16  # Default for Qwen3-VL
    
    @property
    def spatial_merge_size(self) -> int:
        """Get spatial merge size - following ColQwen2.5 pattern"""
        if hasattr(self, 'visual') and hasattr(self.visual, 'config'):
            return getattr(self.visual.config, 'spatial_merge_size', 2)
        return 2  # Default for Qwen3-VL
    
    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """Enable gradient checkpointing for memory efficiency"""
        if hasattr(super(), 'gradient_checkpointing_enable'):
            super().gradient_checkpointing_enable(gradient_checkpointing_kwargs)
    
    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing"""
        if hasattr(super(), 'gradient_checkpointing_disable'):
            super().gradient_checkpointing_disable()


class ColQwen3Processor(BaseVisualRetrieverProcessor, Qwen2VLProcessor):
    """
    Processor for ColQwen3 - using Qwen2VLProcessor which handles images correctly
    """
    
    visual_prompt_prefix: ClassVar[str] = (
        "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Describe the image.<|im_end|><|endoftext|>"
    )
    # Use same augmentation token as ColQwen2.5 - this is CRITICAL for training!
    query_augmentation_token: ClassVar[str] = "<|endoftext|>"
    image_token: ClassVar[str] = "<|image_pad|>"
    image_token_id: ClassVar[int] = 151655
    
    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        video_processor=None,
        chat_template=None,
        **kwargs,
    ):
        # Explicit named params required by transformers 5.x:
        # ProcessorMixin.get_attributes() inspects __init__ signature to locate
        # sub-components (tokenizer, image_processor, video_processor).
        # *args/**kwargs → get_attributes() returns [] → setattr never called →
        # self.tokenizer missing and all components silently None.
        super().__init__(
            image_processor=image_processor,
            tokenizer=tokenizer,
            video_processor=video_processor,
            chat_template=chat_template,
            **kwargs,
        )
        if tokenizer is not None:
            self.tokenizer.padding_side = "left"

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        *args,
        device_map: Optional[str] = None,
        max_num_visual_tokens: Optional[int] = None,
        **kwargs,
    ):
        # With __init__ now declaring named modality params, get_attributes()
        # correctly returns ['image_processor', 'tokenizer', 'video_processor']
        # and ProcessorMixin._get_arguments_from_pretrained auto-loads all three.
        # device_map is model-level only — do not forward to processor.
        instance = super().from_pretrained(pretrained_model_name_or_path, *args, **kwargs)

        if max_num_visual_tokens is not None:
            instance.image_processor.max_pixels = max_num_visual_tokens * 28 * 28
            instance.image_processor.size["longest_edge"] = instance.image_processor.max_pixels

        return instance
    
    def process_images(
        self,
        images: List[Image.Image],
    ) -> Union[BatchFeature, BatchEncoding]:
        """Process images - EXACTLY like ColQwen2.5"""
        images = [image.convert("RGB") for image in images]
        
        # Use self directly since we inherit from Qwen2VLProcessor
        batch_doc = self(
            text=[self.visual_prompt_prefix] * len(images),
            images=images,
            padding="longest",
            return_tensors="pt",
        )
        
        # CRITICAL: Remove token_type_ids as per Qwen3VL documentation
        batch_doc.pop("token_type_ids", None)
        
        # CRITICAL FIX: Apply same pixel_values padding as ColQwen2.5!
        # This ensures consistent behavior with DDP on multiple GPUs
        offsets = batch_doc["image_grid_thw"][:, 1] * batch_doc["image_grid_thw"][:, 2]  # (batch_size,)
        
        # Split the pixel_values tensor into a list of tensors, one per image
        pixel_values = list(
            torch.split(batch_doc["pixel_values"], offsets.tolist())
        )  # [(num_patches_image_0, pixel_values), ..., (num_patches_image_n, pixel_values)]
        
        # Pad the list of pixel_value tensors to the same length along the sequence dimension
        batch_doc["pixel_values"] = torch.nn.utils.rnn.pad_sequence(
            pixel_values, batch_first=True
        )  # (batch_size, max_num_patches, pixel_values)
        
        return batch_doc
    
    def process_texts(
        self, 
        texts: List[str]
    ) -> Union[BatchFeature, BatchEncoding]:
        """Process texts - EXACTLY like ColQwen2.5"""
        batch = self(
            text=texts,
            return_tensors="pt",
            padding="longest",
        )
        
        # CRITICAL: Remove token_type_ids as per Qwen3VL documentation
        batch.pop("token_type_ids", None)
        
        return batch
    def score(
        self,
        qs: List[torch.Tensor],
        ps: List[torch.Tensor],
        device: Optional[Union[str, torch.device]] = None,
        **kwargs,
    ) -> torch.Tensor:
        return self.score_multi_vector(qs, ps, device=device, **kwargs)
    
    def get_n_patches(
        self,
        image_size: Tuple[int, int],
        spatial_merge_size: int,
    ) -> Tuple[int, int]:
        """
        Get the number of patches (n_patches_x, n_patches_y) that will be used to process an image of
        size (height, width) with the given patch size.
        
        EXACTLY like ColQwen2.5 - using smart_resize!
        """
        patch_size = self.image_processor.patch_size
        
        height_new, width_new = smart_resize(
            width=image_size[0],
            height=image_size[1],
            factor=patch_size * self.image_processor.merge_size,
            min_pixels=self.image_processor.size["shortest_edge"],
            max_pixels=self.image_processor.size["longest_edge"],
        )
        
        n_patches_x = width_new // patch_size // spatial_merge_size
        n_patches_y = height_new // patch_size // spatial_merge_size
        
        return n_patches_x, n_patches_y
    
    def get_image_mask(self, batch_images: BatchFeature) -> torch.Tensor:
        return batch_images.input_ids == self.image_token_id

