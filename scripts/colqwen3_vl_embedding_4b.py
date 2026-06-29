import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor
from typing import List, Union, Optional


class OpsColQwen3Embedder:
    """
    Embedder for OpsColQwen3-4B model.
    """

    def __init__(
        self,
        model_name: str = "OpenSearch-AI/Ops-Colqwen3-4B",
        dims: int = 2560,
        device: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize the embedder.

        Args:
            model_name: Model path or hub name
            dims: Embedding dimensions
            device: Device to use for inference ('mps', 'cuda', or 'cpu')
            **kwargs: Additional arguments passed to from_pretrained
        """

        device_map = kwargs.pop('device_map', None)
        if not device_map:
            if device:
                device_map = device
            elif torch.cuda.is_available():
                device_map = "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device_map = "mps" # Use MPS for Apple Silicon
            else:
                device_map = "cpu"

        dtype = kwargs.pop('dtype', torch.float16 if device_map != "cpu" else torch.float32)

        self.model = AutoModel.from_pretrained(
            model_name,
            dims=dims,
            trust_remote_code=True,
            dtype=dtype,
            device_map=device_map,
            **kwargs
        )
        self.model.eval()

        self.processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True,
            **kwargs
        )

        self.device = device_map
        self.dims = dims

    def encode_queries(
        self,
        queries: List[str]
    ) -> List[torch.Tensor]:
        """
        Encode a list of text queries.

        Args:
            queries: List of query texts

        Returns:
            List of query embeddings
        """
        query_inputs = self.processor.process_queries(queries)
        query_inputs = {k: v.to(self.device) for k, v in query_inputs.items()}

        with torch.no_grad():
            query_embeddings = self.model(**query_inputs)

        return [q.cpu() for q in query_embeddings]

    def encode_images(
        self,
        images: List[Union[str, Image.Image]]
    ) -> List[torch.Tensor]:
        """
        Encode a list of images.

        Args:
            images: List of image paths or PIL Images

        Returns:
            List of image embeddings
        """
        image_objects = []
        for img in images:
            if isinstance(img, str):
                image_objects.append(Image.open(img).convert("RGB"))
            elif isinstance(img, Image.Image):
                image_objects.append(img)
            else:
                raise ValueError(f"Unsupported image type: {type(img)}")

        image_inputs = self.processor.process_images(image_objects)
        image_inputs = {k: v.to(self.device) for k, v in image_inputs.items()}

        with torch.no_grad():
            image_embeddings = self.model(**image_inputs)

        return [i.cpu() for i in image_embeddings]

    def compute_scores(
        self,
        query_embeddings: List[torch.Tensor],
        image_embeddings: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute similarity scores between queries and images.

        Args:
            query_embeddings: List of query embeddings
            image_embeddings: List of image embeddings

        Returns:
            Similarity scores matrix
        """
        return self.processor.score_multi_vector(query_embeddings, image_embeddings)

    def encode_and_score(
        self,
        queries: List[str],
        images: List[Union[str, Image.Image]]
    ):
        """
        Convenience method to encode queries and images and compute scores.

        Args:
            queries: List of query texts
            images: List of images (paths or PIL objects)

        Returns:
            Similarity scores between queries and images
        """
        query_embeddings = self.encode_queries(queries)
        image_embeddings = self.encode_images(images)
        return self.compute_scores(query_embeddings, image_embeddings)

