"""Compatibility shims for Phi-3-Vision remote code on newer transformers."""

from __future__ import annotations


def patch_phi3_auto_image_processor_register() -> None:
    """Allow Phi-3-Vision remote code to import on newer transformers.

    The upstream Phi-3-Vision processor registers its image processor with a
    string key. Recent transformers versions expect a config class and crash
    when they access ``key.__module__``. The registration is not needed for our
    direct ``AutoProcessor.from_pretrained(..., trust_remote_code=True)`` path,
    so string-key registrations can be safely ignored.
    """
    try:
        from transformers.models.auto.image_processing_auto import AutoImageProcessor
    except Exception:
        return

    register = getattr(AutoImageProcessor, "register", None)
    if register is None or getattr(register, "_phi3_string_safe", False):
        return

    def _safe_register(config_class, image_processor_class=None, *args, **kwargs):
        if isinstance(config_class, str):
            return None
        return register(config_class, image_processor_class, *args, **kwargs)

    _safe_register._phi3_string_safe = True  # type: ignore[attr-defined]
    AutoImageProcessor.register = _safe_register

    try:
        from transformers.cache_utils import DynamicCache
    except Exception:
        return

    if not hasattr(DynamicCache, "from_legacy_cache"):
        @classmethod
        def _from_legacy_cache(cls, past_key_values=None):
            if isinstance(past_key_values, cls):
                return past_key_values
            return cls()

        DynamicCache.from_legacy_cache = _from_legacy_cache
