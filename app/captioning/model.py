"""Qwen2-VL captioner: loads the model once and generates captions with token/latency metadata."""

import time
from typing import Any, Dict, Optional

from PIL import Image

from app.config import settings


class Captioner:
    """
    Wraps the Qwen2-VL vision-language model (a navigation-assistant fine-tune).

    The model is loaded lazily via load() (called once at app startup) because it is several GB.
    generate() runs the chat-style image+instruction prompt and returns the reply plus token counts
    and latency.
    """

    def __init__(self, model_name: Optional[str] = None, max_new_tokens: Optional[int] = None) -> None:
        self.model_name = model_name or settings.MODEL_NAME
        self.max_new_tokens = max_new_tokens or settings.MAX_NEW_TOKENS
        self.default_prompt = settings.CAPTION_PROMPT
        self.device = "cpu"
        self.processor = None
        self.model = None

    @property
    def is_loaded(self) -> bool:
        return self.model is not None and self.processor is not None

    def load(self) -> None:
        """Loads the processor and model weights. Downloads several GB on first run."""
        import torch
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # bf16 on GPU: native on Blackwell (RTX PRO 6000) and Ampere+; fp32 on CPU.
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32

        # FlashAttention 2 cuts attention time on the vision-token-heavy prefill.
        # Falls back to PyTorch SDPA (no extra dependency) if flash-attn isn't installed.
        if self.device == "cuda":
            try:
                import flash_attn  # noqa: F401
                attn_implementation = "flash_attention_2"
            except ImportError:
                attn_implementation = "sdpa"
        else:
            attn_implementation = "eager"

        print(f"Loading {self.model_name} on {self.device} ({dtype}, attn={attn_implementation})...")
        # Cap vision tokens: input tokens (and thus prefill latency) scale with image
        # resolution. The default max_pixels allows much larger images than captioning
        # needs. Tune max_pixels if captions lose small-object detail.
        self.processor = AutoProcessor.from_pretrained(
            self.model_name,
            min_pixels=256 * 28 * 28,
            max_pixels=768 * 28 * 28,
        )
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=dtype,
            attn_implementation=attn_implementation,
        ).to(self.device)
        self.model.eval()
        print("Model loaded.")

    def generate(self, image: Image.Image, prompt: Optional[str] = None) -> Dict[str, Any]:
        """
        Generates a caption for the image.

        Returns a dict with: caption, input_tokens, output_tokens, total_tokens, latency_ms.
        input_tokens includes the expanded image (vision) tokens, so it scales with image size.
        """
        if not self.is_loaded:
            raise RuntimeError("Model is not loaded.")

        import torch

        if image.mode != "RGB":
            image = image.convert("RGB")

        instruction = prompt or self.default_prompt
        # Mirror the model author's reference usage: the instruction is a SYSTEM message and the user
        # turn carries only the image. The checkpoint was developed/validated this way; putting the
        # instruction in the user turn (as we did before) diverges from that and gives weaker output.
        messages = [
            {"role": "system", "content": instruction},
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                ],
            },
        ]
        # Build the chat-formatted text (with the image placeholder), then bind the actual image.
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[text], images=[image], padding=True, return_tensors="pt"
        ).to(self.device)

        input_len = int(inputs.input_ids.shape[-1])

        start = time.perf_counter()
        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,  # deterministic greedy decode, matching the author's reference usage
                use_cache=True,
                pad_token_id=self.processor.tokenizer.eos_token_id,
            )
        latency_ms = (time.perf_counter() - start) * 1000

        # Drop the prompt tokens; decode only what the model generated.
        trimmed = generated_ids[:, input_len:]
        caption = self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()
        output_tokens = int(trimmed.shape[-1])

        return {
            "caption": caption,
            "input_tokens": input_len,
            "output_tokens": output_tokens,
            "total_tokens": input_len + output_tokens,
            "latency_ms": round(latency_ms, 2),
        }