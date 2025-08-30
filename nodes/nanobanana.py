# gemini_image_edit_node_channel_last.py
# requirements: google-genai (or google.genai), pillow, numpy, torch
from nodes import IO
import traceback
from io import BytesIO
from PIL import Image
import numpy as np
import torch

# defensive import
try:
    from google import genai
except Exception:
    try:
        import genai
    except Exception:
        genai = None


class GeminiImageEditNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "prompt": ("STRING", {"default": "Edit the image as requested."}),
            },
            "optional": {
                "model_name": ("STRING", {"default": "gemini-2.5-flash-image-preview"}),
                "api_key": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("edited_image",)
    FUNCTION = "generate"
    CATEGORY = "Google Gemini"

    def __init__(self):
        pass

    def _to_pil(self, image):
        """Convert ComfyUI IMAGE (tensor / numpy / PIL) -> PIL.Image (RGB)."""
        # Torch tensor
        if isinstance(image, torch.Tensor):
            img = image.detach().cpu()
            # if batch dim present (1, C, H, W) or (1, H, W, C)
            if img.ndim == 4 and img.shape[0] == 1:
                img = img.squeeze(0)
            # channel-first (C,H,W) -> (H,W,C)
            if (
                img.ndim == 3
                and img.shape[0] in (1, 3, 4)
                and img.shape[0] < img.shape[1]
            ):
                img = img.permute(1, 2, 0)
            arr = img.numpy()
            # floats in [0..1] -> scale to 0..255
            if arr.dtype in (np.float32, np.float64) and arr.max() <= 1.0:
                arr = (arr * 255).clip(0, 255).astype(np.uint8)
            else:
                arr = arr.astype(np.uint8)
            # grayscale -> 3ch
            if arr.ndim == 2:
                arr = np.stack([arr] * 3, axis=-1)
            if arr.shape[2] == 1:
                arr = np.concatenate([arr, arr, arr], axis=2)
            return Image.fromarray(arr).convert("RGB")

        # Numpy array
        if isinstance(image, np.ndarray):
            arr = image
            # channel-first heuristic
            if (
                arr.ndim == 3
                and arr.shape[0] in (1, 3, 4)
                and arr.shape[0] < arr.shape[1]
            ):
                arr = np.transpose(arr, (1, 2, 0))
            if arr.dtype in (np.float32, np.float64) and arr.max() <= 1.0:
                arr = (arr * 255).clip(0, 255).astype(np.uint8)
            else:
                arr = arr.astype(np.uint8)
            if arr.ndim == 2:
                arr = np.stack([arr] * 3, axis=-1)
            if arr.shape[2] == 1:
                arr = np.concatenate([arr, arr, arr], axis=2)
            return Image.fromarray(arr).convert("RGB")

        # PIL
        if isinstance(image, Image.Image):
            return image.convert("RGB")

        raise TypeError(f"Unsupported image type: {type(image)}")

    def _pil_to_tensor_channel_last(self, pil_img):
        """
        Convert PIL.Image -> torch.Tensor (1, H, W, 3), float32 in [0,1].
        This matches your comment: "결과 이미지를 tensor로 변환 (범위 [0,1], 배치 차원 추가)".
        """
        arr = np.array(pil_img.convert("RGB")).astype(np.float32) / 255.0  # H,W,3
        tensor = torch.from_numpy(arr).unsqueeze(0)  # 1,H,W,3
        return tensor

    def generate(
        self, image, prompt, model_name="gemini-2.5-flash-image-preview", api_key=""
    ):
        if genai is None:
            raise RuntimeError(
                "google.genai (or genai) library not found. Install google-genai."
            )

        # convert input to PIL image
        try:
            pil_image = self._to_pil(image)
        except Exception as e:
            raise RuntimeError(f"Failed to convert input image to PIL: {e}")

        # init client (use env var if api_key empty)
        try:
            client = genai.Client(api_key=api_key) if api_key else genai.Client()
        except Exception as e:
            raise RuntimeError(f"Failed to create genai.Client: {e}")

        # call model (NO timeout argument)
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=[prompt, pil_image],
            )
        except Exception as e:
            tb = traceback.format_exc()
            raise RuntimeError(f"Model request failed: {e}\n{tb}")

        # extract image bytes from response
        img_bytes = None
        try:
            if getattr(response, "candidates", None):
                for cand in response.candidates:
                    content = getattr(cand, "content", None)
                    if not content:
                        continue
                    parts = getattr(content, "parts", None) or []
                    for part in parts:
                        inline = getattr(part, "inline_data", None)
                        if inline is not None and getattr(inline, "data", None):
                            img_bytes = bytes(inline.data)
                            break
                    if img_bytes:
                        break
        except Exception:
            img_bytes = None

        if img_bytes is None:
            raise RuntimeError("No image data found in model response.")

        # open PIL image from bytes
        try:
            out_pil = Image.open(BytesIO(img_bytes)).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Failed to open returned image bytes: {e}")

        # convert to torch tensor in shape (1,H,W,3), float32 [0,1]
        out_tensor = self._pil_to_tensor_channel_last(out_pil)

        return (out_tensor,)
