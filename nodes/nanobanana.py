from nodes import IO  # (ComfyUI 런타임에 따라 미사용일 수 있음)
import traceback
from io import BytesIO
from PIL import Image
import numpy as np
import torch

# Try import genai from the two common namespaces
genai = None
try:
    # preferred official package namespace
    from google import genai as _genai
    from google.genai.types import GenerateContentConfig, Modality, Part
    genai = _genai
except Exception:
    try:
        import genai as _genai
        from genai.types import GenerateContentConfig, Modality, Part
        genai = _genai
    except Exception:
        genai = None


class GeminiImageEditNode:
    """
    ComfyUI 노드: 입력 이미지(+옵션 레퍼런스 이미지들)와 프롬프트를
    Google Gemini 이미지 생성/편집 모델로 보내 결과 이미지를 반환합니다.

    - 필수 입력: image, prompt
    - 선택 입력: reference_image_1..4 (최대 4장 레퍼런스)
    - 모델 기본값: 'gemini-2.5-flash-image-preview' (이미지 응답 지원)
    - 반환: torch.Tensor, shape (1, H, W, 3), float32 in [0,1]
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "prompt": ("STRING", {"default": "Edit the image as requested."}),
                "model_name": ("STRING", {"default": "gemini-2.5-flash-image-preview"}),
                "api_key": ("STRING", {"default": ""}),
            },
            "optional": {
                "reference_image_1": ("IMAGE", {"forceInput": False}),
                "reference_image_2": ("IMAGE", {"forceInput": False}),
                "reference_image_3": ("IMAGE", {"forceInput": False}),
                "reference_image_4": ("IMAGE", {"forceInput": False}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("edited_image",)
    FUNCTION = "generate"
    CATEGORY = "Google Gemini"

    def __init__(self):
        pass

    # -----------------------
    # Utility converters
    # -----------------------
    def _to_pil(self, image):
        """Convert ComfyUI IMAGE (torch.Tensor / numpy.ndarray / PIL.Image) -> PIL.Image (RGB)."""
        # Torch tensor
        if isinstance(image, torch.Tensor):
            img = image.detach().cpu()
            # if batch dim present (1, C, H, W) or (1, H, W, C)
            if img.ndim == 4 and img.shape[0] == 1:
                img = img.squeeze(0)
            # channel-first (C,H,W) -> (H,W,C)
            if img.ndim == 3 and img.shape[0] in (1, 3, 4) and img.shape[0] < img.shape[1]:
                img = img.permute(1, 2, 0)
            arr = img.numpy()
            # floats in [0..1] -> scale to 0..255
            if arr.dtype in (np.float32, np.float64):
                # 보수적으로 스케일 판단: 값이 1.0 이하이면 [0..1]로 간주
                if np.nanmax(arr) <= 1.0:
                    arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
                else:
                    arr = np.clip(arr, 0, 255).astype(np.uint8)
            else:
                arr = arr.astype(np.uint8)
            # grayscale -> 3ch
            if arr.ndim == 2:
                arr = np.stack([arr] * 3, axis=-1)
            if arr.ndim == 3 and arr.shape[2] == 1:
                arr = np.concatenate([arr, arr, arr], axis=2)
            return Image.fromarray(arr).convert("RGB")

        # Numpy array
        if isinstance(image, np.ndarray):
            arr = image
            # channel-first heuristic
            if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[0] < arr.shape[1]:
                arr = np.transpose(arr, (1, 2, 0))
            if arr.dtype in (np.float32, np.float64):
                if np.nanmax(arr) <= 1.0:
                    arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
                else:
                    arr = np.clip(arr, 0, 255).astype(np.uint8)
            else:
                arr = arr.astype(np.uint8)
            if arr.ndim == 2:
                arr = np.stack([arr] * 3, axis=-1)
            if arr.ndim == 3 and arr.shape[2] == 1:
                arr = np.concatenate([arr, arr, arr], axis=2)
            return Image.fromarray(arr).convert("RGB")

        # PIL
        if isinstance(image, Image.Image):
            return image.convert("RGB")

        raise TypeError(f"Unsupported image type: {type(image)}")

    def _pil_to_tensor_channel_last(self, pil_img):
        """
        Convert PIL.Image -> torch.Tensor (1, H, W, 3), float32 in [0,1].
        """
        arr = np.array(pil_img.convert("RGB")).astype(np.float32) / 255.0  # H,W,3
        tensor = torch.from_numpy(arr).unsqueeze(0)  # 1,H,W,3
        return tensor

    # -----------------------
    # Core
    # -----------------------
    def generate(
        self,
        image,
        prompt,
        model_name="gemini-2.5-flash-image-preview",
        api_key="",
        reference_image_1=None,
        reference_image_2=None,
        reference_image_3=None,
        reference_image_4=None,
    ):
        """
        Gemini 모델로 멀티이미지(기준+레퍼런스들) 편집/생성을 요청해
        최종 이미지를 반환합니다.
        """
        if genai is None:
            raise RuntimeError(
                "google-genai(또는 google.genai) 라이브러리를 찾을 수 없습니다. "
                "pip install google-genai 로 설치해 주세요."
            )

        # 1) 필수 입력 이미지 -> PIL
        try:
            pil_image = self._to_pil(image)
        except Exception as e:
            raise RuntimeError(f"Failed to convert input image to PIL: {e}")

        # 2) PIL -> PNG bytes
        try:
            buf = BytesIO()
            pil_image.save(buf, format="PNG")
            image_bytes = buf.getvalue()
        except Exception as e:
            raise RuntimeError(f"Failed to serialize PIL image to bytes: {e}")

        # 3) 클라이언트 생성 (api_key 비었으면 환경변수 GOOGLE_API_KEY 사용)
        try:
            client = genai.Client(api_key=api_key) if api_key else genai.Client()
        except Exception as e:
            raise RuntimeError(f"Failed to create genai.Client: {e}")

        # 4) 메인 이미지 Part
        try:
            base_part = Part.from_bytes(data=image_bytes, mime_type="image/png")
        except Exception as e:
            tb = traceback.format_exc()
            raise RuntimeError(
                f"Failed to create image Part for request. Ensure SDK exposes Part.from_bytes.\n{e}\n{tb}"
            )

        # 5) 레퍼런스 이미지들(옵션) -> Part 리스트
        ref_images = [
            reference_image_1,
            reference_image_2,
            reference_image_3,
            reference_image_4,
        ]
        ref_parts = []
        for idx, ref in enumerate(ref_images, start=1):
            if ref is None:
                continue
            try:
                pil_ref = self._to_pil(ref)
                rbuf = BytesIO()
                pil_ref.save(rbuf, format="PNG")
                ref_parts.append(Part.from_bytes(data=rbuf.getvalue(), mime_type="image/png"))
            except Exception as e:
                raise RuntimeError(f"Failed to process reference_image_{idx}: {e}")

        # 6) 요청 contents 구성: [프롬프트, 기준 이미지, 레퍼런스1..4]
        contents = [prompt, base_part] + ref_parts

        # 7) 모델 호출 (이미지 응답 모드)
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=contents,
                config=GenerateContentConfig(
                    response_modalities=[Modality.IMAGE]
                ),
            )
        except Exception as e:
            tb = traceback.format_exc()
            raise RuntimeError(f"Model request failed: {e}\n{tb}")

        # 8) 응답에서 이미지 바이트 추출 (첫 번째 inline_data)
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
                            data_field = inline.data
                            if isinstance(data_field, (bytes, bytearray)):
                                img_bytes = bytes(data_field)
                            else:
                                # 일부 런타임에서 data가 memoryview/iterable인 경우
                                img_bytes = bytes(list(data_field))
                            break
                    if img_bytes:
                        break
        except Exception:
            img_bytes = None

        # 9) 이미지가 안 왔으면 텍스트 스니펫 첨부하여 에러
        if img_bytes is None:
            text_preview = None
            try:
                textual_parts = []
                if getattr(response, "candidates", None):
                    for cand in response.candidates:
                        content = getattr(cand, "content", None)
                        if not content:
                            continue
                        for part in getattr(content, "parts", []) or []:
                            t = getattr(part, "text", None)
                            if t:
                                textual_parts.append(t)
                if textual_parts:
                    text_preview = "\n".join(textual_parts[:3])
            except Exception:
                text_preview = None

            msg = "No image data found in model response."
            if text_preview:
                snippet = text_preview.strip()
                if len(snippet) > 800:
                    snippet = snippet[:800] + "..."
                msg += f" Model returned text instead:\n{snippet}"
            raise RuntimeError(msg)

        # 10) 바이트 -> PIL -> 텐서(1,H,W,3)
        try:
            out_pil = Image.open(BytesIO(img_bytes)).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Failed to open returned image bytes: {e}")

        out_tensor = self._pil_to_tensor_channel_last(out_pil)
        return (out_tensor,)
