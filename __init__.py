from .nodes import *

from .nodes.nanobanana import GeminiImageEditNode

NODE_CLASS_MAPPINGS = {
    "NanoBanana API🍌": GeminiImageEditNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NanoBanana API🍌": "NanoBanana API🍌",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
