try:
    from .nodes import BlueMagpieModelLoader, BlueMagpieTTS
except ImportError:
    from nodes import BlueMagpieModelLoader, BlueMagpieTTS


NODE_CLASS_MAPPINGS = {
    "BlueMagpieModelLoader": BlueMagpieModelLoader,
    "BlueMagpieTTS": BlueMagpieTTS,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BlueMagpieModelLoader": "BlueMagpie Model Loader",
    "BlueMagpieTTS": "BlueMagpie TTS",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

__version__ = "0.1.0"
