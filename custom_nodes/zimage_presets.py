"""
Z-Image Presets: Resolution Presets & Model/Sampler Preset Switch for ComfyUI.
"""

import folder_paths
import comfy.sd
import comfy.samplers
from nodes import EmptyLatentImage


RESOLUTION_PRESETS = [
    "1024x1024 ( 1:1 )",
    "1152x896 ( 9:7 )",
    "896x1152 ( 7:9 )",
    "1152x864 ( 4:3 )",
    "864x1152 ( 3:4 )",
    "1248x832 ( 3:2 )",
    "832x1248 ( 2:3 )",
    "1280x720 ( 16:9 )",
    "720x1280 ( 9:16 )",
    "1344x576 ( 21:9 )",
    "576x1344 ( 9:21 )",
]


class ZImageResolutionPreset:
    """Provides standard DiT / Z-Image aspect ratio resolution presets in a single dropdown."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "dimensions": (RESOLUTION_PRESETS, {"default": "864x1152 ( 3:4 )"}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 64}),
            }
        }

    RETURN_TYPES = ("LATENT", "INT", "INT")
    RETURN_NAMES = ("LATENT", "WIDTH", "HEIGHT")
    FUNCTION = "generate"
    CATEGORY = "latent/presets"
    DESCRIPTION = "Empty Latent Generator with pre-configured aspect ratio presets."

    def generate(self, dimensions, batch_size=1):
        res_str = dimensions.split()[0]
        width, height = map(int, res_str.split("x"))
        latent = EmptyLatentImage().generate(width, height, batch_size)[0]
        return (latent, width, height)


class ZImagePresetLoader:
    """Controller for Z-Image Turbo vs Base model loading and primary sampling configuration."""

    @classmethod
    def INPUT_TYPES(s):
        diff_models = folder_paths.get_filename_list("diffusion_models")
        turbo_def = "z_image_turbo_int8_convrot.safetensors" if "z_image_turbo_int8_convrot.safetensors" in diff_models else (diff_models[0] if diff_models else "")
        base_def = "z_image_int8_convrot.safetensors" if "z_image_int8_convrot.safetensors" in diff_models else (diff_models[0] if diff_models else "")

        return {
            "required": {
                "preset": (
                    [
                        "ZIT (Turbo - 8 Steps, CFG 1.0)",
                        "ZI (Base - 28 Steps, CFG 4.0)"
                    ],
                    {"default": "ZIT (Turbo - 8 Steps, CFG 1.0)"}
                ),
                "sampler": (comfy.samplers.KSampler.SAMPLERS, {"default": "euler_ancestral"}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": "beta"}),
                "turbo_model": (diff_models, {"default": turbo_def}),
                "base_model": (diff_models, {"default": base_def}),
            }
        }

    RETURN_TYPES = (
        "MODEL", "INT", "FLOAT",
        comfy.samplers.KSampler.SAMPLERS,
        comfy.samplers.KSampler.SCHEDULERS
    )
    RETURN_NAMES = (
        "MODEL", "STEPS", "CFG", "SAMPLER_NAME", "SCHEDULER"
    )
    FUNCTION = "load_preset"
    CATEGORY = "model/presets"
    DESCRIPTION = "Select between Z-Image Turbo and Base with synchronized sampling configuration."

    def load_preset(self, preset, sampler, scheduler, turbo_model, base_model):
        if "Turbo" in preset:
            model_name = turbo_model
            steps, cfg = 8, 1.0
        else:
            model_name = base_model
            steps, cfg = 28, 4.0

        unet_path = folder_paths.get_full_path_or_raise("diffusion_models", model_name)
        model = comfy.sd.load_diffusion_model(unet_path)
        return (model, steps, cfg, sampler, scheduler)


NODE_CLASS_MAPPINGS = {
    "ZImageResolutionPreset": ZImageResolutionPreset,
    "ZImagePresetLoader": ZImagePresetLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ZImageResolutionPreset": "Resolution Preset (Empty Latent)",
    "ZImagePresetLoader": "Z-Image Preset Switch (Turbo / Base)",
}
