"""
Telegram Bot Sidecar Client for ComfyUI.
Provides interactive workflow controller with 100% exposed parameters,
live progress tracking, authorization security, and dual media delivery.
"""

import os
import sys
import json
import uuid
import random
import asyncio
from pathlib import Path

import aiohttp
from PIL import Image

from telegram import (
    Update,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
)
from telegram.constants import ParseMode
from telegram.request import HTTPXRequest
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    ContextTypes,
    filters,
)

# Fix Windows console emoji encoding
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

CONFIG_PATH = Path("scripts/config.telegram.toml")
TEMPLATE_PATH = Path("scripts/z_image_api_template.json")

user_sessions = {}

# Map high-level aspect labels to 6x-downscaled base dimensions for EmptySD3LatentImage
BASE_DIMS_MAP = {
    "864x1152 ( 3:4 )": (224, 288),
    "1024x1024 ( 1:1 )": (256, 256),
    "720x1280 ( 9:16 )": (176, 320),
    "1280x720 ( 16:9 )": (320, 176),
    "1152x864 ( 4:3 )": (288, 224),
    "832x1248 ( 2:3 )": (208, 312),
    "1248x832 ( 3:2 )": (312, 208),
    "896x1152 ( 7:9 )": (224, 288),
    "1152x896 ( 9:7 )": (288, 224),
    "1344x576 ( 21:9 )": (384, 160),
    "576x1344 ( 9:21 )": (160, 384),
}

ALL_ASPECT_RATIOS = [
    ("3:4 Portrait (864x1152)", "864x1152 ( 3:4 )"),
    ("1:1 Square (1024x1024)", "1024x1024 ( 1:1 )"),
    ("9:16 Story (720x1280)", "720x1280 ( 9:16 )"),
    ("16:9 Landscape (1280x720)", "1280x720 ( 16:9 )"),
    ("4:3 Classic (1152x864)", "1152x864 ( 4:3 )"),
    ("2:3 Photo (832x1248)", "832x1248 ( 2:3 )"),
    ("3:2 Photo (1248x832)", "1248x832 ( 3:2 )"),
    ("7:9 Book (896x1152)", "896x1152 ( 7:9 )"),
    ("9:7 Book (1152x896)", "1152x896 ( 9:7 )"),
    ("21:9 Cinema (1344x576)", "1344x576 ( 21:9 )"),
    ("9:21 Ultra-Tall (576x1344)", "576x1344 ( 9:21 )"),
]

SAMPLERS_LIST = [
    "euler_ancestral", "dpmpp_sde", "euler", "res_multistep",
    "dpmpp_2m_sde", "dpmpp_2m", "heun", "lcm"
]

SCHEDULERS_LIST = [
    "beta", "simple", "ddim_uniform", "sgm_uniform", "karras", "normal"
]


def load_config() -> dict:
    if not CONFIG_PATH.exists():
        print(f"Error: Konfigurasi '{CONFIG_PATH}' tidak ditemukan.")
        print("Silakan salin 'scripts/config.telegram.example.toml' ke 'scripts/config.telegram.toml' lalu isi bot token.")
        sys.exit(1)

    try:
        import tomllib
    except ImportError:
        import tomli as tomllib

    with open(CONFIG_PATH, "rb") as f:
        return tomllib.load(f)


def is_authorized(user_id: int, allowed_ids: list[int]) -> bool:
    return user_id in allowed_ids


def load_default_preferences() -> dict:
    defaults = {
        "preset": "ZIT (Turbo - 8 Steps, CFG 1.0)",
        "dimensions": "864x1152 ( 3:4 )",
        "custom_width": 224,
        "custom_height": 288,
        "batch_size": 1,
        "pipeline_mode": "direct",
        "lora_strength": 0.8,
        "lora_clip_strength": 0.8,
        "trigger_prefix": True,
        "seed_mode": "random",
        "fixed_seed": 351971767740904,
        "filename_prefix": "zi",
        # AuraFlow Sampling
        "auraflow_shift": 6.0,
        # Pass 1
        "steps_pass1": 8,
        "cfg_pass1": 1.0,
        "sampler_pass1": "euler_ancestral",
        "scheduler": "beta",
        # Pass 2
        "steps_pass2": 8,
        "cfg_pass2": 1.0,
        "sampler_pass2": "dpmpp_sde",
        "scheduler_pass2": "beta",
        "denoise_pass2": 0.38,
        "latent_upscale_method": "bislerp",
        "latent_scale_by": 6.0,
        # Pass 3 SeedVR2
        "seedvr2_scale_by": 2.0,
        "seedvr2_direct_scale": 3.0,
        "seedvr2_upscale_method": "lanczos",
        "seedvr2_color_correction": "lab",
        "seedvr2_steps": 1,
        "seedvr2_cfg": 1.0,
        "seedvr2_sampler": "dpmpp_sde",
        "seedvr2_scheduler": "beta",
        # Tiled VAE
        "vae_tile_size": 512,
        "vae_overlap": 64,
    }

    if TEMPLATE_PATH.exists():
        try:
            with open(TEMPLATE_PATH, "r", encoding="utf-8") as f:
                graph = json.load(f)

            if "13" in graph:
                unet_name = graph["13"].get("inputs", {}).get("unet_name", "")
                if "turbo" in unet_name.lower():
                    defaults["preset"] = "ZIT (Turbo - 8 Steps, CFG 1.0)"
                elif unet_name:
                    defaults["preset"] = "ZI (Base - 28 Steps, CFG 4.0)"

            if "15" in graph:
                defaults["auraflow_shift"] = float(graph["15"].get("inputs", {}).get("shift", defaults["auraflow_shift"]))

            if "69" in graph:
                in69 = graph["69"].get("inputs", {})
                defaults["custom_width"] = int(in69.get("width", defaults["custom_width"]))
                defaults["custom_height"] = int(in69.get("height", defaults["custom_height"]))
                defaults["batch_size"] = int(in69.get("batch_size", defaults["batch_size"]))

            if "7" in graph:
                in7 = graph["7"].get("inputs", {})
                defaults["steps_pass1"] = int(in7.get("steps", defaults["steps_pass1"]))
                defaults["cfg_pass1"] = float(in7.get("cfg", defaults["cfg_pass1"]))
                defaults["sampler_pass1"] = in7.get("sampler_name", defaults["sampler_pass1"])
                defaults["scheduler"] = in7.get("scheduler", defaults["scheduler"])

            if "70" in graph:
                in70 = graph["70"].get("inputs", {})
                defaults["latent_upscale_method"] = in70.get("upscale_method", defaults["latent_upscale_method"])
                defaults["latent_scale_by"] = float(in70.get("scale_by", defaults["latent_scale_by"]))

            if "29" in graph:
                in29 = graph["29"].get("inputs", {})
                defaults["steps_pass2"] = int(in29.get("steps", defaults["steps_pass2"]))
                defaults["cfg_pass2"] = float(in29.get("cfg", defaults["cfg_pass2"]))
                defaults["sampler_pass2"] = in29.get("sampler_name", defaults["sampler_pass2"])
                defaults["scheduler_pass2"] = in29.get("scheduler", defaults["scheduler_pass2"])
                if "denoise" in in29 and isinstance(in29["denoise"], (int, float)):
                    defaults["denoise_pass2"] = float(in29["denoise"])

            if "31" in graph:
                defaults["vae_tile_size"] = int(graph["31"].get("inputs", {}).get("tile_size", defaults["vae_tile_size"]))
                defaults["vae_overlap"] = int(graph["31"].get("inputs", {}).get("overlap", defaults["vae_overlap"]))

            if "52" in graph:
                in52 = graph["52"].get("inputs", {})
                defaults["seedvr2_upscale_method"] = in52.get("upscale_method", defaults["seedvr2_upscale_method"])
                defaults["seedvr2_scale_by"] = float(in52.get("scale_by", defaults["seedvr2_scale_by"]))

            if "56" in graph:
                in56 = graph["56"].get("inputs", {})
                defaults["seedvr2_steps"] = int(in56.get("steps", defaults["seedvr2_steps"]))
                defaults["seedvr2_cfg"] = float(in56.get("cfg", defaults["seedvr2_cfg"]))
                defaults["seedvr2_sampler"] = in56.get("sampler_name", defaults["seedvr2_sampler"])
                defaults["seedvr2_scheduler"] = in56.get("scheduler", defaults["seedvr2_scheduler"])

            if "58" in graph:
                defaults["seedvr2_color_correction"] = graph["58"].get("inputs", {}).get("color_correction_method", defaults["seedvr2_color_correction"])

            if "67" in graph:
                lora_cfg = graph["67"].get("inputs", {}).get("lora_1", {})
                if isinstance(lora_cfg, dict) and "strength" in lora_cfg:
                    defaults["lora_strength"] = float(lora_cfg["strength"])
                    defaults["lora_clip_strength"] = float(lora_cfg.get("strengthTwo", defaults["lora_strength"]))

            if "68" in graph:
                seed_val = int(graph["68"].get("inputs", {}).get("seed", -1))
                if seed_val == -1:
                    defaults["seed_mode"] = "random"
                else:
                    defaults["seed_mode"] = "fixed"
                    defaults["fixed_seed"] = seed_val

        except Exception as e:
            print(f"Warning: Gagal membaca template '{TEMPLATE_PATH}': {e}")

    return defaults


def get_user_pref(user_id: int) -> dict:
    if user_id not in user_sessions:
        user_sessions[user_id] = load_default_preferences()
    return user_sessions[user_id]


async def check_comfy_online(server_url: str) -> bool:
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{server_url}/system_stats",
                timeout=aiohttp.ClientTimeout(total=2),
            ) as resp:
                return resp.status == 200
    except Exception:
        return False


# ==========================================
# KEYBOARD BUILDERS
# ==========================================

def build_main_settings_keyboard(user_id: int) -> InlineKeyboardMarkup:
    pref = get_user_pref(user_id)
    is_turbo = "Turbo" in pref["preset"]
    pipe_labels = {"direct": "💎 Direct 4K", "full": "🏛️ Full 4K", "hires": "⚡ Hi-Res", "draft": "🚀 Draft"}

    mode_label = "Turbo (8s)" if is_turbo else "Base (28s)"
    pipe_label = pipe_labels.get(pref["pipeline_mode"], "Direct 4K")
    seed_label = "Random" if pref["seed_mode"] == "random" else f"🔒 {pref['fixed_seed']}"
    aspect_label = pref["dimensions"].split()[0]
    prefix_label = "ON" if pref.get("trigger_prefix", True) else "OFF"

    buttons = [
        [
            InlineKeyboardButton(f"⚡ Mode: {mode_label}", callback_data="toggle_mode"),
            InlineKeyboardButton(f"💎 Alur: {pipe_label}", callback_data="menu_pipeline"),
        ],
        [
            InlineKeyboardButton(f"📐 Rasio: {aspect_label}", callback_data="menu_aspect"),
            InlineKeyboardButton(f"🌸 LoRA: {pref['lora_strength']}", callback_data="menu_lora"),
        ],
        [
            InlineKeyboardButton("🎛️ Pass 1 Controls", callback_data="menu_pass1"),
            InlineKeyboardButton("🔬 Pass 2 Controls", callback_data="menu_pass2"),
        ],
        [
            InlineKeyboardButton("🧬 SeedVR2 Controls", callback_data="menu_seedvr2"),
            InlineKeyboardButton(f"🎲 Seed: {seed_label}", callback_data="menu_seed"),
        ],
        [
            InlineKeyboardButton(f"🌊 Shift: {pref['auraflow_shift']}", callback_data="menu_shift"),
            InlineKeyboardButton(f"📦 Batch: {pref['batch_size']}", callback_data="menu_batch"),
        ],
        [
            InlineKeyboardButton(f"🧩 VAE Tile: {pref['vae_tile_size']}", callback_data="menu_tile"),
            InlineKeyboardButton(f"🔤 Trigger: {prefix_label}", callback_data="toggle_trigger"),
        ],
        [
            InlineKeyboardButton("🔄 Reset Default", callback_data="reset_settings"),
            InlineKeyboardButton("❌ Tutup Menu", callback_data="close_menu"),
        ],
    ]
    return InlineKeyboardMarkup(buttons)


def build_pipeline_keyboard(user_id: int) -> InlineKeyboardMarkup:
    pref = get_user_pref(user_id)
    cur = pref["pipeline_mode"]

    buttons = [
        [InlineKeyboardButton(f"{'✅ ' if cur == 'direct' else ''}💎 Direct 4K (Pass 1 ➔ SeedVR2 - Alami & Cepat)", callback_data="set_pipe:direct")],
        [InlineKeyboardButton(f"{'✅ ' if cur == 'full' else ''}🏛️ Full Studio 4K (Pass 1 + 2 + SeedVR2)", callback_data="set_pipe:full")],
        [InlineKeyboardButton(f"{'✅ ' if cur == 'hires' else ''}⚡ Fast Hi-Res (Pass 1 + 2 Tanpa SeedVR2)", callback_data="set_pipe:hires")],
        [InlineKeyboardButton(f"{'✅ ' if cur == 'draft' else ''}🚀 Draft Cepat (Pass 1 Base Saja)", callback_data="set_pipe:draft")],
        [InlineKeyboardButton("🔙 Kembali ke Menu Utama", callback_data="menu_main")],
    ]
    return InlineKeyboardMarkup(buttons)


def build_aspect_keyboard(user_id: int) -> InlineKeyboardMarkup:
    pref = get_user_pref(user_id)
    cur = pref["dimensions"]

    buttons = []
    row = []
    for label, dim in ALL_ASPECT_RATIOS:
        active = "✅ " if dim == cur else ""
        short_name = " ".join(label.split()[:2])
        row.append(InlineKeyboardButton(f"{active}{short_name}", callback_data=f"set_aspect:{dim}"))
        if len(row) == 2:
            buttons.append(row)
            row = []
    if row:
        buttons.append(row)

    buttons.append([InlineKeyboardButton("🔙 Kembali ke Menu Utama", callback_data="menu_main")])
    return InlineKeyboardMarkup(buttons)


def build_lora_keyboard(user_id: int) -> InlineKeyboardMarkup:
    pref = get_user_pref(user_id)
    cur = pref["lora_strength"]
    strengths = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5]

    row1 = [InlineKeyboardButton(f"{'✅ ' if s == cur else ''}{s}{' (Off)' if s == 0 else ''}", callback_data=f"set_lora:{s}") for s in strengths[:4]]
    row2 = [InlineKeyboardButton(f"{'✅ ' if s == cur else ''}{s}", callback_data=f"set_lora:{s}") for s in strengths[4:]]
    return InlineKeyboardMarkup([row1, row2, [InlineKeyboardButton("🔙 Kembali ke Menu Utama", callback_data="menu_main")]])


def build_seed_keyboard(user_id: int) -> InlineKeyboardMarkup:
    pref = get_user_pref(user_id)
    mode = pref["seed_mode"]

    buttons = [
        [
            InlineKeyboardButton(f"{'✅ ' if mode == 'random' else ''}🎲 Acak Tiap Generate", callback_data="set_seed_mode:random"),
            InlineKeyboardButton(f"{'✅ ' if mode == 'fixed' else ''}🔒 Kunci Seed Ini", callback_data="set_seed_mode:fixed"),
        ],
        [InlineKeyboardButton("🎲 Roll Seed Baru Acak", callback_data="set_seed_roll")],
        [InlineKeyboardButton("🔙 Kembali ke Menu Utama", callback_data="menu_main")],
    ]
    return InlineKeyboardMarkup(buttons)


# ==========================================
# PASS 1 CONTROLLER SUB-MENUS
# ==========================================

def build_pass1_keyboard(user_id: int) -> InlineKeyboardMarkup:
    pref = get_user_pref(user_id)
    buttons = [
        [
            InlineKeyboardButton(f"Steps: {pref['steps_pass1']}", callback_data="menu_p1_steps"),
            InlineKeyboardButton(f"CFG: {pref['cfg_pass1']}", callback_data="menu_p1_cfg"),
        ],
        [
            InlineKeyboardButton(f"Sampler: {pref['sampler_pass1'][:10]}", callback_data="menu_p1_sam"),
            InlineKeyboardButton(f"Scheduler: {pref['scheduler']}", callback_data="menu_p1_sched"),
        ],
        [InlineKeyboardButton("🔙 Kembali ke Menu Utama", callback_data="menu_main")],
    ]
    return InlineKeyboardMarkup(buttons)


def build_p1_steps_keyboard(user_id: int) -> InlineKeyboardMarkup:
    pref = get_user_pref(user_id)
    cur = pref["steps_pass1"]
    steps = [4, 6, 8, 10, 12, 16, 20, 28]
    r1 = [InlineKeyboardButton(f"{'✅ ' if s == cur else ''}{s}", callback_data=f"set_p1_steps:{s}") for s in steps[:4]]
    r2 = [InlineKeyboardButton(f"{'✅ ' if s == cur else ''}{s}", callback_data=f"set_p1_steps:{s}") for s in steps[4:]]
    return InlineKeyboardMarkup([r1, r2, [InlineKeyboardButton("🔙 Kembali ke Pass 1", callback_data="menu_pass1")]])


def build_p1_cfg_keyboard(user_id: int) -> InlineKeyboardMarkup:
    pref = get_user_pref(user_id)
    cur = pref["cfg_pass1"]
    cfgs = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0]
    r1 = [InlineKeyboardButton(f"{'✅ ' if c == cur else ''}{c}", callback_data=f"set_p1_cfg:{c}") for c in cfgs[:3]]
    r2 = [InlineKeyboardButton(f"{'✅ ' if c == cur else ''}{c}", callback_data=f"set_p1_cfg:{c}") for c in cfgs[3:]]
    return InlineKeyboardMarkup([r1, r2, [InlineKeyboardButton("🔙 Kembali ke Pass 1", callback_data="menu_pass1")]])


# ==========================================
# PASS 2 CONTROLLER SUB-MENUS
# ==========================================

def build_pass2_keyboard(user_id: int) -> InlineKeyboardMarkup:
    pref = get_user_pref(user_id)
    buttons = [
        [
            InlineKeyboardButton(f"Denoise: {pref['denoise_pass2']}", callback_data="menu_p2_denoise"),
            InlineKeyboardButton(f"Steps: {pref['steps_pass2']}", callback_data="menu_p2_steps"),
        ],
        [
            InlineKeyboardButton(f"CFG: {pref['cfg_pass2']}", callback_data="menu_p2_cfg"),
            InlineKeyboardButton(f"Scale: {pref['latent_scale_by']}x", callback_data="menu_p2_scale"),
        ],
        [
            InlineKeyboardButton(f"Method: {pref['latent_upscale_method']}", callback_data="menu_p2_method"),
            InlineKeyboardButton(f"Sampler: {pref['sampler_pass2'][:10]}", callback_data="menu_p2_sam"),
        ],
        [
            InlineKeyboardButton(f"Scheduler: {pref.get('scheduler_pass2', pref['scheduler'])}", callback_data="menu_p2_sched"),
        ],
        [InlineKeyboardButton("🔙 Kembali ke Menu Utama", callback_data="menu_main")],
    ]
    return InlineKeyboardMarkup(buttons)


def build_p2_denoise_keyboard(user_id: int) -> InlineKeyboardMarkup:
    pref = get_user_pref(user_id)
    cur = pref["denoise_pass2"]
    values = [0.25, 0.28, 0.32, 0.35, 0.38, 0.42, 0.45, 0.50]
    r1 = [InlineKeyboardButton(f"{'✅ ' if v == cur else ''}{v}", callback_data=f"set_p2_denoise:{v}") for v in values[:4]]
    r2 = [InlineKeyboardButton(f"{'✅ ' if v == cur else ''}{v}", callback_data=f"set_p2_denoise:{v}") for v in values[4:]]
    return InlineKeyboardMarkup([r1, r2, [InlineKeyboardButton("🔙 Kembali ke Pass 2", callback_data="menu_pass2")]])


def build_p2_steps_keyboard(user_id: int) -> InlineKeyboardMarkup:
    pref = get_user_pref(user_id)
    cur = pref["steps_pass2"]
    steps = [4, 6, 8, 10, 12, 15, 20]
    r1 = [InlineKeyboardButton(f"{'✅ ' if s == cur else ''}{s}", callback_data=f"set_p2_steps:{s}") for s in steps[:4]]
    r2 = [InlineKeyboardButton(f"{'✅ ' if s == cur else ''}{s}", callback_data=f"set_p2_steps:{s}") for s in steps[4:]]
    return InlineKeyboardMarkup([r1, r2, [InlineKeyboardButton("🔙 Kembali ke Pass 2", callback_data="menu_pass2")]])


def build_p2_cfg_keyboard(user_id: int) -> InlineKeyboardMarkup:
    pref = get_user_pref(user_id)
    cur = pref["cfg_pass2"]
    cfgs = [1.0, 1.5, 2.0, 3.0, 4.0]
    row = [InlineKeyboardButton(f"{'✅ ' if c == cur else ''}{c}", callback_data=f"set_p2_cfg:{c}") for c in cfgs]
    return InlineKeyboardMarkup([row, [InlineKeyboardButton("🔙 Kembali ke Pass 2", callback_data="menu_pass2")]])


def build_p2_scale_keyboard(user_id: int) -> InlineKeyboardMarkup:
    pref = get_user_pref(user_id)
    cur = pref["latent_scale_by"]
    scales = [4.0, 5.0, 6.0, 8.0]
    row = [InlineKeyboardButton(f"{'✅ ' if s == cur else ''}{s}x", callback_data=f"set_p2_scale:{s}") for s in scales]
    return InlineKeyboardMarkup([row, [InlineKeyboardButton("🔙 Kembali ke Pass 2", callback_data="menu_pass2")]])


def build_p2_method_keyboard(user_id: int) -> InlineKeyboardMarkup:
    pref = get_user_pref(user_id)
    cur = pref["latent_upscale_method"]
    methods = ["bislerp", "bicubic", "bilinear", "nearest-exact"]
    buttons = [[InlineKeyboardButton(f"{'✅ ' if m == cur else ''}{m}", callback_data=f"set_p2_method:{m}")] for m in methods]
    buttons.append([InlineKeyboardButton("🔙 Kembali ke Pass 2", callback_data="menu_pass2")])
    return InlineKeyboardMarkup(buttons)


# ==========================================
# PASS 3 SEEDVR2 CONTROLLER SUB-MENUS
# ==========================================

def build_seedvr2_keyboard(user_id: int) -> InlineKeyboardMarkup:
    pref = get_user_pref(user_id)
    buttons = [
        [
            InlineKeyboardButton(f"Direct Scale: {pref.get('seedvr2_direct_scale', 3.0)}x", callback_data="menu_svr_dscale"),
            InlineKeyboardButton(f"Pass 2 Scale: {pref['seedvr2_scale_by']}x", callback_data="menu_svr_scale"),
        ],
        [
            InlineKeyboardButton(f"Upscale: {pref['seedvr2_upscale_method']}", callback_data="menu_svr_method"),
            InlineKeyboardButton(f"Color: {pref['seedvr2_color_correction']}", callback_data="menu_svr_color"),
        ],
        [
            InlineKeyboardButton(f"SVR Steps: {pref['seedvr2_steps']}", callback_data="menu_svr_steps"),
            InlineKeyboardButton(f"SVR Sampler: {pref['seedvr2_sampler'][:8]}", callback_data="menu_svr_sam"),
        ],
        [InlineKeyboardButton("🔙 Kembali ke Menu Utama", callback_data="menu_main")],
    ]
    return InlineKeyboardMarkup(buttons)


def build_svr_scale_keyboard(target: str, user_id: int) -> InlineKeyboardMarkup:
    pref = get_user_pref(user_id)
    cur = pref.get("seedvr2_direct_scale", 3.0) if target == "direct" else pref["seedvr2_scale_by"]
    scales = [1.5, 2.0, 2.5, 3.0, 4.0]
    row = [InlineKeyboardButton(f"{'✅ ' if s == cur else ''}{s}x", callback_data=f"set_svr_scale:{target}:{s}") for s in scales]
    return InlineKeyboardMarkup([row, [InlineKeyboardButton("🔙 Kembali ke SeedVR2", callback_data="menu_seedvr2")]])


def build_svr_method_keyboard(user_id: int) -> InlineKeyboardMarkup:
    pref = get_user_pref(user_id)
    cur = pref["seedvr2_upscale_method"]
    methods = ["lanczos", "bicubic", "bilinear", "nearest-exact"]
    buttons = [[InlineKeyboardButton(f"{'✅ ' if m == cur else ''}{m}", callback_data=f"set_svr_method:{m}")] for m in methods]
    buttons.append([InlineKeyboardButton("🔙 Kembali ke SeedVR2", callback_data="menu_seedvr2")])
    return InlineKeyboardMarkup(buttons)


def build_svr_color_keyboard(user_id: int) -> InlineKeyboardMarkup:
    pref = get_user_pref(user_id)
    cur = pref["seedvr2_color_correction"]
    colors = ["lab", "rgb", "none"]
    row = [InlineKeyboardButton(f"{'✅ ' if c == cur else ''}{c}", callback_data=f"set_svr_color:{c}") for c in colors]
    return InlineKeyboardMarkup([row, [InlineKeyboardButton("🔙 Kembali ke SeedVR2", callback_data="menu_seedvr2")]])


def build_svr_steps_keyboard(user_id: int) -> InlineKeyboardMarkup:
    pref = get_user_pref(user_id)
    cur = pref["seedvr2_steps"]
    steps = [1, 2, 3]
    row = [InlineKeyboardButton(f"{'✅ ' if s == cur else ''}{s}", callback_data=f"set_svr_steps:{s}") for s in steps]
    return InlineKeyboardMarkup([row, [InlineKeyboardButton("🔙 Kembali ke SeedVR2", callback_data="menu_seedvr2")]])


# ==========================================
# SHIFT, BATCH, TILE SUB-MENUS
# ==========================================

def build_shift_keyboard(user_id: int) -> InlineKeyboardMarkup:
    pref = get_user_pref(user_id)
    cur = pref["auraflow_shift"]
    shifts = [1.0, 3.0, 3.16, 4.0, 5.0, 6.0, 8.0]
    r1 = [InlineKeyboardButton(f"{'✅ ' if s == cur else ''}{s}", callback_data=f"set_shift:{s}") for s in shifts[:4]]
    r2 = [InlineKeyboardButton(f"{'✅ ' if s == cur else ''}{s}", callback_data=f"set_shift:{s}") for s in shifts[4:]]
    return InlineKeyboardMarkup([r1, r2, [InlineKeyboardButton("🔙 Kembali ke Menu Utama", callback_data="menu_main")]])


def build_batch_keyboard(user_id: int) -> InlineKeyboardMarkup:
    pref = get_user_pref(user_id)
    cur = pref["batch_size"]
    batches = [1, 2, 3, 4]
    row = [InlineKeyboardButton(f"{'✅ ' if b == cur else ''}{b}", callback_data=f"set_batch:{b}") for b in batches]
    return InlineKeyboardMarkup([row, [InlineKeyboardButton("🔙 Kembali ke Menu Utama", callback_data="menu_main")]])


def build_tile_keyboard(user_id: int) -> InlineKeyboardMarkup:
    pref = get_user_pref(user_id)
    cur = pref["vae_tile_size"]
    tiles = [256, 384, 512, 768, 1024]
    row = [InlineKeyboardButton(f"{'✅ ' if t == cur else ''}{t}", callback_data=f"set_tile:{t}") for t in tiles]
    return InlineKeyboardMarkup([row, [InlineKeyboardButton("🔙 Kembali ke Menu Utama", callback_data="menu_main")]])


# Generic Sampler/Scheduler Selectors
def build_generic_sampler_keyboard(target: str, user_id: int) -> InlineKeyboardMarkup:
    pref = get_user_pref(user_id)
    if target == "p1":
        cur = pref["sampler_pass1"]
        back_target = "menu_pass1"
    elif target == "p2":
        cur = pref["sampler_pass2"]
        back_target = "menu_pass2"
    else:
        cur = pref["seedvr2_sampler"]
        back_target = "menu_seedvr2"

    buttons = []
    row = []
    for s in SAMPLERS_LIST:
        active = "✅ " if s == cur else ""
        row.append(InlineKeyboardButton(f"{active}{s}", callback_data=f"set_sam:{target}:{s}"))
        if len(row) == 2:
            buttons.append(row)
            row = []
    if row:
        buttons.append(row)
    buttons.append([InlineKeyboardButton("🔙 Kembali", callback_data=back_target)])
    return InlineKeyboardMarkup(buttons)


def build_generic_scheduler_keyboard(target: str, user_id: int) -> InlineKeyboardMarkup:
    pref = get_user_pref(user_id)
    if target == "p1":
        cur = pref["scheduler"]
        back_target = "menu_pass1"
    elif target == "p2":
        cur = pref.get("scheduler_pass2", pref["scheduler"])
        back_target = "menu_pass2"
    else:
        cur = pref["seedvr2_scheduler"]
        back_target = "menu_seedvr2"

    buttons = []
    row = []
    for sc in SCHEDULERS_LIST:
        active = "✅ " if sc == cur else ""
        row.append(InlineKeyboardButton(f"{active}{sc}", callback_data=f"set_sched:{target}:{sc}"))
        if len(row) == 2:
            buttons.append(row)
            row = []
    if row:
        buttons.append(row)
    buttons.append([InlineKeyboardButton("🔙 Kembali", callback_data=back_target)])
    return InlineKeyboardMarkup(buttons)


def format_settings_text(user_id: int) -> str:
    pref = get_user_pref(user_id)
    pipe_desc = {
        "direct": "💎 Direct 4K (Pass 1 ➔ SeedVR2 - Alami & Cepat)",
        "full": "🏛️ Full Studio 4K (Pass 1 + Pass 2 6x + SeedVR2 4K)",
        "hires": "⚡ Fast Hi-Res (Pass 1 + Pass 2 6x)",
        "draft": "🚀 Draft Cepat (Pass 1 Base Saja)",
    }
    seed_text = "Acak Tiap Generate" if pref["seed_mode"] == "random" else f"Terkunci ({pref['fixed_seed']})"
    trigger_status = "Aktif ('2s12an5, ')" if pref.get("trigger_prefix", True) else "Mati"

    return (
        "⚙️ **PANEL KONTROL STUDIO COMFYUI (100% EXPOSED)**\n\n"
        f"• **Model:** `{pref['preset']}` | **Shift:** `{pref['auraflow_shift']}`\n"
        f"• **Alur:** `{pipe_desc.get(pref['pipeline_mode'])}`\n"
        f"• **Ukuran:** `{pref['dimensions']}` | **Batch:** `{pref['batch_size']}`\n"
        f"• **Trigger Word:** `{trigger_status}` | **LoRA:** `{pref['lora_strength']}`\n"
        f"• **Master Seed:** `{seed_text}` | **VAE Tile:** `{pref['vae_tile_size']}px`\n\n"
        f"🎛️ **Pass 1:** Steps: `{pref['steps_pass1']}` | CFG: `{pref['cfg_pass1']}` | Sampler: `{pref['sampler_pass1']}` | Sched: `{pref['scheduler']}`\n"
        f"🔬 **Pass 2:** Denoise: `{pref['denoise_pass2']}` | Steps: `{pref['steps_pass2']}` | CFG: `{pref['cfg_pass2']}` | Upscale: `{pref['latent_upscale_method']} ({pref['latent_scale_by']}x)` | Sampler: `{pref['sampler_pass2']}`\n"
        f"🧬 **SeedVR2:** Direct Scale: `{pref.get('seedvr2_direct_scale', 3.0)}x` | P2 Scale: `{pref['seedvr2_scale_by']}x` | Steps: `{pref['seedvr2_steps']}` | Method: `{pref['seedvr2_upscale_method']}` | Color: `{pref['seedvr2_color_correction']}`\n\n"
        "💡 *Ketik `/set <param> <nilai>` untuk mengubah nilai bebas apapun kapan saja.*"
    )


# ==========================================
# COMMAND & CALLBACK HANDLERS
# ==========================================

async def start_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    allowed = context.bot_data["config"]["telegram"]["allowed_user_ids"]

    if not is_authorized(user.id, allowed):
        await update.message.reply_text("⛔ Akses ditolak. Bot ini diprivat untuk pemilik saja.")
        return

    server_url = context.bot_data["config"]["comfyui"]["server_url"]
    online = await check_comfy_online(server_url)
    status_str = "🟢 **Online** (Siap Generate)" if online else "🔴 **Offline** (Jalankan `uv run main.py --cuda-device 0`)"

    msg = (
        f"Halo **{user.first_name}**! 🎨\n\n"
        f"🖥️ **Status ComfyUI:** {status_str}\n\n"
        "💡 **Panduan Kontrol Studio Penuh:**\n"
        "• Ketik /settings untuk membuka panel kontrol studio interaktif lengkap.\n"
        "• Ketik /set `<param>` `<nilai>` untuk mengubah parameter apa saja bebas (contoh: `/set shift 3.16`, `/set denoise 0.35`, `/set batch 2`, `/set tile 768`).\n"
        "• Ketik /seed untuk mengecek atau mengunci seed aktif.\n"
        "• Kirim prompt apa saja untuk mulai render studio 4K!\n"
    )
    await update.message.reply_text(msg, parse_mode=ParseMode.MARKDOWN)


async def settings_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    allowed = context.bot_data["config"]["telegram"]["allowed_user_ids"]
    if not is_authorized(user.id, allowed):
        return

    await update.message.reply_text(
        format_settings_text(user.id),
        reply_markup=build_main_settings_keyboard(user.id),
        parse_mode=ParseMode.MARKDOWN,
    )


async def set_command_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    allowed = context.bot_data["config"]["telegram"]["allowed_user_ids"]
    if not is_authorized(user.id, allowed):
        return

    args = context.args
    if len(args) < 2:
        help_msg = (
            "🛠️ **Daftar Parameter Lengkap /set:**\n\n"
            "• `/set shift <float>` (misal `/set shift 3.16` atau `/set shift 6.0`)\n"
            "• `/set batch <int>` (misal `/set batch 2`)\n"
            "• `/set denoise <float>` (misal `/set denoise 0.35`)\n"
            "• `/set steps <int>` (Pass 1 steps)\n"
            "• `/set cfg <float>` (Pass 1 CFG)\n"
            "• `/set p2steps <int>` (Pass 2 steps)\n"
            "• `/set p2cfg <float>` (Pass 2 CFG)\n"
            "• `/set scale <float>` (Latent scale 6.0x)\n"
            "• `/set svrscale <float>` (SeedVR2 scale)\n"
            "• `/set svrsteps <int>` (SeedVR2 steps)\n"
            "• `/set svrcfg <float>` (SeedVR2 CFG)\n"
            "• `/set tile <int>` (VAE tile size: 256, 512, 768)\n"
            "• `/set overlap <int>` (VAE overlap)\n"
            "• `/set lora <float>` (LoRA strength)\n"
            "• `/set seed <int>` (Lock master seed)\n"
            "• `/set prefix <str>` (Filename prefix)\n"
        )
        await update.message.reply_text(help_msg, parse_mode=ParseMode.MARKDOWN)
        return

    key = args[0].lower().strip()
    val = args[1].strip()
    pref = get_user_pref(user.id)

    try:
        if key in ["shift", "auraflow_shift"]:
            pref["auraflow_shift"] = float(val)
            await update.message.reply_text(f"✅ AuraFlow Shift diubah menjadi `{pref['auraflow_shift']}`", parse_mode=ParseMode.MARKDOWN)
        elif key in ["batch", "batch_size"]:
            pref["batch_size"] = max(1, min(16, int(val)))
            await update.message.reply_text(f"✅ Batch Size diubah menjadi `{pref['batch_size']}`", parse_mode=ParseMode.MARKDOWN)
        elif key in ["tile", "tile_size", "vae_tile_size"]:
            pref["vae_tile_size"] = int(val)
            await update.message.reply_text(f"✅ VAE Tile Size diubah menjadi `{pref['vae_tile_size']}`px", parse_mode=ParseMode.MARKDOWN)
        elif key in ["overlap", "vae_overlap"]:
            pref["vae_overlap"] = int(val)
            await update.message.reply_text(f"✅ VAE Overlap diubah menjadi `{pref['vae_overlap']}`px", parse_mode=ParseMode.MARKDOWN)
        elif key in ["denoise", "p2denoise", "denoise_pass2"]:
            pref["denoise_pass2"] = float(val)
            await update.message.reply_text(f"✅ Pass 2 Denoise diubah menjadi `{pref['denoise_pass2']}`", parse_mode=ParseMode.MARKDOWN)
        elif key in ["steps", "p1steps", "steps_pass1"]:
            pref["steps_pass1"] = int(val)
            await update.message.reply_text(f"✅ Pass 1 Steps diubah menjadi `{pref['steps_pass1']}`", parse_mode=ParseMode.MARKDOWN)
        elif key in ["cfg", "p1cfg", "cfg_pass1"]:
            pref["cfg_pass1"] = float(val)
            await update.message.reply_text(f"✅ Pass 1 CFG diubah menjadi `{pref['cfg_pass1']}`", parse_mode=ParseMode.MARKDOWN)
        elif key in ["p2steps", "steps_pass2"]:
            pref["steps_pass2"] = int(val)
            await update.message.reply_text(f"✅ Pass 2 Steps diubah menjadi `{pref['steps_pass2']}`", parse_mode=ParseMode.MARKDOWN)
        elif key in ["p2cfg", "cfg_pass2"]:
            pref["cfg_pass2"] = float(val)
            await update.message.reply_text(f"✅ Pass 2 CFG diubah menjadi `{pref['cfg_pass2']}`", parse_mode=ParseMode.MARKDOWN)
        elif key in ["svrsteps", "seedvr2_steps"]:
            pref["seedvr2_steps"] = int(val)
            await update.message.reply_text(f"✅ SeedVR2 Steps diubah menjadi `{pref['seedvr2_steps']}`", parse_mode=ParseMode.MARKDOWN)
        elif key in ["svrcfg", "seedvr2_cfg"]:
            pref["seedvr2_cfg"] = float(val)
            await update.message.reply_text(f"✅ SeedVR2 CFG diubah menjadi `{pref['seedvr2_cfg']}`", parse_mode=ParseMode.MARKDOWN)
        elif key in ["svrscale", "seedvr2_scale"]:
            pref["seedvr2_scale_by"] = float(val)
            pref["seedvr2_direct_scale"] = float(val)
            await update.message.reply_text(f"✅ SeedVR2 Scale diubah menjadi `{val}x`", parse_mode=ParseMode.MARKDOWN)
        elif key in ["lora", "lora_strength"]:
            pref["lora_strength"] = float(val)
            await update.message.reply_text(f"✅ LoRA Strength diubah menjadi `{pref['lora_strength']}`", parse_mode=ParseMode.MARKDOWN)
        elif key in ["seed"]:
            pref["fixed_seed"] = int(val)
            pref["seed_mode"] = "fixed"
            await update.message.reply_text(f"✅ Master Seed dikunci ke `{pref['fixed_seed']}`", parse_mode=ParseMode.MARKDOWN)
        elif key in ["scale", "latent_scale"]:
            pref["latent_scale_by"] = float(val)
            await update.message.reply_text(f"✅ Latent Scale diubah menjadi `{pref['latent_scale_by']}x`", parse_mode=ParseMode.MARKDOWN)
        elif key in ["prefix", "filename_prefix"]:
            pref["filename_prefix"] = val
            await update.message.reply_text(f"✅ Filename Prefix diubah menjadi `{pref['filename_prefix']}`", parse_mode=ParseMode.MARKDOWN)
        else:
            await update.message.reply_text(f"⚠️ Parameter `{key}` tidak dikenal. Ketik `/set` tanpa argumen untuk melihat bantuan.", parse_mode=ParseMode.MARKDOWN)
    except ValueError:
        await update.message.reply_text(f"⚠️ Format nilai salah untuk parameter `{key}`.", parse_mode=ParseMode.MARKDOWN)


async def seed_command_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    allowed = context.bot_data["config"]["telegram"]["allowed_user_ids"]
    if not is_authorized(user.id, allowed):
        return

    pref = get_user_pref(user.id)
    if context.args:
        try:
            val = int(context.args[0])
            pref["fixed_seed"] = val
            pref["seed_mode"] = "fixed"
            await update.message.reply_text(f"🔒 **Seed Terkunci:** `{val}`", parse_mode=ParseMode.MARKDOWN)
            return
        except ValueError:
            pass

    status = "🎲 **Acak Tiap Generate**" if pref["seed_mode"] == "random" else f"🔒 **Terkunci:** `{pref['fixed_seed']}`"
    await update.message.reply_text(
        f"Status Master Seed Anda saat ini: {status}\n\nKetik `/seed <angka>` untuk mengunci seed spesifik.",
        parse_mode=ParseMode.MARKDOWN,
    )


async def callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    user = query.from_user
    allowed = context.bot_data["config"]["telegram"]["allowed_user_ids"]
    if not is_authorized(user.id, allowed):
        return

    pref = get_user_pref(user.id)
    data = query.data

    # Navigation & General
    if data == "close_menu":
        await query.message.delete()
        return

    elif data == "menu_main":
        await query.edit_message_text(
            format_settings_text(user.id),
            reply_markup=build_main_settings_keyboard(user.id),
            parse_mode=ParseMode.MARKDOWN,
        )
        return

    elif data == "toggle_mode":
        is_turbo = "Turbo" in pref["preset"]
        if is_turbo:
            pref["preset"] = "ZI (Base - 28 Steps, CFG 4.0)"
            pref["steps_pass1"] = 28
            pref["cfg_pass1"] = 4.0
            pref["steps_pass2"] = 20
            pref["cfg_pass2"] = 4.0
        else:
            pref["preset"] = "ZIT (Turbo - 8 Steps, CFG 1.0)"
            pref["steps_pass1"] = 8
            pref["cfg_pass1"] = 1.0
            pref["steps_pass2"] = 8
            pref["cfg_pass2"] = 1.0

    elif data == "toggle_trigger":
        pref["trigger_prefix"] = not pref.get("trigger_prefix", True)

    elif data == "menu_pipeline":
        await query.edit_message_reply_markup(reply_markup=build_pipeline_keyboard(user.id))
        return

    elif data.startswith("set_pipe:"):
        pref["pipeline_mode"] = data.split("set_pipe:")[1]

    elif data == "menu_aspect":
        await query.edit_message_reply_markup(reply_markup=build_aspect_keyboard(user.id))
        return

    elif data.startswith("set_aspect:"):
        pref["dimensions"] = data.split("set_aspect:")[1]

    elif data == "menu_lora":
        await query.edit_message_reply_markup(reply_markup=build_lora_keyboard(user.id))
        return

    elif data.startswith("set_lora:"):
        pref["lora_strength"] = float(data.split("set_lora:")[1])

    elif data == "menu_seed":
        await query.edit_message_reply_markup(reply_markup=build_seed_keyboard(user.id))
        return

    elif data.startswith("set_seed_mode:"):
        pref["seed_mode"] = data.split("set_seed_mode:")[1]

    elif data == "set_seed_roll":
        pref["fixed_seed"] = random.randint(1, 2**48 - 1)
        pref["seed_mode"] = "fixed"
        await query.edit_message_reply_markup(reply_markup=build_seed_keyboard(user.id))
        return

    # Shift, Batch, Tile Menus
    elif data == "menu_shift":
        await query.edit_message_reply_markup(reply_markup=build_shift_keyboard(user.id))
        return

    elif data.startswith("set_shift:"):
        pref["auraflow_shift"] = float(data.split("set_shift:")[1])

    elif data == "menu_batch":
        await query.edit_message_reply_markup(reply_markup=build_batch_keyboard(user.id))
        return

    elif data.startswith("set_batch:"):
        pref["batch_size"] = int(data.split("set_batch:")[1])

    elif data == "menu_tile":
        await query.edit_message_reply_markup(reply_markup=build_tile_keyboard(user.id))
        return

    elif data.startswith("set_tile:"):
        pref["vae_tile_size"] = int(data.split("set_tile:")[1])

    # Pass 1 sub-menus
    elif data == "menu_pass1":
        await query.edit_message_reply_markup(reply_markup=build_pass1_keyboard(user.id))
        return

    elif data == "menu_p1_steps":
        await query.edit_message_reply_markup(reply_markup=build_p1_steps_keyboard(user.id))
        return

    elif data.startswith("set_p1_steps:"):
        pref["steps_pass1"] = int(data.split("set_p1_steps:")[1])
        await query.edit_message_reply_markup(reply_markup=build_pass1_keyboard(user.id))
        return

    elif data == "menu_p1_cfg":
        await query.edit_message_reply_markup(reply_markup=build_p1_cfg_keyboard(user.id))
        return

    elif data.startswith("set_p1_cfg:"):
        pref["cfg_pass1"] = float(data.split("set_p1_cfg:")[1])
        await query.edit_message_reply_markup(reply_markup=build_pass1_keyboard(user.id))
        return

    elif data == "menu_p1_sam":
        await query.edit_message_reply_markup(reply_markup=build_generic_sampler_keyboard("p1", user.id))
        return

    elif data == "menu_p1_sched":
        await query.edit_message_reply_markup(reply_markup=build_generic_scheduler_keyboard("p1", user.id))
        return

    # Pass 2 sub-menus
    elif data == "menu_pass2":
        await query.edit_message_reply_markup(reply_markup=build_pass2_keyboard(user.id))
        return

    elif data == "menu_p2_denoise":
        await query.edit_message_reply_markup(reply_markup=build_p2_denoise_keyboard(user.id))
        return

    elif data.startswith("set_p2_denoise:"):
        pref["denoise_pass2"] = float(data.split("set_p2_denoise:")[1])
        await query.edit_message_reply_markup(reply_markup=build_pass2_keyboard(user.id))
        return

    elif data == "menu_p2_steps":
        await query.edit_message_reply_markup(reply_markup=build_p2_steps_keyboard(user.id))
        return

    elif data.startswith("set_p2_steps:"):
        pref["steps_pass2"] = int(data.split("set_p2_steps:")[1])
        await query.edit_message_reply_markup(reply_markup=build_pass2_keyboard(user.id))
        return

    elif data == "menu_p2_cfg":
        await query.edit_message_reply_markup(reply_markup=build_p2_cfg_keyboard(user.id))
        return

    elif data.startswith("set_p2_cfg:"):
        pref["cfg_pass2"] = float(data.split("set_p2_cfg:")[1])
        await query.edit_message_reply_markup(reply_markup=build_pass2_keyboard(user.id))
        return

    elif data == "menu_p2_scale":
        await query.edit_message_reply_markup(reply_markup=build_p2_scale_keyboard(user.id))
        return

    elif data.startswith("set_p2_scale:"):
        pref["latent_scale_by"] = float(data.split("set_p2_scale:")[1])
        await query.edit_message_reply_markup(reply_markup=build_pass2_keyboard(user.id))
        return

    elif data == "menu_p2_method":
        await query.edit_message_reply_markup(reply_markup=build_p2_method_keyboard(user.id))
        return

    elif data.startswith("set_p2_method:"):
        pref["latent_upscale_method"] = data.split("set_p2_method:")[1]
        await query.edit_message_reply_markup(reply_markup=build_pass2_keyboard(user.id))
        return

    elif data == "menu_p2_sam":
        await query.edit_message_reply_markup(reply_markup=build_generic_sampler_keyboard("p2", user.id))
        return

    elif data == "menu_p2_sched":
        await query.edit_message_reply_markup(reply_markup=build_generic_scheduler_keyboard("p2", user.id))
        return

    # Generic Sampler/Scheduler Callbacks
    elif data.startswith("set_sam:"):
        _, target, sam = data.split(":")
        if target == "p1":
            pref["sampler_pass1"] = sam
            target_menu = build_pass1_keyboard(user.id)
        elif target == "p2":
            pref["sampler_pass2"] = sam
            target_menu = build_pass2_keyboard(user.id)
        else:
            pref["seedvr2_sampler"] = sam
            target_menu = build_seedvr2_keyboard(user.id)
        await query.edit_message_reply_markup(reply_markup=target_menu)
        return

    elif data.startswith("set_sched:"):
        _, target, sc = data.split(":")
        if target == "p1":
            pref["scheduler"] = sc
            target_menu = build_pass1_keyboard(user.id)
        elif target == "p2":
            pref["scheduler_pass2"] = sc
            target_menu = build_pass2_keyboard(user.id)
        else:
            pref["seedvr2_scheduler"] = sc
            target_menu = build_seedvr2_keyboard(user.id)
        await query.edit_message_reply_markup(reply_markup=target_menu)
        return

    # SeedVR2 sub-menus
    elif data == "menu_seedvr2":
        await query.edit_message_reply_markup(reply_markup=build_seedvr2_keyboard(user.id))
        return

    elif data == "menu_svr_dscale":
        await query.edit_message_reply_markup(reply_markup=build_svr_scale_keyboard("direct", user.id))
        return

    elif data == "menu_svr_scale":
        await query.edit_message_reply_markup(reply_markup=build_svr_scale_keyboard("pass2", user.id))
        return

    elif data.startswith("set_svr_scale:"):
        _, target, sc = data.split(":")
        val = float(sc)
        if target == "direct":
            pref["seedvr2_direct_scale"] = val
        else:
            pref["seedvr2_scale_by"] = val
        await query.edit_message_reply_markup(reply_markup=build_seedvr2_keyboard(user.id))
        return

    elif data == "menu_svr_method":
        await query.edit_message_reply_markup(reply_markup=build_svr_method_keyboard(user.id))
        return

    elif data.startswith("set_svr_method:"):
        pref["seedvr2_upscale_method"] = data.split("set_svr_method:")[1]
        await query.edit_message_reply_markup(reply_markup=build_seedvr2_keyboard(user.id))
        return

    elif data == "menu_svr_color":
        await query.edit_message_reply_markup(reply_markup=build_svr_color_keyboard(user.id))
        return

    elif data.startswith("set_svr_color:"):
        pref["seedvr2_color_correction"] = data.split("set_svr_color:")[1]
        await query.edit_message_reply_markup(reply_markup=build_seedvr2_keyboard(user.id))
        return

    elif data == "menu_svr_steps":
        await query.edit_message_reply_markup(reply_markup=build_svr_steps_keyboard(user.id))
        return

    elif data.startswith("set_svr_steps:"):
        pref["seedvr2_steps"] = int(data.split("set_svr_steps:")[1])
        await query.edit_message_reply_markup(reply_markup=build_seedvr2_keyboard(user.id))
        return

    elif data == "menu_svr_sam":
        await query.edit_message_reply_markup(reply_markup=build_generic_sampler_keyboard("svr", user.id))
        return

    elif data == "reset_settings":
        user_sessions[user.id] = load_default_preferences()

    # Re-render main menu
    await query.edit_message_text(
        format_settings_text(user.id),
        reply_markup=build_main_settings_keyboard(user.id),
        parse_mode=ParseMode.MARKDOWN,
    )


# ==========================================
# PROMPT EXECUTION ENGINE
# ==========================================

def render_progress_bar(current: int, total: int, length: int = 10) -> str:
    if total <= 0:
        return "░" * length
    fraction = min(1.0, max(0.0, current / total))
    filled = int(round(length * fraction))
    return "█" * filled + "░" * (length - filled)


async def execute_prompt(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    allowed = context.bot_data["config"]["telegram"]["allowed_user_ids"]

    if not is_authorized(user.id, allowed):
        await update.message.reply_text("⛔ Akses ditolak. ID Telegram Anda tidak terdaftar.")
        return

    server_url = context.bot_data["config"]["comfyui"]["server_url"]
    online = await check_comfy_online(server_url)
    if not online:
        await update.message.reply_text("⚠️ ComfyUI backend sedang offline.\nSilakan jalankan `uv run main.py --cuda-device 0` di PC Anda.")
        return

    raw_text = update.message.text.strip()
    pref = get_user_pref(user.id)

    # Negative Prompt handling via --neg flag
    neg_prompt_text = ""
    if "--neg" in raw_text:
        parts = raw_text.split("--neg", 1)
        prompt_text = parts[0].strip()
        neg_prompt_text = parts[1].strip()
    else:
        prompt_text = raw_text

    # Trigger Word prefix handling
    if pref.get("trigger_prefix", True) and not prompt_text.startswith("2s12an5"):
        prompt_text = f"2s12an5, {prompt_text}"

    current_seed = (
        random.randint(1, 2**48 - 1)
        if pref["seed_mode"] == "random"
        else pref["fixed_seed"]
    )

    # Load base API graph
    with open(TEMPLATE_PATH, "r", encoding="utf-8") as f:
        graph = json.load(f)

    # 1. Positive Prompt (Node 3)
    graph["3"]["inputs"]["text"] = prompt_text

    # 2. Base Dimensions & Batch (Node 69)
    if "6" in graph:
        graph["6"]["inputs"]["dimensions"] = pref["dimensions"]
    elif "69" in graph:
        dim_key = pref["dimensions"]
        if dim_key in BASE_DIMS_MAP:
            w, h = BASE_DIMS_MAP[dim_key]
        else:
            w, h = pref.get("custom_width", 224), pref.get("custom_height", 288)
        graph["69"]["inputs"]["width"] = w
        graph["69"]["inputs"]["height"] = h
        graph["69"]["inputs"]["batch_size"] = pref.get("batch_size", 1)

    # 3. Model UNET (Node 13) & AuraFlow Shift (Node 15)
    is_turbo = "Turbo" in pref["preset"]
    graph["13"]["inputs"]["unet_name"] = "z_image_turbo_int8_convrot.safetensors" if is_turbo else "z_image_int8_convrot.safetensors"
    if "15" in graph:
        graph["15"]["inputs"]["shift"] = pref.get("auraflow_shift", 6.0)

    # 4. Pass 1 KSampler (Node 7)
    if "7" in graph:
        graph["7"]["inputs"]["steps"] = pref["steps_pass1"]
        graph["7"]["inputs"]["cfg"] = pref["cfg_pass1"]
        graph["7"]["inputs"]["sampler_name"] = pref["sampler_pass1"]
        graph["7"]["inputs"]["scheduler"] = pref["scheduler"]

    # 5. Pass 2 Latent Upscale (Node 70)
    if "70" in graph:
        graph["70"]["inputs"]["upscale_method"] = pref.get("latent_upscale_method", "bislerp")
        graph["70"]["inputs"]["scale_by"] = pref.get("latent_scale_by", 6.0)

    # 6. Pass 2 KSampler (Node 29)
    if "29" in graph:
        graph["29"]["inputs"]["steps"] = pref["steps_pass2"]
        graph["29"]["inputs"]["cfg"] = pref["cfg_pass2"]
        graph["29"]["inputs"]["sampler_name"] = pref["sampler_pass2"]
        graph["29"]["inputs"]["scheduler"] = pref.get("scheduler_pass2", pref["scheduler"])
        graph["29"]["inputs"]["denoise"] = pref["denoise_pass2"]

    # 7. SeedVR2 Post / Color / Upscale (Node 52, 56, 58)
    if "52" in graph:
        graph["52"]["inputs"]["upscale_method"] = pref.get("seedvr2_upscale_method", "lanczos")
    if "56" in graph:
        graph["56"]["inputs"]["steps"] = pref.get("seedvr2_steps", 1)
        graph["56"]["inputs"]["cfg"] = pref.get("seedvr2_cfg", 1.0)
        graph["56"]["inputs"]["sampler_name"] = pref.get("seedvr2_sampler", "dpmpp_sde")
        graph["56"]["inputs"]["scheduler"] = pref.get("seedvr2_scheduler", "beta")
    if "58" in graph:
        graph["58"]["inputs"]["color_correction_method"] = pref.get("seedvr2_color_correction", "lab")

    # 8. Tiled VAE Settings (Node 31, 54, 57)
    tile_size = pref.get("vae_tile_size", 512)
    tile_overlap = pref.get("vae_overlap", 64)
    for vn in ["31", "54", "57"]:
        if vn in graph:
            graph[vn]["inputs"]["tile_size"] = tile_size
            graph[vn]["inputs"]["overlap"] = tile_overlap

    # 9. Master Seed & File Prefixes
    fn_pfx = pref.get("filename_prefix", "zi")
    graph["68"]["inputs"]["seed"] = current_seed
    if "14" in graph:
        graph["14"]["inputs"]["filename_prefix"] = f"{fn_pfx}-{current_seed}-1-base"
    if "32" in graph:
        graph["32"]["inputs"]["filename_prefix"] = f"{fn_pfx}-{current_seed}-2-hires"
    if "59" in graph:
        graph["59"]["inputs"]["filename_prefix"] = f"{fn_pfx}-{current_seed}-3-4k"

    # 10. LoRA (Node 67)
    lora_strength = pref["lora_strength"]
    graph["67"]["inputs"]["lora_1"] = {
        "on": lora_strength > 0,
        "lora": "NIceAsians_Zimage.safetensors",
        "strength": lora_strength,
        "strengthTwo": pref.get("lora_clip_strength", lora_strength),
    }

    # 11. Remove UI-only comparison node for Telegram headless executions
    graph.pop("38", None)

    # 12. Adapt Pipeline Mode (Prune unneeded execution nodes)
    pipeline_mode = pref["pipeline_mode"]
    target_save_node = "59"

    if pipeline_mode == "direct":
        for nid in ["29", "70", "31", "32"]:
            graph.pop(nid, None)
        graph["52"]["inputs"]["image"] = ["9", 0]
        graph["52"]["inputs"]["scale_by"] = pref.get("seedvr2_direct_scale", 3.0)
        target_save_node = "59"
    elif pipeline_mode == "draft":
        for nid in ["29", "70", "31", "32", "50", "51", "52", "53", "54", "55", "56", "57", "58", "59"]:
            graph.pop(nid, None)
        target_save_node = "14"
    elif pipeline_mode == "hires":
        for nid in ["50", "51", "52", "53", "54", "55", "56", "57", "58", "59"]:
            graph.pop(nid, None)
        target_save_node = "32"
    else:  # full
        graph["52"]["inputs"]["image"] = ["31", 0]
        graph["52"]["inputs"]["scale_by"] = pref.get("seedvr2_scale_by", 2.0)
        target_save_node = "59"

    mode_label = {
        "direct": "💎 Direct 4K SeedVR2 Studio",
        "full": "🏛️ Full 3-Pass 4K Studio",
        "hires": "⚡ Fast Hi-Res 1.5x",
        "draft": "🚀 Draft Cepat Pass 1",
    }[pipeline_mode]

    status_msg = await update.message.reply_text(
        f"⏳ **Mengirim antrean ke ComfyUI...**\n"
        f"• Alur: **{mode_label}**\n"
        f"• Seed: `{current_seed}`\n"
        f"• Ukuran: `{pref['dimensions']}` (Batch: `{pref.get('batch_size', 1)}`)\n"
        f"• Shift: `{pref.get('auraflow_shift', 6.0)}` | LoRA: `{lora_strength}`",
        parse_mode=ParseMode.MARKDOWN,
    )

    client_id = str(uuid.uuid4())
    ws_host = server_url.replace("http://", "").replace("https://", "")
    ws_url = f"ws://{ws_host}/ws?clientId={client_id}"

    prompt_id = None
    last_ui_update = 0.0
    last_text = ""

    async with aiohttp.ClientSession() as session:
        try:
            async with session.ws_connect(ws_url) as ws:
                payload = {"prompt": graph, "client_id": client_id}
                async with session.post(f"{server_url}/prompt", json=payload) as resp:
                    if resp.status != 200:
                        err_text = await resp.text()
                        await status_msg.edit_text(f"❌ **Gagal mengirim prompt ke ComfyUI:**\n`{err_text}`", parse_mode=ParseMode.MARKDOWN)
                        return
                    res_data = await resp.json()
                    prompt_id = res_data.get("prompt_id")

                async for msg in ws:
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        event = json.loads(msg.data)
                        etype = event.get("type")
                        edata = event.get("data", {})

                        if etype == "progress":
                            val = edata.get("value", 0)
                            max_v = edata.get("max", 0)
                            now = asyncio.get_event_loop().time()
                            if now - last_ui_update > 0.8:
                                bar = render_progress_bar(val, max_v)
                                new_text = (
                                    f"🎨 **Sedang Merender...**\n"
                                    f"`[{bar}]` **{val}/{max_v}** ({int(val/max_v*100 if max_v else 0)}%)\n"
                                    f"• Alur: **{mode_label}**\n"
                                    f"• Seed: `{current_seed}`"
                                )
                                if new_text != last_text:
                                    try:
                                        await status_msg.edit_text(new_text, parse_mode=ParseMode.MARKDOWN)
                                        last_text = new_text
                                        last_ui_update = now
                                    except Exception:
                                        pass

                        elif etype == "executing":
                            node = edata.get("node")
                            if node is None and edata.get("prompt_id") == prompt_id:
                                break

                        elif etype == "execution_error":
                            if edata.get("prompt_id") == prompt_id:
                                err_trace = edata.get("exception_message", "Unknown error")
                                await status_msg.edit_text(f"❌ **Terjadi kesalahan eksekusi:**\n`{err_trace}`", parse_mode=ParseMode.MARKDOWN)
                                return
        except Exception as e:
            print(f"WS error: {e}")

    # Fetch Execution History & Output Images
    images = []
    async with aiohttp.ClientSession() as session:
        async with session.get(f"{server_url}/history/{prompt_id}") as resp:
            if resp.status == 200:
                hdata = await resp.json()
                outputs = hdata.get(prompt_id, {}).get("outputs", {})
                if target_save_node in outputs and "images" in outputs[target_save_node]:
                    images = outputs[target_save_node]["images"]
                else:
                    for out_node in [target_save_node, "59", "32", "14"]:
                        if out_node in outputs and "images" in outputs[out_node]:
                            images = outputs[out_node]["images"]
                            break

    if not images:
        await status_msg.edit_text("❌ Render selesai tetapi file gambar tidak ditemukan di output ComfyUI.")
        return

    try:
        await status_msg.delete()
    except Exception:
        pass

    for img_idx, final_img_info in enumerate(images, 1):
        filename = final_img_info["filename"]
        subfolder = final_img_info.get("subfolder", "")
        local_output_path = Path("output") / subfolder / filename

        real_w, real_h = 0, 0
        file_size_mb = 0.0
        if local_output_path.exists():
            with Image.open(local_output_path) as im:
                real_w, real_h = im.size
            file_size_mb = local_output_path.stat().st_size / (1024 * 1024)

        batch_str = f" [Item {img_idx}/{len(images)}]" if len(images) > 1 else ""
        caption = (
            f"✨ **Render Selesai!**{batch_str}\n"
            f"• Alur: **{mode_label}**\n"
            f"• Resolusi: **{real_w}x{real_h}** ({file_size_mb:.2f} MB)\n"
            f"• Seed: `{current_seed}` | Shift: `{pref.get('auraflow_shift', 6.0)}`\n"
            f"• P1 Steps/CFG: `{pref['steps_pass1']}` / `{pref['cfg_pass1']}`\n"
            f"• P2 Denoise/Upscale: `{pref['denoise_pass2']}` / `{pref.get('latent_scale_by', 6.0)}x`\n"
            f"• LoRA: `{lora_strength}` | Tile: `{tile_size}px`\n"
        )

        try:
            with open(local_output_path, "rb") as f:
                await update.message.reply_photo(
                    photo=f,
                    caption=caption,
                    parse_mode=ParseMode.MARKDOWN,
                )
                f.seek(0)
                await update.message.reply_document(
                    document=f,
                    filename=filename,
                    caption=f"📁 File Asli: `{filename}`",
                    parse_mode=ParseMode.MARKDOWN,
                )
        except Exception as e:
            print(f"Delivery exception: {e}")


async def global_error_handler(update: object, context: ContextTypes.DEFAULT_TYPE):
    from telegram.error import BadRequest, TimedOut
    if isinstance(context.error, BadRequest) and "Message is not modified" in str(context.error):
        return
    if isinstance(context.error, TimedOut):
        print(f"Network timeout handled: {context.error}")
        return
    print(f"Unhandled error: {context.error}")


def main():
    config = load_config()
    token = config["telegram"]["bot_token"]

    httpx_request = HTTPXRequest(
        connection_pool_size=8,
        connect_timeout=60.0,
        read_timeout=None,
        write_timeout=None,
    )

    app = Application.builder().token(token).request(httpx_request).build()
    app.bot_data["config"] = config

    app.add_handler(CommandHandler("start", start_handler))
    app.add_handler(CommandHandler("settings", settings_handler))
    app.add_handler(CommandHandler("seed", seed_command_handler))
    app.add_handler(CommandHandler("set", set_command_handler))
    app.add_handler(CallbackQueryHandler(callback_handler))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, execute_prompt))

    app.add_error_handler(global_error_handler)

    print("==================================================")
    print("🤖 Telegram Bot Studio Client ComfyUI Siap Beroperasi!")
    print(f"   Server Target : {config['comfyui']['server_url']}")
    print(f"   Mode Kontrol  : 100% Full Studio Parameters Exposed")
    print("==================================================")

    app.run_polling()


if __name__ == "__main__":
    main()
