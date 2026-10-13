"""
Telegram Bot Sidecar Client for ComfyUI.
Clean, dedicated exposure of core studio parameters (without LoRA):
1. Shift AuraFlow (Node 15)
2. Master Seed (Node 68)
3. KSampler Pass 1 All Inputs (Node 7: steps, cfg, sampler, scheduler, denoise)
4. KSampler Pass 2 All Inputs (Node 29: steps, cfg, sampler, scheduler, denoise)
"""

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
from telegram.error import TimedOut, BadRequest
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

# Set Windows console UTF-8 encoding
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

CONFIG_PATH = Path("scripts/config.telegram.toml")
WORKFLOW_PATH = Path("user/default/workflows/z-image.json")
TEMPLATE_PATH = Path("scripts/z_image_api_template.json")

user_sessions = {}

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
    """Reads default values directly from the web workflow JSON file."""
    defaults = {
        # 1. Shift AuraFlow
        "auraflow_shift": 6.0,
        # 2. Master Seed
        "seed_mode": "random",
        "fixed_seed": 351971767740904,
        # 3. KSampler Pass 1 All Inputs
        "p1_steps": 9,
        "p1_cfg": 4.0,
        "p1_sampler": "euler_ancestral",
        "p1_scheduler": "ddim_uniform",
        "p1_denoise": 1.0,
        # 4. KSampler Pass 2 All Inputs
        "p2_steps": 9,
        "p2_cfg": 1.0,
        "p2_sampler": "dpmpp_sde",
        "p2_scheduler": "beta",
        "p2_denoise": 0.70,
    }

    if WORKFLOW_PATH.exists():
        try:
            with open(WORKFLOW_PATH, "r", encoding="utf-8") as f:
                wf = json.load(f)
            nodes = {n["id"]: n for n in wf.get("nodes", [])}

            # 1. Shift AuraFlow (Node 15)
            if 15 in nodes:
                wv = nodes[15].get("widgets_values") or []
                if wv:
                    defaults["auraflow_shift"] = float(wv[0])

            # 2. Master Seed (Node 68)
            if 68 in nodes:
                wv = nodes[68].get("widgets_values") or []
                if wv:
                    defaults["fixed_seed"] = int(wv[0])
                    if len(wv) > 1 and str(wv[1]).lower() == "randomize":
                        defaults["seed_mode"] = "random"
                    else:
                        defaults["seed_mode"] = "fixed"

            # 3. KSampler Pass 1 (Node 7)
            if 7 in nodes:
                wv = nodes[7].get("widgets_values") or []
                if len(wv) >= 7:
                    defaults["p1_steps"] = int(wv[2])
                    defaults["p1_cfg"] = float(wv[3])
                    defaults["p1_sampler"] = str(wv[4])
                    defaults["p1_scheduler"] = str(wv[5])
                    defaults["p1_denoise"] = float(wv[6])

            # 4. KSampler Pass 2 (Node 29)
            if 29 in nodes:
                wv = nodes[29].get("widgets_values") or []
                if len(wv) >= 7:
                    defaults["p2_steps"] = int(wv[2])
                    defaults["p2_cfg"] = float(wv[3])
                    defaults["p2_sampler"] = str(wv[4])
                    defaults["p2_scheduler"] = str(wv[5])
                    defaults["p2_denoise"] = float(wv[6])

        except Exception as e:
            print(f"Warning: Gagal membaca workflow '{WORKFLOW_PATH}': {e}")

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
    seed_label = "Random" if pref["seed_mode"] == "random" else f"🔒 {pref['fixed_seed']}"

    buttons = [
        [
            InlineKeyboardButton(f"🌊 Shift AuraFlow: {pref['auraflow_shift']}", callback_data="menu_shift"),
            InlineKeyboardButton(f"🎲 Seed: {seed_label}", callback_data="menu_seed"),
        ],
        [
            InlineKeyboardButton("🎛️ KSampler Pass 1 (All Inputs)", callback_data="menu_p1"),
        ],
        [
            InlineKeyboardButton("🔬 KSampler Pass 2 (All Inputs)", callback_data="menu_p2"),
        ],
        [
            InlineKeyboardButton("🔄 Reset Default (Web Workflow)", callback_data="reset_settings"),
            InlineKeyboardButton("❌ Tutup Menu", callback_data="close_menu"),
        ],
    ]
    return InlineKeyboardMarkup(buttons)


# --- 1. Shift AuraFlow Keyboard ---
def build_shift_keyboard(user_id: int) -> InlineKeyboardMarkup:
    pref = get_user_pref(user_id)
    cur = pref["auraflow_shift"]
    shifts = [1.0, 3.0, 3.16, 4.0, 5.0, 6.0, 8.0]
    r1 = [InlineKeyboardButton(f"{'✅ ' if s == cur else ''}{s}", callback_data=f"set_shift:{s}") for s in shifts[:4]]
    r2 = [InlineKeyboardButton(f"{'✅ ' if s == cur else ''}{s}", callback_data=f"set_shift:{s}") for s in shifts[4:]]
    return InlineKeyboardMarkup([r1, r2, [InlineKeyboardButton("🔙 Kembali ke Menu Utama", callback_data="menu_main")]])


# --- 2. Seed Keyboard ---
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


# --- 3. KSampler Pass 1 Menu & Sub-menus ---
def build_p1_keyboard(user_id: int) -> InlineKeyboardMarkup:
    pref = get_user_pref(user_id)
    buttons = [
        [
            InlineKeyboardButton(f"Steps: {pref['p1_steps']}", callback_data="menu_p1_steps"),
            InlineKeyboardButton(f"CFG: {pref['p1_cfg']}", callback_data="menu_p1_cfg"),
        ],
        [
            InlineKeyboardButton(f"Sampler: {pref['p1_sampler'][:10]}", callback_data="menu_p1_sam"),
            InlineKeyboardButton(f"Scheduler: {pref['p1_scheduler']}", callback_data="menu_p1_sched"),
        ],
        [
            InlineKeyboardButton(f"Denoise: {pref['p1_denoise']}", callback_data="menu_p1_denoise"),
        ],
        [InlineKeyboardButton("🔙 Kembali ke Menu Utama", callback_data="menu_main")],
    ]
    return InlineKeyboardMarkup(buttons)


def build_p1_steps_keyboard(user_id: int) -> InlineKeyboardMarkup:
    pref = get_user_pref(user_id)
    cur = pref["p1_steps"]
    steps = [4, 6, 8, 9, 10, 12, 16, 20, 28]
    r1 = [InlineKeyboardButton(f"{'✅ ' if s == cur else ''}{s}", callback_data=f"set_p1_steps:{s}") for s in steps[:5]]
    r2 = [InlineKeyboardButton(f"{'✅ ' if s == cur else ''}{s}", callback_data=f"set_p1_steps:{s}") for s in steps[5:]]
    return InlineKeyboardMarkup([r1, r2, [InlineKeyboardButton("🔙 Kembali ke Pass 1", callback_data="menu_p1")]])


def build_p1_cfg_keyboard(user_id: int) -> InlineKeyboardMarkup:
    pref = get_user_pref(user_id)
    cur = pref["p1_cfg"]
    cfgs = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0]
    r1 = [InlineKeyboardButton(f"{'✅ ' if c == cur else ''}{c}", callback_data=f"set_p1_cfg:{c}") for c in cfgs[:3]]
    r2 = [InlineKeyboardButton(f"{'✅ ' if c == cur else ''}{c}", callback_data=f"set_p1_cfg:{c}") for c in cfgs[3:]]
    return InlineKeyboardMarkup([r1, r2, [InlineKeyboardButton("🔙 Kembali ke Pass 1", callback_data="menu_p1")]])


def build_p1_denoise_keyboard(user_id: int) -> InlineKeyboardMarkup:
    pref = get_user_pref(user_id)
    cur = pref["p1_denoise"]
    vals = [0.5, 0.7, 0.85, 1.0]
    row = [InlineKeyboardButton(f"{'✅ ' if v == cur else ''}{v}", callback_data=f"set_p1_denoise:{v}") for v in vals]
    return InlineKeyboardMarkup([row, [InlineKeyboardButton("🔙 Kembali ke Pass 1", callback_data="menu_p1")]])


# --- 4. KSampler Pass 2 Menu & Sub-menus ---
def build_p2_keyboard(user_id: int) -> InlineKeyboardMarkup:
    pref = get_user_pref(user_id)
    buttons = [
        [
            InlineKeyboardButton(f"Steps: {pref['p2_steps']}", callback_data="menu_p2_steps"),
            InlineKeyboardButton(f"CFG: {pref['p2_cfg']}", callback_data="menu_p2_cfg"),
        ],
        [
            InlineKeyboardButton(f"Sampler: {pref['p2_sampler'][:10]}", callback_data="menu_p2_sam"),
            InlineKeyboardButton(f"Scheduler: {pref['p2_scheduler']}", callback_data="menu_p2_sched"),
        ],
        [
            InlineKeyboardButton(f"Denoise: {pref['p2_denoise']}", callback_data="menu_p2_denoise"),
        ],
        [InlineKeyboardButton("🔙 Kembali ke Menu Utama", callback_data="menu_main")],
    ]
    return InlineKeyboardMarkup(buttons)


def build_p2_steps_keyboard(user_id: int) -> InlineKeyboardMarkup:
    pref = get_user_pref(user_id)
    cur = pref["p2_steps"]
    steps = [4, 6, 8, 9, 10, 12, 15, 20]
    r1 = [InlineKeyboardButton(f"{'✅ ' if s == cur else ''}{s}", callback_data=f"set_p2_steps:{s}") for s in steps[:4]]
    r2 = [InlineKeyboardButton(f"{'✅ ' if s == cur else ''}{s}", callback_data=f"set_p2_steps:{s}") for s in steps[4:]]
    return InlineKeyboardMarkup([r1, r2, [InlineKeyboardButton("🔙 Kembali ke Pass 2", callback_data="menu_p2")]])


def build_p2_cfg_keyboard(user_id: int) -> InlineKeyboardMarkup:
    pref = get_user_pref(user_id)
    cur = pref["p2_cfg"]
    cfgs = [1.0, 1.5, 2.0, 3.0, 4.0]
    row = [InlineKeyboardButton(f"{'✅ ' if c == cur else ''}{c}", callback_data=f"set_p2_cfg:{c}") for c in cfgs]
    return InlineKeyboardMarkup([row, [InlineKeyboardButton("🔙 Kembali ke Pass 2", callback_data="menu_p2")]])


def build_p2_denoise_keyboard(user_id: int) -> InlineKeyboardMarkup:
    pref = get_user_pref(user_id)
    cur = pref["p2_denoise"]
    values = [0.28, 0.35, 0.38, 0.45, 0.55, 0.70]
    r1 = [InlineKeyboardButton(f"{'✅ ' if v == cur else ''}{v}", callback_data=f"set_p2_denoise:{v}") for v in values[:3]]
    r2 = [InlineKeyboardButton(f"{'✅ ' if v == cur else ''}{v}", callback_data=f"set_p2_denoise:{v}") for v in values[3:]]
    return InlineKeyboardMarkup([r1, r2, [InlineKeyboardButton("🔙 Kembali ke Pass 2", callback_data="menu_p2")]])


# Generic Sampler/Scheduler Selectors
def build_sampler_keyboard(target: str, user_id: int) -> InlineKeyboardMarkup:
    pref = get_user_pref(user_id)
    cur = pref["p1_sampler"] if target == "p1" else pref["p2_sampler"]
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
    back_target = "menu_p1" if target == "p1" else "menu_p2"
    buttons.append([InlineKeyboardButton("🔙 Kembali", callback_data=back_target)])
    return InlineKeyboardMarkup(buttons)


def build_scheduler_keyboard(target: str, user_id: int) -> InlineKeyboardMarkup:
    pref = get_user_pref(user_id)
    cur = pref["p1_scheduler"] if target == "p1" else pref["p2_scheduler"]
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
    back_target = "menu_p1" if target == "p1" else "menu_p2"
    buttons.append([InlineKeyboardButton("🔙 Kembali", callback_data=back_target)])
    return InlineKeyboardMarkup(buttons)


def format_settings_text(user_id: int) -> str:
    pref = get_user_pref(user_id)
    seed_text = "Acak Tiap Generate" if pref["seed_mode"] == "random" else f"Terkunci ({pref['fixed_seed']})"

    return (
        "⚙️ **PANEL KONTROL STUDIO Z-IMAGE**\n\n"
        f"• **AuraFlow Shift:** `{pref['auraflow_shift']}`\n"
        f"• **Master Seed:** `{seed_text}`\n\n"
        f"🎛️ **KSampler Pass 1 (Node 7):**\n"
        f"  • Steps: `{pref['p1_steps']}` | CFG: `{pref['p1_cfg']}` | Denoise: `{pref['p1_denoise']}`\n"
        f"  • Sampler: `{pref['p1_sampler']}`\n"
        f"  • Scheduler: `{pref['p1_scheduler']}`\n\n"
        f"🔬 **KSampler Pass 2 (Node 29):**\n"
        f"  • Steps: `{pref['p2_steps']}` | CFG: `{pref['p2_cfg']}` | Denoise: `{pref['p2_denoise']}`\n"
        f"  • Sampler: `{pref['p2_sampler']}`\n"
        f"  • Scheduler: `{pref['p2_scheduler']}`\n\n"
        "💡 *Tips: Anda bisa mengetik perintah `/set <param> <nilai>` kapan saja.*"
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
        "💡 **Cara Pakai:**\n"
        "• Ketik /settings untuk membuka panel kontrol studio (Shift, Seed, KSampler P1 & P2).\n"
        "• Ketik /set `<parameter>` `<nilai>` untuk mengubah nilai bebas (contoh: `/set denoise 0.40`).\n"
        "• Ketik /seed untuk melihat atau mengunci seed aktif.\n"
        "• Kirim prompt teks apa saja langsung ke chat ini untuk generate!\n"
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
            "🛠️ **Daftar Perintah /set:**\n\n"
            "• `/set shift <float>` (misal `/set shift 6.0` atau `3.16`)\n"
            "• `/set seed <int>` (misal `/set seed 123456789`)\n\n"
            "• `/set p1steps <int>` (Pass 1 steps)\n"
            "• `/set p1cfg <float>` (Pass 1 CFG)\n"
            "• `/set p1denoise <float>` (Pass 1 denoise)\n\n"
            "• `/set p2steps <int>` (Pass 2 steps)\n"
            "• `/set p2cfg <float>` (Pass 2 CFG)\n"
            "• `/set p2denoise <float>` atau `/set denoise <float>` (Pass 2 denoise)\n"
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
        elif key in ["seed"]:
            pref["fixed_seed"] = int(val)
            pref["seed_mode"] = "fixed"
            await update.message.reply_text(f"✅ Master Seed dikunci ke `{pref['fixed_seed']}`", parse_mode=ParseMode.MARKDOWN)
        # Pass 1 inputs
        elif key in ["p1steps", "steps", "steps1"]:
            pref["p1_steps"] = int(val)
            await update.message.reply_text(f"✅ Pass 1 Steps diubah menjadi `{pref['p1_steps']}`", parse_mode=ParseMode.MARKDOWN)
        elif key in ["p1cfg", "cfg", "cfg1"]:
            pref["p1_cfg"] = float(val)
            await update.message.reply_text(f"✅ Pass 1 CFG diubah menjadi `{pref['p1_cfg']}`", parse_mode=ParseMode.MARKDOWN)
        elif key in ["p1denoise", "denoise1"]:
            pref["p1_denoise"] = float(val)
            await update.message.reply_text(f"✅ Pass 1 Denoise diubah menjadi `{pref['p1_denoise']}`", parse_mode=ParseMode.MARKDOWN)
        # Pass 2 inputs
        elif key in ["p2steps", "steps2"]:
            pref["p2_steps"] = int(val)
            await update.message.reply_text(f"✅ Pass 2 Steps diubah menjadi `{pref['p2_steps']}`", parse_mode=ParseMode.MARKDOWN)
        elif key in ["p2cfg", "cfg2"]:
            pref["p2_cfg"] = float(val)
            await update.message.reply_text(f"✅ Pass 2 CFG diubah menjadi `{pref['p2_cfg']}`", parse_mode=ParseMode.MARKDOWN)
        elif key in ["p2denoise", "denoise", "denoise2"]:
            pref["p2_denoise"] = float(val)
            await update.message.reply_text(f"✅ Pass 2 Denoise diubah menjadi `{pref['p2_denoise']}`", parse_mode=ParseMode.MARKDOWN)
        else:
            await update.message.reply_text(f"⚠️ Parameter `{key}` tidak dikenal. Ketik `/set` untuk melihat daftar.", parse_mode=ParseMode.MARKDOWN)
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

    # Main navigation
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

    elif data == "reset_settings":
        user_sessions[user.id] = load_default_preferences()

    # 1. Shift
    elif data == "menu_shift":
        await query.edit_message_reply_markup(reply_markup=build_shift_keyboard(user.id))
        return

    elif data.startswith("set_shift:"):
        pref["auraflow_shift"] = float(data.split("set_shift:")[1])

    # 2. Seed
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

    # 3. Pass 1 Menus
    elif data == "menu_p1":
        await query.edit_message_reply_markup(reply_markup=build_p1_keyboard(user.id))
        return

    elif data == "menu_p1_steps":
        await query.edit_message_reply_markup(reply_markup=build_p1_steps_keyboard(user.id))
        return

    elif data.startswith("set_p1_steps:"):
        pref["p1_steps"] = int(data.split("set_p1_steps:")[1])
        await query.edit_message_reply_markup(reply_markup=build_p1_keyboard(user.id))
        return

    elif data == "menu_p1_cfg":
        await query.edit_message_reply_markup(reply_markup=build_p1_cfg_keyboard(user.id))
        return

    elif data.startswith("set_p1_cfg:"):
        pref["p1_cfg"] = float(data.split("set_p1_cfg:")[1])
        await query.edit_message_reply_markup(reply_markup=build_p1_keyboard(user.id))
        return

    elif data == "menu_p1_denoise":
        await query.edit_message_reply_markup(reply_markup=build_p1_denoise_keyboard(user.id))
        return

    elif data.startswith("set_p1_denoise:"):
        pref["p1_denoise"] = float(data.split("set_p1_denoise:")[1])
        await query.edit_message_reply_markup(reply_markup=build_p1_keyboard(user.id))
        return

    elif data == "menu_p1_sam":
        await query.edit_message_reply_markup(reply_markup=build_sampler_keyboard("p1", user.id))
        return

    elif data == "menu_p1_sched":
        await query.edit_message_reply_markup(reply_markup=build_scheduler_keyboard("p1", user.id))
        return

    # 4. Pass 2 Menus
    elif data == "menu_p2":
        await query.edit_message_reply_markup(reply_markup=build_p2_keyboard(user.id))
        return

    elif data == "menu_p2_steps":
        await query.edit_message_reply_markup(reply_markup=build_p2_steps_keyboard(user.id))
        return

    elif data.startswith("set_p2_steps:"):
        pref["p2_steps"] = int(data.split("set_p2_steps:")[1])
        await query.edit_message_reply_markup(reply_markup=build_p2_keyboard(user.id))
        return

    elif data == "menu_p2_cfg":
        await query.edit_message_reply_markup(reply_markup=build_p2_cfg_keyboard(user.id))
        return

    elif data.startswith("set_p2_cfg:"):
        pref["p2_cfg"] = float(data.split("set_p2_cfg:")[1])
        await query.edit_message_reply_markup(reply_markup=build_p2_keyboard(user.id))
        return

    elif data == "menu_p2_denoise":
        await query.edit_message_reply_markup(reply_markup=build_p2_denoise_keyboard(user.id))
        return

    elif data.startswith("set_p2_denoise:"):
        pref["p2_denoise"] = float(data.split("set_p2_denoise:")[1])
        await query.edit_message_reply_markup(reply_markup=build_p2_keyboard(user.id))
        return

    elif data == "menu_p2_sam":
        await query.edit_message_reply_markup(reply_markup=build_sampler_keyboard("p2", user.id))
        return

    elif data == "menu_p2_sched":
        await query.edit_message_reply_markup(reply_markup=build_scheduler_keyboard("p2", user.id))
        return

    # Generic Sampler/Scheduler Callbacks
    elif data.startswith("set_sam:"):
        _, target, sam = data.split(":")
        if target == "p1":
            pref["p1_sampler"] = sam
            target_menu = build_p1_keyboard(user.id)
        else:
            pref["p2_sampler"] = sam
            target_menu = build_p2_keyboard(user.id)
        await query.edit_message_reply_markup(reply_markup=target_menu)
        return

    elif data.startswith("set_sched:"):
        _, target, sc = data.split(":")
        if target == "p1":
            pref["p1_scheduler"] = sc
            target_menu = build_p1_keyboard(user.id)
        else:
            pref["p2_scheduler"] = sc
            target_menu = build_p2_keyboard(user.id)
        await query.edit_message_reply_markup(reply_markup=target_menu)
        return

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

    prompt_text = update.message.text.strip()
    pref = get_user_pref(user.id)

    current_seed = (
        random.randint(1, 2**48 - 1)
        if pref["seed_mode"] == "random"
        else pref["fixed_seed"]
    )

    # Load base API graph template
    with open(TEMPLATE_PATH, "r", encoding="utf-8") as f:
        graph = json.load(f)

    # --- 1. Positive Prompt (Node 3) ---
    graph["3"]["inputs"]["text"] = prompt_text

    # --- 2. Shift AuraFlow (Node 15) ---
    if "15" in graph:
        graph["15"]["inputs"]["shift"] = pref["auraflow_shift"]

    # --- 3. Master Seed (Node 68) ---
    graph["68"]["inputs"]["seed"] = current_seed

    # --- 4. KSampler Pass 1 All Inputs (Node 7) ---
    if "7" in graph:
        graph["7"]["inputs"]["steps"] = pref["p1_steps"]
        graph["7"]["inputs"]["cfg"] = pref["p1_cfg"]
        graph["7"]["inputs"]["sampler_name"] = pref["p1_sampler"]
        graph["7"]["inputs"]["scheduler"] = pref["p1_scheduler"]
        graph["7"]["inputs"]["denoise"] = pref["p1_denoise"]

    # --- 5. KSampler Pass 2 All Inputs (Node 29) ---
    if "29" in graph:
        graph["29"]["inputs"]["steps"] = pref["p2_steps"]
        graph["29"]["inputs"]["cfg"] = pref["p2_cfg"]
        graph["29"]["inputs"]["sampler_name"] = pref["p2_sampler"]
        graph["29"]["inputs"]["scheduler"] = pref["p2_scheduler"]
        graph["29"]["inputs"]["denoise"] = pref["p2_denoise"]

    # File prefixes & remove UI slider
    if "14" in graph:
        graph["14"]["inputs"]["filename_prefix"] = f"zi-{current_seed}-1-base"
    if "32" in graph:
        graph["32"]["inputs"]["filename_prefix"] = f"zi-{current_seed}-2-hires"
    if "59" in graph:
        graph["59"]["inputs"]["filename_prefix"] = f"zi-{current_seed}-3-4k"

    graph.pop("38", None)
    target_save_node = "59"

    status_msg = await update.message.reply_text(
        f"⏳ **Mengirim antrean ke ComfyUI...**\n"
        f"• Seed: `{current_seed}` | Shift: `{pref['auraflow_shift']}`\n"
        f"• KSampler P1: `{pref['p1_steps']}` steps | `{pref['p1_cfg']}` cfg | `{pref['p1_denoise']}` denoise | `{pref['p1_sampler']}` | `{pref['p1_scheduler']}`\n"
        f"• KSampler P2: `{pref['p2_steps']}` steps | `{pref['p2_cfg']}` cfg | `{pref['p2_denoise']}` denoise | `{pref['p2_sampler']}` | `{pref['p2_scheduler']}`",
        parse_mode=ParseMode.MARKDOWN,
    )

    client_id = str(uuid.uuid4())
    ws_host = server_url.replace("http://", "").replace("https://", "")
    ws_url = f"ws://{ws_host}/ws?clientId={client_id}"

    prompt_id = None
    last_ui_update = 0.0
    last_text = ""
    current_stage_name = "Inisialisasi Antrean"

    node_stage_names = {
        "13": "Memuat Model Z-Image",
        "2":  "Memuat Text Encoder CLIP",
        "5":  "Memuat VAE",
        "69": "Membuat Base Latent Canvas",
        "3":  "Encoding Teks Prompt",
        "15": "AuraFlow Timestep Shift",
        "7":  "🎛️ KSampler Pass 1 (Base)",
        "9":  "🖼️ VAE Decode Pass 1",
        "14": "💾 Menyimpan Preview Pass 1",
        "70": "📐 Latent Upscale 6x (Bislerp)",
        "29": "🔬 KSampler Pass 2 (Hi-Res)",
        "31": "🖼️ VAE Decode Tiled Pass 2",
        "32": "💾 Menyimpan Hasil Pass 2",
        "50": "🧬 Memuat Model SeedVR2",
        "51": "Memuat VAE SeedVR2",
        "52": "📐 Image Resampling (Lanczos)",
        "53": "SeedVR2 Pre-Processing",
        "54": "🧩 VAE Encode Tiled SeedVR2",
        "55": "SeedVR2 Conditioning",
        "56": "✨ KSampler SeedVR2 (Restoration)",
        "57": "🖼️ VAE Decode Tiled 4K",
        "58": "🎨 Color Correction (Lab)",
        "59": "💾 Menyimpan Final 4K",
    }

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
                                    f"• **Tahap:** `{current_stage_name}`\n"
                                    f"`[{bar}]` **{val}/{max_v}** ({int(val/max_v*100 if max_v else 0)}%)\n"
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
                            elif node is not None:
                                current_stage_name = node_stage_names.get(str(node), f"Node #{node}")
                                now = asyncio.get_event_loop().time()
                                if now - last_ui_update > 1.2:
                                    new_text = (
                                        f"⏳ **Sedang Memproses...**\n"
                                        f"• **Tahap:** `{current_stage_name}`\n"
                                        f"• Seed: `{current_seed}`"
                                    )
                                    if new_text != last_text:
                                        try:
                                            await status_msg.edit_text(new_text, parse_mode=ParseMode.MARKDOWN)
                                            last_text = new_text
                                            last_ui_update = now
                                        except Exception:
                                            pass

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

    final_img_info = images[-1]
    filename = final_img_info["filename"]
    subfolder = final_img_info.get("subfolder", "")
    local_output_path = Path("output") / subfolder / filename

    real_w, real_h = 0, 0
    file_size_mb = 0.0
    if local_output_path.exists():
        with Image.open(local_output_path) as im:
            real_w, real_h = im.size
        file_size_mb = local_output_path.stat().st_size / (1024 * 1024)

    caption = (
        f"✨ **Render Selesai!**\n"
        f"• Resolusi: **{real_w}x{real_h}** ({file_size_mb:.2f} MB)\n"
        f"• Seed: `{current_seed}` | Shift: `{pref['auraflow_shift']}`\n"
        f"• KSampler P1: `{pref['p1_steps']}` steps | `{pref['p1_cfg']}` cfg | `{pref['p1_denoise']}` denoise | `{pref['p1_sampler']}` | `{pref['p1_scheduler']}`\n"
        f"• KSampler P2: `{pref['p2_steps']}` steps | `{pref['p2_cfg']}` cfg | `{pref['p2_denoise']}` denoise | `{pref['p2_sampler']}` | `{pref['p2_scheduler']}`\n"
    )

    try:
        with open(local_output_path, "rb") as f:
            # 1. Send fast photo preview
            await update.message.reply_photo(
                photo=f,
                caption=caption,
                parse_mode=ParseMode.MARKDOWN,
                read_timeout=120,
                write_timeout=120,
            )
            f.seek(0)
            # 2. Send uncompressed raw document with extended streaming window
            try:
                await update.message.reply_document(
                    document=f,
                    filename=filename,
                    caption=f"📁 File Asli: `{filename}`",
                    parse_mode=ParseMode.MARKDOWN,
                    read_timeout=300,
                    write_timeout=300,
                )
            except TimedOut:
                print("Raw document streaming delivered with background server ack.")
    except Exception as e:
        print(f"Delivery exception: {e}")


async def global_error_handler(update: object, context: ContextTypes.DEFAULT_TYPE):
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
    print("🤖 Telegram Bot Studio Client (Clean Rebuild)")
    print(f"   Server Target : {config['comfyui']['server_url']}")
    print("   Exposed Scope : Shift, Seed, KSampler P1 & P2 All Inputs")
    print("==================================================")

    app.run_polling()


if __name__ == "__main__":
    main()
