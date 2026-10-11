"""
Telegram Bot Sidecar Client for ComfyUI.
Comprehensive remote control: Turbo/Base, 11 Aspect Ratios, LoRA Strength,
Pipeline Modes (4K / Hi-Res / Draft), Samplers, Schedulers, Denoise, and Seed controls.
"""

import json
import logging
import os
import random
import asyncio
from pathlib import Path
import tomllib

import aiohttp
from telegram import (
    Update,
    BotCommand,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
)
from telegram.request import HTTPXRequest
from telegram.constants import ParseMode
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    ContextTypes,
    filters,
)

logging.basicConfig(
    format="%(asctime)s - [%(name)s] - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("ComfyTelegramBot")

CONFIG_PATH = Path("scripts/config.telegram.toml")
TEMPLATE_PATH = Path("scripts/z_image_api_template.json")

# In-memory user preferences
user_sessions = {}

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

DEFAULT_PREFERENCES = {
    "preset": "ZIT (Turbo - 8 Steps, CFG 1.0)",
    "dimensions": "864x1152 ( 3:4 )",
    "lora_strength": 0.2,
    "seed_mode": "random",  # "random" or "fixed"
    "fixed_seed": 351971767740904,
    "pipeline_mode": "full",  # "full" (4K SeedVR2), "hires" (Pass 2 only), "draft" (Pass 1 only)
    "sampler_pass1": "euler_ancestral",
    "sampler_pass2": "dpmpp_sde",
    "scheduler": "beta",
    "denoise_pass2": 0.45,
}


def load_config() -> dict:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Config file not found at '{CONFIG_PATH}'. "
            "Please copy 'scripts/config.telegram.example.toml' to 'scripts/config.telegram.toml' "
            "and fill in your bot token and user ID."
        )
    with open(CONFIG_PATH, "rb") as f:
        return tomllib.load(f)


def is_authorized(user_id: int, allowed_ids: list[int]) -> bool:
    return user_id in allowed_ids


def get_user_pref(user_id: int) -> dict:
    if user_id not in user_sessions:
        user_sessions[user_id] = DEFAULT_PREFERENCES.copy()
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
    pipe_labels = {"full": "💎 4K SeedVR2", "hires": "⚡ Hi-Res 1.5x", "draft": "🚀 Draft Pass 1"}

    mode_label = "Turbo (8s)" if is_turbo else "Base (28s)"
    pipe_label = pipe_labels.get(pref["pipeline_mode"], "4K")
    seed_label = "Random" if pref["seed_mode"] == "random" else f"🔒 {pref['fixed_seed']}"
    aspect_label = pref["dimensions"].split()[0]

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
            InlineKeyboardButton(f"🎲 Seed: {seed_label}", callback_data="menu_seed"),
            InlineKeyboardButton("🎛️ Advanced (Sampler)", callback_data="menu_advanced"),
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
        [InlineKeyboardButton(f"{'✅ ' if cur == 'full' else ''}💎 Full Studio 4K (Pass 1 + 2 + SeedVR2)", callback_data="set_pipe:full")],
        [InlineKeyboardButton(f"{'✅ ' if cur == 'hires' else ''}⚡ Fast Hi-Res (Pass 1 + 2 Tanpa SeedVR2)", callback_data="set_pipe:hires")],
        [InlineKeyboardButton(f"{'✅ ' if cur == 'draft' else ''}🚀 Draft Cepat (Pass 1 Base Saja)", callback_data="set_pipe:draft")],
        [InlineKeyboardButton("🔙 Kembali", callback_data="menu_main")],
    ]
    return InlineKeyboardMarkup(buttons)


def build_aspect_keyboard(user_id: int) -> InlineKeyboardMarkup:
    pref = get_user_pref(user_id)
    cur = pref["dimensions"]

    # Render in 2 columns for neatness
    buttons = []
    row = []
    for label, dim in ALL_ASPECT_RATIOS:
        active = "✅ " if dim == cur else ""
        # e.g. "3:4 Portrait (864x1152)" -> "3:4 Portrait"
        short_name = " ".join(label.split()[:2])
        row.append(InlineKeyboardButton(f"{active}{short_name}", callback_data=f"set_aspect:{dim}"))
        if len(row) == 2:
            buttons.append(row)
            row = []
    if row:
        buttons.append(row)

    buttons.append([InlineKeyboardButton("🔙 Kembali", callback_data="menu_main")])
    return InlineKeyboardMarkup(buttons)


def build_lora_keyboard(user_id: int) -> InlineKeyboardMarkup:
    pref = get_user_pref(user_id)
    cur = pref["lora_strength"]

    strengths = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    row1 = [
        InlineKeyboardButton(f"{'✅ ' if s == cur else ''}{s}{' (Off)' if s == 0 else ''}", callback_data=f"set_lora:{s}")
        for s in strengths[:3]
    ]
    row2 = [
        InlineKeyboardButton(f"{'✅ ' if s == cur else ''}{s}", callback_data=f"set_lora:{s}")
        for s in strengths[3:]
    ]
    buttons = [row1, row2, [InlineKeyboardButton("🔙 Kembali", callback_data="menu_main")]]
    return InlineKeyboardMarkup(buttons)


def build_seed_keyboard(user_id: int) -> InlineKeyboardMarkup:
    pref = get_user_pref(user_id)
    is_rand = pref["seed_mode"] == "random"

    buttons = [
        [InlineKeyboardButton(f"{'✅ ' if is_rand else ''}🎲 Mode Random (Acak Terus)", callback_data="set_seed_mode:random")],
        [InlineKeyboardButton(f"{'✅ ' if not is_rand else ''}🔒 Mode Kunci Seed ({pref['fixed_seed']})", callback_data="set_seed_mode:fixed")],
        [InlineKeyboardButton("🎲 Acak & Kunci Seed Baru", callback_data="set_seed_roll")],
        [InlineKeyboardButton("🔙 Kembali", callback_data="menu_main")],
    ]
    return InlineKeyboardMarkup(buttons)


def build_advanced_keyboard(user_id: int) -> InlineKeyboardMarkup:
    pref = get_user_pref(user_id)
    buttons = [
        [
            InlineKeyboardButton(f"P1 Sampler: {pref['sampler_pass1'][:8]}", callback_data="menu_sam_p1"),
            InlineKeyboardButton(f"P2 Sampler: {pref['sampler_pass2'][:8]}", callback_data="menu_sam_p2"),
        ],
        [
            InlineKeyboardButton(f"Scheduler: {pref['scheduler']}", callback_data="menu_sched"),
            InlineKeyboardButton(f"P2 Denoise: {pref['denoise_pass2']}", callback_data="menu_denoise"),
        ],
        [InlineKeyboardButton("🔙 Kembali ke Menu Utama", callback_data="menu_main")],
    ]
    return InlineKeyboardMarkup(buttons)


def build_denoise_keyboard(user_id: int) -> InlineKeyboardMarkup:
    pref = get_user_pref(user_id)
    cur = pref["denoise_pass2"]
    values = [0.35, 0.40, 0.45, 0.50, 0.55]
    row = [
        InlineKeyboardButton(f"{'✅ ' if v == cur else ''}{v}", callback_data=f"set_denoise:{v}")
        for v in values
    ]
    return InlineKeyboardMarkup([row, [InlineKeyboardButton("🔙 Kembali ke Advanced", callback_data="menu_advanced")]])


def build_sampler_keyboard(target: str, user_id: int) -> InlineKeyboardMarkup:
    pref = get_user_pref(user_id)
    cur = pref["sampler_pass1"] if target == "p1" else pref["sampler_pass2"]
    samplers = ["euler_ancestral", "dpmpp_sde", "euler", "res_multistep"]

    buttons = [
        [InlineKeyboardButton(f"{'✅ ' if s == cur else ''}{s}", callback_data=f"set_sam:{target}:{s}")]
        for s in samplers
    ]
    buttons.append([InlineKeyboardButton("🔙 Kembali ke Advanced", callback_data="menu_advanced")])
    return InlineKeyboardMarkup(buttons)


def build_scheduler_keyboard(user_id: int) -> InlineKeyboardMarkup:
    pref = get_user_pref(user_id)
    cur = pref["scheduler"]
    schedulers = ["beta", "simple", "ddim_uniform", "normal"]

    buttons = [
        [InlineKeyboardButton(f"{'✅ ' if s == cur else ''}{s}", callback_data=f"set_sched:{s}")]
        for s in schedulers
    ]
    buttons.append([InlineKeyboardButton("🔙 Kembali ke Advanced", callback_data="menu_advanced")])
    return InlineKeyboardMarkup(buttons)


def format_settings_text(user_id: int) -> str:
    pref = get_user_pref(user_id)
    pipe_desc = {
        "full": "💎 Full Studio 4K (Pass 1 + Pass 2 + SeedVR2 4K)",
        "hires": "⚡ Fast Hi-Res (Pass 1 + Pass 2 1.5x)",
        "draft": "🚀 Draft Cepat (Pass 1 Base Saja)",
    }
    seed_text = "Acak Tiap Generate" if pref["seed_mode"] == "random" else f"Terkunci ({pref['fixed_seed']})"
    return (
        "⚙️ **PENGATURAN STUDIO COMFYUI**\n\n"
        f"• **Model:** `{pref['preset']}`\n"
        f"• **Alur:** `{pipe_desc.get(pref['pipeline_mode'])}`\n"
        f"• **Ukuran:** `{pref['dimensions']}`\n"
        f"• **Kekuatan LoRA:** `{pref['lora_strength']}`\n"
        f"• **Seed:** `{seed_text}`\n"
        f"• **Sampler:** `P1: {pref['sampler_pass1']}` | `P2: {pref['sampler_pass2']}`\n"
        f"• **Scheduler:** `{pref['scheduler']}` | **P2 Denoise:** `{pref['denoise_pass2']}`\n"
    )


# ==========================================
# HANDLERS
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
        "• Ketik /settings untuk membuka panel kontrol lengkap (Mode, Ukuran, LoRA, Pipeline 4K, Sampler, Seed).\n"
        "• Ketik `/seed <angka>` untuk mengatur seed manual tertentu.\n"
        "• Langsung **kirim teks prompt** apa saja untuk mulai generate gambar!\n\n"
        "Trigger LoRA (`2s12an5`) akan otomatis disisipkan jika belum ada."
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


async def seed_command_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    allowed = context.bot_data["config"]["telegram"]["allowed_user_ids"]
    if not is_authorized(user.id, allowed):
        return

    pref = get_user_pref(user.id)
    if not context.args:
        await update.message.reply_text(
            f"ℹ️ Seed saat ini: `{pref['fixed_seed']}` ({pref['seed_mode']})\n"
            "Gunakan format: `/seed 12345678` untuk mengunci seed, atau `/seed random` untuk mengacak."
        )
        return

    arg = context.args[0].lower()
    if arg in ["random", "rand", "acak"]:
        pref["seed_mode"] = "random"
        await update.message.reply_text("🎲 Mode seed diubah ke **Random** (acak tiap kali generate).", parse_mode=ParseMode.MARKDOWN)
    elif arg.isdigit():
        pref["seed_mode"] = "fixed"
        pref["fixed_seed"] = int(arg)
        await update.message.reply_text(f"🔒 Seed berhasil dikunci ke: `{pref['fixed_seed']}`", parse_mode=ParseMode.MARKDOWN)
    else:
        await update.message.reply_text("❌ Input salah! Masukkan angka seed atau 'random'.")


async def callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    user = query.from_user
    allowed = context.bot_data["config"]["telegram"]["allowed_user_ids"]
    if not is_authorized(user.id, allowed):
        return

    pref = get_user_pref(user.id)
    data = query.data

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
        pref["preset"] = (
            "ZI (Base - 28 Steps, CFG 4.0)"
            if "Turbo" in pref["preset"]
            else "ZIT (Turbo - 8 Steps, CFG 1.0)"
        )

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

    elif data == "menu_advanced":
        await query.edit_message_reply_markup(reply_markup=build_advanced_keyboard(user.id))
        return

    elif data == "menu_sam_p1":
        await query.edit_message_reply_markup(reply_markup=build_sampler_keyboard("p1", user.id))
        return

    elif data == "menu_sam_p2":
        await query.edit_message_reply_markup(reply_markup=build_sampler_keyboard("p2", user.id))
        return

    elif data.startswith("set_sam:"):
        _, target, sam = data.split(":")
        key = "sampler_pass1" if target == "p1" else "sampler_pass2"
        pref[key] = sam

    elif data == "menu_sched":
        await query.edit_message_reply_markup(reply_markup=build_scheduler_keyboard(user.id))
        return

    elif data.startswith("set_sched:"):
        pref["scheduler"] = data.split("set_sched:")[1]

    elif data == "menu_denoise":
        await query.edit_message_reply_markup(reply_markup=build_denoise_keyboard(user.id))
        return

    elif data.startswith("set_denoise:"):
        pref["denoise_pass2"] = float(data.split("set_denoise:")[1])

    elif data == "reset_settings":
        user_sessions[user.id] = DEFAULT_PREFERENCES.copy()

    # Re-render main menu
    await query.edit_message_text(
        format_settings_text(user.id),
        reply_markup=build_main_settings_keyboard(user.id),
        parse_mode=ParseMode.MARKDOWN,
    )


# ==========================================
# PROMPT EXECUTION ENGINE
# ==========================================

async def prompt_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    allowed = context.bot_data["config"]["telegram"]["allowed_user_ids"]
    if not is_authorized(user.id, allowed):
        return

    server_url = context.bot_data["config"]["comfyui"]["server_url"]
    ws_url = context.bot_data["config"]["comfyui"]["ws_url"]

    if not await check_comfy_online(server_url):
        await update.message.reply_text(
            "🔴 ComfyUI sedang offline!\nPastikan terminal ComfyUI di PC sudah dijalankan (`uv run main.py --cuda-device 0`)."
        )
        return

    prompt_text = update.message.text.strip()
    if not prompt_text.startswith("2s12an5"):
        prompt_text = f"2s12an5, {prompt_text}"

    pref = get_user_pref(user.id)

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

    # 2. Resolution Preset (Node 6)
    graph["6"]["inputs"]["dimensions"] = pref["dimensions"]

    # 3. Model & Sampler Switch (Node 13)
    graph["13"]["inputs"]["preset"] = pref["preset"]
    graph["13"]["inputs"]["sampler_pass1"] = pref["sampler_pass1"]
    graph["13"]["inputs"]["sampler_pass2"] = pref["sampler_pass2"]
    graph["13"]["inputs"]["scheduler"] = pref["scheduler"]

    # 4. Master Seed (Node 68) & Dynamic File Prefixes
    graph["68"]["inputs"]["seed"] = current_seed
    if "14" in graph:
        graph["14"]["inputs"]["filename_prefix"] = f"zi-{current_seed}-1-base"
    if "32" in graph:
        graph["32"]["inputs"]["filename_prefix"] = f"zi-{current_seed}-2-hires"
    if "59" in graph:
        graph["59"]["inputs"]["filename_prefix"] = f"zi-{current_seed}-3-4k"

    # 5. LoRA Strength (Node 67)
    lora_strength = pref["lora_strength"]
    graph["67"]["inputs"]["lora_1"] = {
        "on": lora_strength > 0,
        "lora": "NIceAsians_Zimage.safetensors",
        "strength": lora_strength,
    }

    # 6. Adapt Pipeline Mode (Prune unneeded execution nodes)
    pipeline_mode = pref["pipeline_mode"]
    target_save_node = "59"  # Default 4K

    if pipeline_mode == "draft":
        # Disconnect Pass 2 and SeedVR2 by removing their outputs
        for nid in ["29", "30", "31", "32", "50", "51", "52", "53", "54", "55", "56", "57", "58", "59"]:
            graph.pop(nid, None)
        target_save_node = "14"  # Save 1: Base
    elif pipeline_mode == "hires":
        # Keep Pass 1 and 2, but remove SeedVR2 (Node 50-59)
        for nid in ["50", "51", "52", "53", "54", "55", "56", "57", "58", "59"]:
            graph.pop(nid, None)
        target_save_node = "32"  # Save 2: Hi-Res

    mode_label = {
        "full": "💎 4K SeedVR2 Studio",
        "hires": "⚡ Fast Hi-Res 1.5x",
        "draft": "🚀 Draft Cepat Pass 1",
    }[pipeline_mode]

    status_msg = await update.message.reply_text(
        f"⏳ **Mengirim antrean ke ComfyUI...**\n"
        f"• Alur: **{mode_label}**\n"
        f"• Seed: `{current_seed}`\n"
        f"• Ukuran: `{pref['dimensions']}`\n"
        f"• LoRA: `{lora_strength}`",
        parse_mode=ParseMode.MARKDOWN,
    )

    client_id = f"tg_{user.id}_{random.randint(1000, 9999)}"
    prompt_payload = {"prompt": graph, "client_id": client_id}

    async with aiohttp.ClientSession() as session:
        try:
            async with session.ws_connect(f"{ws_url}?clientId={client_id}") as ws:
                async with session.post(f"{server_url}/prompt", json=prompt_payload) as resp:
                    if resp.status != 200:
                        err = await resp.text()
                        await status_msg.edit_text(f"❌ Gagal mengirim prompt:\n`{err[:200]}`")
                        return
                    prompt_res = await resp.json()
                    prompt_id = prompt_res.get("prompt_id")

                last_progress_edit = 0
                node_names = {
                    "7": "Pass 1 (Base Latent)",
                    "9": "VAE Decode (Pass 1)",
                    "29": "Pass 2 (Hi-Res Denoise)",
                    "31": "VAE Decode Tiled (Pass 2)",
                    "56": "Pass 3 (SeedVR2 4K Ultra HD)",
                    "57": "VAE Decode Tiled (SeedVR2 4K)",
                }

                async for msg in ws:
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        data = json.loads(msg.data)
                        msg_type = data.get("type")

                        if msg_type == "progress":
                            pdata = data.get("data", {})
                            value = pdata.get("value", 0)
                            max_val = pdata.get("max", 0)
                            node_id = str(pdata.get("node", ""))
                            stage = node_names.get(node_id, f"Node {node_id}")

                            now = asyncio.get_event_loop().time()
                            if now - last_progress_edit > 1.2:
                                try:
                                    await status_msg.edit_text(
                                        f"⚙️ **Sedang memproses:**\n"
                                        f"• Tahap: **{stage}**\n"
                                        f"• Progres: `{value}/{max_val} steps`\n"
                                        f"• Seed: `{current_seed}`",
                                        parse_mode=ParseMode.MARKDOWN,
                                    )
                                    last_progress_edit = now
                                except Exception:
                                    pass

                        elif msg_type == "executing":
                            edata = data.get("data", {})
                            if edata.get("prompt_id") == prompt_id and edata.get("node") is None:
                                break

        except Exception as e:
            await status_msg.edit_text(f"❌ Terjadi kesalahan saat memantau progres:\n`{str(e)}`")
            return

    # Fetch output from history
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{server_url}/history/{prompt_id}") as resp:
                history_data = await resp.json()

        outputs = history_data.get(prompt_id, {}).get("outputs", {})
        images = []
        if target_save_node in outputs and "images" in outputs[target_save_node]:
            images = outputs[target_save_node]["images"]
        else:
            # Fallback across all outputs
            for out in outputs.values():
                if "images" in out:
                    images.extend(out["images"])

        if not images:
            await status_msg.edit_text("✅ Generasi selesai, tetapi gambar tidak ditemukan di riwayat.")
            return

        final_img_info = images[-1]
        filename = final_img_info["filename"]
        subfolder = final_img_info.get("subfolder", "")
        img_type = final_img_info.get("type", "output")

        img_url = f"{server_url}/view?filename={filename}&subfolder={subfolder}&type={img_type}"
        local_output_path = Path("output") / subfolder / filename

        # Read actual image dimensions and file size
        real_w, real_h = 0, 0
        file_size_mb = 0.0
        if local_output_path.exists():
            from PIL import Image
            with Image.open(local_output_path) as im:
                real_w, real_h = im.size
            file_size_mb = local_output_path.stat().st_size / (1024 * 1024)

        await status_msg.edit_text("📤 Mengunggah gambar ke Telegram...")

        res_tag = f"{real_w}x{real_h} (4K Ultra HD)" if real_w >= 2000 else f"{real_w}x{real_h}"
        caption_text = (
            f"✨ **Generasi Selesai!**\n"
            f"• **Resolusi Asli:** `{res_tag}`\n"
            f"• **Ukuran File:** `{file_size_mb:.1f} MB`\n"
            f"• **Alur:** `{mode_label}`\n"
            f"• **Seed:** `{current_seed}`\n"
            f"• **LoRA:** `{lora_strength}` | **P2 Denoise:** `{pref['denoise_pass2']}`"
        )

        if local_output_path.exists():
            with open(local_output_path, "rb") as f:
                try:
                    await update.message.reply_photo(
                        photo=f,
                        caption=caption_text,
                        parse_mode=ParseMode.MARKDOWN,
                    )
                except Exception as pe:
                    logger.warning(f"Gagal mengirim photo preview: {pe}")

                f.seek(0)
                await update.message.reply_document(
                    document=f,
                    filename=filename,
                    caption=f"📄 **File Asli 4K:** `{filename}`",
                    parse_mode=ParseMode.MARKDOWN,
                )
        else:
            async with aiohttp.ClientSession() as session:
                async with session.get(img_url) as resp:
                    img_bytes = await resp.read()
                    await update.message.reply_photo(
                        photo=img_bytes,
                        caption=caption_text,
                        parse_mode=ParseMode.MARKDOWN,
                    )

        await status_msg.delete()

    except Exception as e:
        await status_msg.edit_text(f"❌ Gagal mengirim gambar final:\n`{str(e)}`")


async def post_init(application: Application):
    commands = [
        BotCommand("settings", "Buka panel kontrol model, rasio, LoRA & sampler"),
        BotCommand("seed", "Atur seed manual atau acak"),
        BotCommand("start", "Cek status koneksi server ComfyUI"),
    ]
    await application.bot.set_my_commands(commands)
    logger.info("Daftar command resmi berhasil didaftarkan ke Telegram.")


def main():
    config = load_config()
    token = config["telegram"]["bot_token"]

    # Gunakan read_timeout & write_timeout = None (tanpa batas waktu buatan)
    # Membiarkan file besar (4K ~8-10 MB) streaming sampai tuntas tanpa false alarm timeout
    request = HTTPXRequest(
        connect_timeout=30.0,
        read_timeout=None,
        write_timeout=None,
        pool_timeout=60.0,
    )

    app = Application.builder().token(token).request(request).post_init(post_init).build()
    app.bot_data["config"] = config

    app.add_handler(CommandHandler("start", start_handler))
    app.add_handler(CommandHandler("settings", settings_handler))
    app.add_handler(CommandHandler("seed", seed_command_handler))
    app.add_handler(CallbackQueryHandler(callback_handler))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, prompt_handler))

    logger.info("Bot Telegram ComfyUI aktif dengan menu pengaturan lengkap...")
    app.run_polling()


if __name__ == "__main__":
    main()
