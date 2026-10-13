"""
Automated Test Suite for Telegram Bot Sidecar Client.
Verifies preference synchronization, keyboard builders, callback state transitions,
/set text command parser, and ComfyUI backend prompt graph validation.
"""

import sys
import json
import asyncio
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from unittest.mock import MagicMock, AsyncMock, patch

# Ensure utf-8
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

import scripts.telegram_bot as tb
import execution
import nodes
import server


def test_authorization():
    print("1. [AUTH] Testing authorization whitelist security...")
    allowed = [111, 222]
    assert tb.is_authorized(111, allowed) is True
    assert tb.is_authorized(222, allowed) is True
    assert tb.is_authorized(999, allowed) is False
    print("   -> PASS: Whitelist security functions properly.")


def test_load_default_preferences():
    print("2. [PREF] Testing extraction from web workflow (user/default/workflows/z-image.json)...")
    pref = tb.load_default_preferences()
    
    # Must have 5 core exposed components
    assert "lora_strength" in pref, "Missing lora_strength"
    assert "auraflow_shift" in pref, "Missing auraflow_shift"
    assert "seed_mode" in pref, "Missing seed_mode"
    assert "fixed_seed" in pref, "Missing fixed_seed"
    assert "p1_steps" in pref, "Missing p1_steps"
    assert "p1_cfg" in pref, "Missing p1_cfg"
    assert "p1_sampler" in pref, "Missing p1_sampler"
    assert "p1_scheduler" in pref, "Missing p1_scheduler"
    assert "p1_denoise" in pref, "Missing p1_denoise"
    assert "p2_steps" in pref, "Missing p2_steps"
    assert "p2_cfg" in pref, "Missing p2_cfg"
    assert "p2_sampler" in pref, "Missing p2_sampler"
    assert "p2_scheduler" in pref, "Missing p2_scheduler"
    assert "p2_denoise" in pref, "Missing p2_denoise"

    print(f"   -> PASS: Successfully loaded defaults (LoRA={pref['lora_strength']}, Shift={pref['auraflow_shift']}, P1={pref['p1_steps']}s/{pref['p1_cfg']}cfg, P2={pref['p2_steps']}s/{pref['p2_denoise']}denoise).")


def test_keyboard_builders():
    print("3. [UI] Testing all inline keyboard builders and rendering...")
    uid = 99999
    
    kb_main = tb.build_main_settings_keyboard(uid)
    assert len(kb_main.inline_keyboard) >= 5, "Main keyboard incomplete"

    kb_lora = tb.build_lora_keyboard(uid)
    assert len(kb_lora.inline_keyboard) >= 2, "LoRA keyboard incomplete"

    kb_shift = tb.build_shift_keyboard(uid)
    assert len(kb_shift.inline_keyboard) >= 2, "Shift keyboard incomplete"

    kb_seed = tb.build_seed_keyboard(uid)
    assert len(kb_seed.inline_keyboard) >= 2, "Seed keyboard incomplete"

    kb_p1 = tb.build_p1_keyboard(uid)
    assert len(kb_p1.inline_keyboard) >= 4, "Pass 1 keyboard incomplete"

    kb_p1_steps = tb.build_p1_steps_keyboard(uid)
    assert len(kb_p1_steps.inline_keyboard) >= 2, "P1 Steps keyboard incomplete"

    kb_p1_cfg = tb.build_p1_cfg_keyboard(uid)
    assert len(kb_p1_cfg.inline_keyboard) >= 2, "P1 CFG keyboard incomplete"

    kb_p1_denoise = tb.build_p1_denoise_keyboard(uid)
    assert len(kb_p1_denoise.inline_keyboard) >= 2, "P1 Denoise keyboard incomplete"

    kb_p2 = tb.build_p2_keyboard(uid)
    assert len(kb_p2.inline_keyboard) >= 4, "Pass 2 keyboard incomplete"

    kb_p2_steps = tb.build_p2_steps_keyboard(uid)
    assert len(kb_p2_steps.inline_keyboard) >= 2, "P2 Steps keyboard incomplete"

    kb_p2_cfg = tb.build_p2_cfg_keyboard(uid)
    assert len(kb_p2_cfg.inline_keyboard) >= 2, "P2 CFG keyboard incomplete"

    kb_p2_denoise = tb.build_p2_denoise_keyboard(uid)
    assert len(kb_p2_denoise.inline_keyboard) >= 2, "P2 Denoise keyboard incomplete"

    kb_sam = tb.build_sampler_keyboard("p1", uid)
    assert len(kb_sam.inline_keyboard) >= 2, "Sampler keyboard incomplete"

    kb_sched = tb.build_scheduler_keyboard("p1", uid)
    assert len(kb_sched.inline_keyboard) >= 2, "Scheduler keyboard incomplete"

    # Verify settings text format without crash
    text = tb.format_settings_text(uid)
    assert "PANEL KONTROL STUDIO" in text
    print("   -> PASS: All 14 keyboard builders & text formatters verified error-free.")


async def test_callback_state_transitions():
    print("4. [CALLBACK] Testing state transitions via simulated callback queries...")
    uid = 88888
    pref = tb.get_user_pref(uid)

    async def simulate_cb(data_str):
        update = MagicMock()
        query = MagicMock()
        query.data = data_str
        query.from_user.id = uid
        query.answer = AsyncMock()
        query.edit_message_text = AsyncMock()
        query.edit_message_reply_markup = AsyncMock()
        update.callback_query = query
        context = MagicMock()
        context.bot_data = {"config": {"telegram": {"allowed_user_ids": [uid]}}}
        await tb.callback_handler(update, context)

    # 1. Test LoRA change
    await simulate_cb("set_lora:0.6")
    assert pref["lora_strength"] == 0.6, f"LoRA not updated: {pref['lora_strength']}"

    # 2. Test Shift change
    await simulate_cb("set_shift:4.0")
    assert pref["auraflow_shift"] == 4.0, f"Shift not updated: {pref['auraflow_shift']}"

    # 3. Test P1 steps change
    await simulate_cb("set_p1_steps:16")
    assert pref["p1_steps"] == 16, f"P1 steps not updated: {pref['p1_steps']}"

    # 4. Test P1 cfg change
    await simulate_cb("set_p1_cfg:2.5")
    assert pref["p1_cfg"] == 2.5, f"P1 cfg not updated: {pref['p1_cfg']}"

    # 5. Test P1 sampler & scheduler change
    await simulate_cb("set_sam:p1:dpmpp_sde")
    assert pref["p1_sampler"] == "dpmpp_sde", f"P1 sampler not updated: {pref['p1_sampler']}"

    await simulate_cb("set_sched:p1:beta")
    assert pref["p1_scheduler"] == "beta", f"P1 scheduler not updated: {pref['p1_scheduler']}"

    # 6. Test P2 denoise change
    await simulate_cb("set_p2_denoise:0.35")
    assert pref["p2_denoise"] == 0.35, f"P2 denoise not updated: {pref['p2_denoise']}"

    # 7. Test Reset settings
    await simulate_cb("reset_settings")
    defaults = tb.load_default_preferences()
    assert tb.get_user_pref(uid)["p1_steps"] == defaults["p1_steps"]

    print("   -> PASS: Callback actions successfully mutate user preferences.")


async def test_set_command_parser():
    print("5. [COMMAND] Testing /set custom parameter parser...")
    uid = 77777
    pref = tb.get_user_pref(uid)

    async def simulate_set(arg_list):
        update = MagicMock()
        update.effective_user.id = uid
        update.message.reply_text = AsyncMock()
        context = MagicMock()
        context.args = arg_list
        context.bot_data = {"config": {"telegram": {"allowed_user_ids": [uid]}}}
        await tb.set_command_handler(update, context)

    await simulate_set(["lora", "0.45"])
    assert pref["lora_strength"] == 0.45

    await simulate_set(["shift", "3.16"])
    assert pref["auraflow_shift"] == 3.16

    await simulate_set(["p1steps", "14"])
    assert pref["p1_steps"] == 14

    await simulate_set(["denoise", "0.33"])
    assert pref["p2_denoise"] == 0.33

    await simulate_set(["seed", "99887766"])
    assert pref["fixed_seed"] == 99887766
    assert pref["seed_mode"] == "fixed"

    print("   -> PASS: /set command correctly updates any arbitrary user parameter.")


async def test_comfy_graph_validation():
    print("6. [GRAPH & COMFY ENGINE] Validating generated prompt payload against ComfyUI core...")
    loop = asyncio.get_running_loop()
    s = server.PromptServer(loop, MagicMock())
    await nodes.init_extra_nodes()

    with open(tb.TEMPLATE_PATH, "r", encoding="utf-8") as f:
        graph = json.load(f)

    uid = 66666
    pref = tb.get_user_pref(uid)
    pref["p1_steps"] = 10
    pref["p1_cfg"] = 2.0
    pref["p1_sampler"] = "euler_ancestral"
    pref["p1_scheduler"] = "beta"
    pref["p1_denoise"] = 1.0
    pref["p2_steps"] = 8
    pref["p2_cfg"] = 1.0
    pref["p2_sampler"] = "dpmpp_sde"
    pref["p2_scheduler"] = "beta"
    pref["p2_denoise"] = 0.38
    pref["auraflow_shift"] = 5.0
    pref["lora_strength"] = 0.7

    current_seed = 445566
    prompt_text = "2s12an5, ultra realistic test photograph"

    # Assemble graph exactly as telegram_bot.py does
    graph["3"]["inputs"]["text"] = prompt_text
    graph["15"]["inputs"]["shift"] = pref["auraflow_shift"]
    graph["67"]["inputs"]["lora_1"] = {
        "on": True,
        "lora": "NIceAsians_Zimage.safetensors",
        "strength": pref["lora_strength"],
        "strengthTwo": pref["lora_strength"],
    }
    graph["68"]["inputs"]["seed"] = current_seed

    graph["7"]["inputs"]["steps"] = pref["p1_steps"]
    graph["7"]["inputs"]["cfg"] = pref["p1_cfg"]
    graph["7"]["inputs"]["sampler_name"] = pref["p1_sampler"]
    graph["7"]["inputs"]["scheduler"] = pref["p1_scheduler"]
    graph["7"]["inputs"]["denoise"] = pref["p1_denoise"]

    graph["29"]["inputs"]["steps"] = pref["p2_steps"]
    graph["29"]["inputs"]["cfg"] = pref["p2_cfg"]
    graph["29"]["inputs"]["sampler_name"] = pref["p2_sampler"]
    graph["29"]["inputs"]["scheduler"] = pref["p2_scheduler"]
    graph["29"]["inputs"]["denoise"] = pref["p2_denoise"]

    graph.pop("38", None)

    valid, errors, executed, extra = await execution.validate_prompt("test-tg-run", graph, None)
    assert valid is True, f"Prompt validation failed: {errors}"
    assert errors is None, f"Prompt has errors: {errors}"
    print(f"   -> PASS: ComfyUI backend validated prompt graph with 0 errors (valid={valid}).")


async def run_all_tests():
    print("==================================================")
    print("🧪 MENJALANKAN TEST SUITE TELEGRAM BOT COMFYUI")
    print("==================================================")
    test_authorization()
    test_load_default_preferences()
    test_keyboard_builders()
    await test_callback_state_transitions()
    await test_set_command_parser()
    await test_comfy_graph_validation()
    print("==================================================")
    print("🎉 SEMUA TEST BERHASIL (6/6 SUITE 100% PASS)!")
    print("==================================================")


if __name__ == "__main__":
    asyncio.run(run_all_tests())
