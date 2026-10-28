"""
LLM 模块:用 Qwen2.5-7B-Instruct(4bit 量化)把用户的自然语言指令,
转成针对每个分割区域的语义标签(亮度/色温/色相/对比度的方向与强度),
再由 Python 代码按固定公式转换成数值调整(rgb delta + 对比度 delta)。

设计说明:
小模型在"选对区域 + 判断方向"这类语义任务上表现不错,但在"记住并套用
r/g/b 加减公式做算术"上很不可靠(容易把暖/冷、r/b 搞反或串位)。
所以这里让 LLM 只输出分类标签(brighter/darker/warmer/cooler/yellow/...),
真正的数值换算固定写在代码里,保证公式永远正确。

模型使用 7B 4bit 量化(bitsandbytes nf4),比 3B 全精度指令遵循能力更强,
显存占用与 3B-fp16 相近(约 5~6GB)。
"""
import os
from pathlib import Path
import json
import re
import torch

# An explicit override allows smaller local models; no silent replacement of the default.
LLM_REPO = os.environ.get("PIE_LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct")
LLM_PATH = LLM_REPO
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DELTA_KEYS = ("r", "g", "b", "contrast")

# 调试开关:True 时会把 LLM 原始输出、解析后的 JSON 都打印出来,方便排查"输出解析成功但内容全是 none"这类问题
DEBUG_LLM = os.environ.get("PIE_DEBUG_LLM", "0") == "1"

# 强度标签 -> 数值幅度
INTENSITY_MAP = {
    "none": 0.0,
    "slight": 0.08,
    "normal": 0.13,
    "strong": 0.20,
    "extreme": 0.28,
}

VALID_BRIGHTNESS = {"brighter", "darker", "none"}
VALID_WARMTH = {"warmer", "cooler", "none"}
VALID_CONTRAST = {"more", "less", "none"}
VALID_INTENSITY = set(INTENSITY_MAP.keys())

# 色相偏移方向 -> (r方向, g方向, b方向) 的单位符号,乘以强度幅度后叠加到 delta 上
HUE_DELTA_DIRECTION = {
    "red":    (+1, -1, -1),
    "orange": (+1, +0.3, -1),
    "yellow": (+1, +1, -1),
    "green":  (-1, +1, -1),
    "cyan":   (-1, +1, +1),
    "blue":   (-1, -1, +1),
    "purple": (+1, -1, +1),
    "pink":   (+1, -0.3, +1),
    "none":   (0, 0, 0),
}
VALID_HUE = set(HUE_DELTA_DIRECTION.keys())


def load_llm():
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    local_only = os.environ.get("PIE_MODEL_LOCAL_FILES_ONLY", "0") == "1"
    tokenizer = AutoTokenizer.from_pretrained(LLM_PATH, local_files_only=local_only)
    if DEVICE == "cuda":
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            LLM_PATH, local_files_only=local_only,
            quantization_config=quant_config, device_map="auto",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            LLM_PATH, local_files_only=local_only, torch_dtype=torch.float32,
        ).to(DEVICE)
    return tokenizer, model.eval()


def build_skeleton(region_names):
    """{"sky": null, "person": null, ..., "global": null}"""
    skeleton = {name: None for name in sorted(set(region_names))}
    skeleton["global"] = None
    return skeleton


SYSTEM_PROMPT = """你是一个图像调色助手。你会收到:
1. 一张图片里出现的区域类别列表(JSON骨架,key是区域名,value目前是null)
2. 用户的调色/编辑指令(自然语言)

你的任务:根据用户指令,判断每个区域应该往哪个方向调整,只需要给出分类标签,
不需要计算具体数值(数值会由程序自动换算,你不用管公式)。

输出格式(每个区域一个对象):
{"区域名": {"brightness": "...", "warmth": "...", "hue": "...", "contrast": "...", "intensity": "..."}, ...}

字段取值范围(必须严格从下面的选项中选,不能自己发明新词):
- brightness(亮度方向): "brighter"(变亮) | "darker"(变暗) | "none"(不涉及亮度)
- warmth(冷暖方向,泛指整体偏暖/偏冷氛围): "warmer"(变暖) | "cooler"(变冷) | "none"(不涉及冷暖)
- hue(色相偏向,指明确偏向某一种具体颜色): "red"(偏红) | "orange"(偏橙) | "yellow"(偏黄) | "green"(偏绿) | "cyan"(偏青) | "blue"(偏蓝) | "purple"(偏紫) | "pink"(偏粉) | "none"(不涉及具体色相)
- contrast(对比度方向): "more"(对比度变大/更通透/更锐利) | "less"(对比度变小/更柔和/更平) | "none"(不涉及对比度)
- intensity(调整强度,几个字段共用同一个强度): "slight"(稍微/一点点) | "normal"(默认,没有程度修饰词) | "strong"(明显/更/较强) | "extreme"(非常/极端/极其/大幅/强烈/夸张)

intensity 判断提示:只要用户用了任何"很/非常/极/大幅/强烈/爆表/拉满/夸张"等强化词,一律选 "extreme",不要因为没见过这个具体的词就选成 "normal"。方向判断(warmer/darker等)和强度判断(intensity)是两件独立的事,不要因为强度词不常见就连方向也不敢判断。

warmth 和 hue 的区别(很重要,选错会导致效果完全不对):
- 用户说"暖一点"、"暖色调"、"更温暖"这种笼统的冷暖描述 → 用 warmth="warmer"(warmth 字段),hue 填 "none"。
- 用户说"冷一点"、"冷色调" → 用 warmth="cooler",hue 填 "none"。
- 用户明确点名某种颜色,比如"黄一点"、"红一点"、"偏绿"、"紫一点"、"更蓝"、"粉一点" → 用 hue 字段填对应颜色,warmth 填 "none"。
- 两者可以同时出现,比如"草地黄一点,同时天空冷一点",分别按各自区域填对应字段。

"global" 区域表示整张图的整体调整(全局指令,例如"整体提亮"、"整体调暖"、"整体偏黄"、"画面极端暖"这种没有点名任何具体区域的指令),
global 支持同样的 contrast 字段,例如全局增加对比度时填 "more"。
只要指令没有点名任何具体区域名(building/sky/tree/grass/mountain/lake等),就只填 global,其余所有区域全部填 "none"。

规则:
- 用户没有提到、也和指令无关的区域,所有字段都填 "none"(intensity 填 "none" 或 "normal" 都可以,因为方向是 none 时强度不会生效)。
- 只输出一个合法 JSON 对象,不要输出任何解释、注释或 Markdown 代码块标记。
- JSON 的 key 必须和骨架里的 key 完全一致,不能新增或删除 key。

示例:
用户指令:"天空调暖一点,并且稍微提亮"
骨架:{"sky": null, "grass": null, "global": null}
输出:
{"sky": {"brightness": "brighter", "warmth": "warmer", "hue": "none", "contrast": "none", "intensity": "slight"}, "grass": {"brightness": "none", "warmth": "none", "hue": "none", "contrast": "none", "intensity": "none"}, "global": {"brightness": "none", "warmth": "none", "hue": "none", "contrast": "none", "intensity": "none"}}

用户指令:"整个画面暖一点"
骨架:{"sky": null, "grass": null, "global": null}
输出:
{"sky": {"brightness": "none", "warmth": "none", "hue": "none", "contrast": "none", "intensity": "none"}, "grass": {"brightness": "none", "warmth": "none", "hue": "none", "contrast": "none", "intensity": "none"}, "global": {"brightness": "none", "warmth": "warmer", "hue": "none", "contrast": "none", "intensity": "normal"}}

用户指令:"草地黄一点"
骨架:{"sky": null, "grass": null, "global": null}
输出:
{"sky": {"brightness": "none", "warmth": "none", "hue": "none", "contrast": "none", "intensity": "none"}, "grass": {"brightness": "none", "warmth": "none", "hue": "yellow", "contrast": "none", "intensity": "slight"}, "global": {"brightness": "none", "warmth": "none", "hue": "none", "contrast": "none", "intensity": "none"}}

用户指令:"画面极端暖"
骨架:{"sky": null, "grass": null, "global": null}
输出:
{"sky": {"brightness": "none", "warmth": "none", "hue": "none", "contrast": "none", "intensity": "none"}, "grass": {"brightness": "none", "warmth": "none", "hue": "none", "contrast": "none", "intensity": "none"}, "global": {"brightness": "none", "warmth": "warmer", "hue": "none", "contrast": "none", "intensity": "extreme"}}
"""


def _extract_json(text):
    """Read the first complete JSON object, tolerating surrounding prose/fences."""
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            value, _ = decoder.raw_decode(text[match.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("The language model response does not contain a complete JSON object.")


def _labels_to_delta(label_obj, is_global):
    """把 LLM 输出的分类标签,按固定公式换算成 r/g/b/contrast 数值。"""
    if not isinstance(label_obj, dict):
        label_obj = {}

    brightness = label_obj.get("brightness")
    warmth = label_obj.get("warmth")
    hue = label_obj.get("hue")
    contrast_dir = label_obj.get("contrast")
    intensity = label_obj.get("intensity")

    if brightness not in VALID_BRIGHTNESS:
        brightness = "none"
    if warmth not in VALID_WARMTH:
        warmth = "none"
    if hue not in VALID_HUE:
        hue = "none"
    if contrast_dir not in VALID_CONTRAST:
        contrast_dir = "none"
    if intensity not in VALID_INTENSITY:
        intensity = "normal"

    amount = INTENSITY_MAP[intensity]

    r = g = b = c = 0.0

    if brightness == "brighter":
        r += amount; g += amount; b += amount
    elif brightness == "darker":
        r -= amount; g -= amount; b -= amount

    if warmth == "warmer":
        r += amount; b -= amount
    elif warmth == "cooler":
        r -= amount; b += amount

    dr, dg, db = HUE_DELTA_DIRECTION[hue]
    r += dr * amount
    g += dg * amount
    b += db * amount

    if contrast_dir == "more":
        c += amount
    elif contrast_dir == "less":
        c -= amount

    r = max(-0.3, min(0.3, r))
    g = max(-0.3, min(0.3, g))
    b = max(-0.3, min(0.3, b))
    c = max(-0.3, min(0.3, c))

    return {"r": r, "g": g, "b": b, "contrast": c}


def _sanitize(filled, skeleton):
    """校验/补全字段,把 LLM 的分类标签换算成最终数值,防止漏填或格式不对导致下游崩溃。"""
    if not isinstance(filled, dict) or not filled or not set(filled).issubset(skeleton):
        raise ValueError("The language model returned an empty or unknown-region plan.")
    options = {"brightness": VALID_BRIGHTNESS, "warmth": VALID_WARMTH,
               "hue": VALID_HUE, "contrast": VALID_CONTRAST, "intensity": VALID_INTENSITY}
    for values in filled.values():
        if not isinstance(values, dict) or not values:
            raise ValueError("The language model returned invalid region adjustments.")
        for field, value in values.items():
            if field not in options or not isinstance(value, str) or value not in options[field]:
                raise ValueError("The language model returned an unsupported adjustment label.")
    out = {}
    for key in skeleton:
        val = filled.get(key) if isinstance(filled, dict) else None
        out[key] = _labels_to_delta(val, is_global=(key == "global"))
    return out


@torch.no_grad()
def generate_region_json(user_instruction, region_names, tokenizer, model, max_new_tokens=800):
    skeleton = build_skeleton(region_names)
    user_content = (
        f"区域骨架:\n{json.dumps(skeleton, ensure_ascii=False)}\n\n"
        f"用户指令:\n{user_instruction}\n\n"
        f"请只输出填好的 JSON。"
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=1.0,
        pad_token_id=tokenizer.eos_token_id,
    )
    generated = output_ids[0][inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(generated, skip_special_tokens=True)

    if DEBUG_LLM:
        print(f"[debug] LLM 原始输出:\n{text}\n")

    try:
        parsed = _extract_json(text)
        result = _sanitize(parsed, skeleton)
    except (ValueError, TypeError) as error:
        raise RuntimeError("The language model could not produce a valid enhancement plan; try a simpler preference.") from error

    if DEBUG_LLM and all(all(v == 0.0 for v in region.values()) for region in result.values()):
        print("[debug] 警告:所有区域换算结果均为 0,请检查上面打印的原始输出和标签 JSON,确认模型是否理解了指令。")

    return result
