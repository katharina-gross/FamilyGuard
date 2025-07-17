import os
import io
import json
import re
from typing import Dict, Union
from PIL import Image
import google.generativeai as genai


genai.configure(api_key="AIzaSyCVF6gWFxqKduhvNXUWAh0fPB4V-o0mRy4")


def moderate_screenshot(
    image: Union[str, bytes],
    categories: list[str] | None = None
) -> dict:
    """Принимает путь к файлу или байты изображения, отправляет его в
    `gemini-1.5-flash` и возвращает JSON‑словарь с полями
    `{status, category, confidence}`.

    Параметры
    ----------
    image : str | bytes
        Путь к изображению (PNG/JPEG) **или** уже прочитанные байты.

    Возвращает
    ---------
    dict
        Дикт с ключами `status, category, confidence` либо поле `error` при сбое.
    """

    img_part = _prepare_image(image)
    if categories:
        prompt = (
            "You're the moderator of parental control. "
            "Consider any form of cruelty: porn, violence, bullying, casinos, betting; "
            "in social networks — mats and insults. Combine forbidden topics."
            f" these are the categories you should pack {categories} into."
            "Check the photos in detail, and don't embellish anything."
            "Also you should create parameter 'confidence' that is a float between 0 and 1."
            "Return only JSON like {\"category\",\"confidence\"}."
        )
    else:
        prompt = (
            "You're the moderator of parental control. "
            "Consider any form of cruelty: porn, violence, bullying, casinos, betting; "
            "in social networks — mats and insults. Combine forbidden topics."
            "Then give name to category of this photo"
            "this is what it looks like: category: gambling!"
            "Check the photos in detail, and don't embellish anything."
            "Also you should create parameter 'confidence' that is a float between 0 and 1."
            "Return only JSON like {\"category\",\"confidence\"}."
        )

    model = genai.GenerativeModel("gemini-1.5-flash")
    resp = model.generate_content([prompt, img_part],
                                  generation_config={"temperature": 0.1, "max_output_tokens": 64})

    raw = resp.text.strip()
    return _extract_json(raw)

_MAX_GOOGLE_BYTES = 7_000_000


def _prepare_image(src: Union[str, bytes]) -> dict:
    """
    Делает «разумное» сжатие: 1024 px + JPEG 70.
    Если всё ещё >7 МБ – уменьшает качество/размер циклом.
    """
    if isinstance(src, str):
        with open(src, "rb") as f:
            src = f.read()

    img = Image.open(io.BytesIO(src)).convert("RGB")

    max_w   = 1024
    quality = 70

    while True:
        buf = io.BytesIO()
        w    = min(max_w, img.width)
        h    = int(img.height * w / img.width)
        img.resize((w, h), Image.LANCZOS).save(buf, format="JPEG",
                                               quality=quality, optimize=True)
        data = buf.getvalue()

        if len(data) <= _MAX_GOOGLE_BYTES:
            break

        if quality > 40:
            quality -= 10 
        elif w > 300:
            max_w = int(max_w * 0.8)  
        else:
            raise ValueError("Картинку не удалось ужать до 7 МБ")

    return {"mime_type": "image/jpeg", "data": data}


def _extract_json(text: str) -> Dict[str, str]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        return json.loads(m.group()) if m else {"error": "JSON not found"}


__all__ = ["moderate_screenshot"]
