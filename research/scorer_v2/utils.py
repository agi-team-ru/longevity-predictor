import openai
from typing import List, Dict, Any
import json

def get_model_response(
    prompt: str,
    model_name: str = "model_base"
) -> str:
    client = openai.OpenAI()
    # Формируем сообщение для модели
    messages = [
        {
            "role": "user",
            "content": prompt
        }
    ]
    response = client.chat.completions.create(
        model="model_base",
        messages=messages
    )
    # Извлекаем текст ответа
    if hasattr(response, "choices") and len(response.choices) > 0:
        message = response.choices[0].message
        if hasattr(message, "content"):
            return message.content
    return ""

def extract_json_from_text(text: str) -> List[str]:
    start_json_char = "{"
    decoder = json.JSONDecoder(strict=False)
    pos = 0
    ret: List[str] = []
    while True:
        start_char_pos = text.find(start_json_char, pos)
        if start_char_pos < 0:
            break
        try:
            result, index = decoder.raw_decode(text[start_char_pos:])
            pos = start_char_pos + index
            ret.append(json.dumps(result, ensure_ascii=False))
        except ValueError:
            pos = start_char_pos + 1
    return ret
