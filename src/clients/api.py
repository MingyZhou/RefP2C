import json
from typing import Dict, List, Optional

from src.clients.my_client import openai_client

client = openai_client


def get_response(prompt, model="gpt-4o-mini", system_prompt_extra=None, temp0=False):
    messages = [{"role": "user", "content": prompt}]
    system_inst = "You are a helpful assistant."
    if system_prompt_extra:
        system_inst += f"\n{system_prompt_extra}"
    system_msg = {"role": "system", "content": system_inst}

    api_params = {
        "model": model,
        "messages": messages,
    }

    if "o1" not in model:
        api_params["messages"].append(system_msg)
    if "o3-mini" in model:
        api_params["reasoning_effort"] = "high"
    if "r1" in model.lower():
        api_params["max_tokens"] = 16384
    if temp0:
        api_params["temperature"] = 0

    chat_completion = client.chat.completions.create(**api_params)

    content = chat_completion.choices[0].message.content

    return content


def get_response_format(
    prompt, format=None, model="gpt-4o-mini", system_prompt_extra=None
):
    messages = [{"role": "user", "content": prompt}]

    system_inst = """
        Your are a helpful assistant.
    """
    system_msg = {"role": "system", "content": system_inst}
    if "o1" not in model:
        messages.append(system_msg)

    completion = client.beta.chat.completions.parse(
        model=model,
        messages=messages,
        response_format=format,
    )
    content = completion.choices[0].message.content
    content = json.loads(content)
    return content


def get_multi_turn_response(
    messages: List[Dict[str, str]],
    new_user_message: str,
    model: str = "gpt-4o-mini",
    system_prompt_extra: Optional[str] = None,
    temp0: bool = False,
) -> str:
    system_inst = "You are a helpful assistant."
    if system_prompt_extra:
        system_inst += f"\n{system_prompt_extra}"

    if "o1" not in model:
        if not messages or messages[0].get("role") != "system":
            messages.insert(0, {"role": "system", "content": system_inst})

    messages.append({"role": "user", "content": new_user_message})

    api_params = {
        "model": model,
        "messages": messages,
    }

    if "o3-mini" in model:
        api_params["reasoning_effort"] = "high"
    if "r1" in model.lower():
        api_params["max_tokens"] = 16384
    if temp0:
        api_params["temperature"] = 0
    chat_completion = client.chat.completions.create(**api_params)

    content = chat_completion.choices[0].message.content
    messages.append({"role": "assistant", "content": content})

    return content


def get_response_from_messages(
    messages: List[Dict[str, str]], model: str = "gpt-4o-mini", temp0=False
):
    system_inst = "You are a helpful assistant."
    if "o1" not in model:
        if not messages or messages[0].get("role") != "system":
            messages.insert(0, {"role": "system", "content": system_inst})
    api_params = {
        "model": model,
        "messages": messages,
    }

    if "o3-mini" in model:
        api_params["reasoning_effort"] = "high"
    if "r1" in model.lower():
        api_params["max_tokens"] = 16384
    if temp0:
        api_params["temperature"] = 0

    chat_completion = client.chat.completions.create(**api_params)
    content = chat_completion.choices[0].message.content
    return content
