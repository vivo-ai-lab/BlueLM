#!/usr/bin/env python
# -*- coding: utf-8 -*-
import json
import os
import platform

import torch

from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_ID = "vivo-ai/BlueLM-7B-Chat-32K"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map="cuda:0", torch_dtype=torch.bfloat16, trust_remote_code=True)
model.eval()

os_name = platform.system()
clear_command = "cls" if os_name == "Windows" else "clear"

tools = [
    {
        "name": "text-translation-en2zh",
        "description": "将输入的英文文本翻译成中文",
        "parameters": [{
            "name": "text",
            "description": "用户输入的英文文本",
            "required": 'True'
        }]
    },
    {
        "name": "text-address",
        "description": "针对中文的地址信息，识别出里面的元素",
        "parameters": [{
            "name": "text",
            "description": "用户输入的地址信息",
            "required": 'True'
        }]
    },
    {
        "name": "current-weather",
        "description": "根据给出的城市，查询即时天气信息",
        "parameters": [{
            "name": "city",
            "description": "用户输入的城市信息",
            "required": 'True'
        }]
    }
]


def init_history():
    content = "你是一个AI助手，尽你所能回答用户的问题，你可以使用的工具如下:\n<APIs>\n- "
    content += str("\n- ".join([str(i) for i in tools]))
    content += '\n</APIs>\n你需要根据用户问题，选择合适的工具，输出的格式为：\n{"answer":"给用户的回复","function_calls":[{"name":"函数名","parameters":{"参数名":"参数"}}]}\n如果不需要额外回复或者没有合适工具，则对应字段输出空。\n'
    return [
        {"role": "system", "content": content}
    ]


def build_prompt(history):
    prompt = ""
    for item in history:
        content = item["content"]
        if item["role"] == "system":
            prompt += "[|SYSTEM|]:"
            prompt += content
        if item["role"] == "user":
            prompt += "[|Human|]:"
            prompt += content
        if item["role"] == "assistant":
            prompt += "[|AI|]:"
            prompt += content
            prompt += "</s>"
        if item["role"] == "observation":
            prompt += "[|Function|]:"
            prompt += content
    prompt += "[|AI|]:"
    return prompt


def main():
    os.system(clear_command)
    role = "user"
    history = init_history()
    while True:
        query = input("\n用户:") if role == "user" else input("\n结果:")
        if query.strip() == "quit":
            break
        if query.strip() == "clear":
            history = init_history()
            os.system(clear_command)
            continue
        history.append({"role": role, "content": query.strip()})
        prompt = build_prompt(history)
        inputs = tokenizer(prompt, return_tensors="pt")["input_ids"]
        inputs = inputs.to("cuda:0")
        input_echo_len = len(inputs[0])
        pred = model.generate(inputs, max_new_tokens=2048, do_sample=False).cpu()
        generated_text = tokenizer.decode(pred[0][input_echo_len:], skip_special_tokens=True)
        print("\nBlueLM:\n")
        print(generated_text)
        history.append({"role": "assistant", "content": generated_text})

        try:
            res = json.loads(generated_text)
            if res["function_calls"] and len(res["function_calls"]) > 0:
                role = "observation"
            else:
                role = "user"

            try:
                if res["function_calls"] is None and res["answer"]:
                    history = history[:-1] + [{"role": "assistant", "content": res["answer"]}]
            except:
                pass
        except:
            role = "user"


if __name__ == '__main__':
    main()
