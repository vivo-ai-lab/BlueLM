import os
import platform
import torch
import readline
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer

DEVICE = "cuda:0"
MODEL_ID = "vivo-ai/BlueLM-7B-Chat"
MAX_NEW_TOKENS = 512
REPETITION_PENALTY = 1.1

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map=DEVICE, torch_dtype=torch.bfloat16,
                                             trust_remote_code=True)
model = model.eval()

os_name = platform.system()
clear_command = "cls" if os_name == "Windows" else "clear"


class BlueLMStreamer(TextStreamer):
    def __init__(self, tokenizer: "AutoTokenizer"):
        self.tokenizer = tokenizer
        self.tokenIds = []
        self.history = []
        self.prompt = ""
        self.response = ""
        self.first = True

    def put(self, value):
        if self.first:
            self.first = False
            return
        self.tokenIds.append(value.item())
        text = tokenizer.decode(self.tokenIds, skip_special_tokens=True)
        if len(self.tokenIds) % 4 == 0 and text and text[-1] != "�":
            self.build_display(self.history, self.prompt, text)

    def end(self):
        self.first = True
        text = tokenizer.decode(self.tokenIds, skip_special_tokens=True)
        self.response = text
        self.build_display(self.history, self.prompt, text)
        self.tokenIds = []

    def build_display(self, history, prompt, cur_text):
        info_str = "欢迎使用 BlueLM-7B 模型，输入内容即可进行对话，clear 清空对话历史，quit 终止程序"
        for query, response in history:
            info_str += f"\n\nUser：{query}"
            info_str += f"\nBlueLM-7B：{response}"
        info_str += f"\n\nUser：{prompt}"
        info_str += f"\nBlueLM-7B：{cur_text}"
        os.system(clear_command)
        print(info_str)


def build_prompt(history, prompt):
    res = ""
    for query, response in history:
        res += f"[|Human|]:{query}[|AI|]:{response}</s>"
    res += f"[|Human|]:{prompt}[|AI|]:"
    return res


def main():
    os.system(clear_command)
    history = []
    print("欢迎使用 BlueLM-7B 模型，输入内容即可进行对话，clear 清空对话历史，quit 终止程序")
    streamer = BlueLMStreamer(tokenizer=tokenizer)
    while True:
        query = input("\nUser：")
        if query.strip() == "quit":
            break
        if query.strip() == "clear":
            history = []
            os.system(clear_command)
            print("欢迎使用 BlueLM-7B 模型，输入内容即可进行对话，clear 清空对话历史，quit 终止程序")
            continue
        streamer.history = history
        streamer.prompt = query
        prompt = build_prompt(history=history, prompt=query.strip())
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = inputs.to(DEVICE)
        input_ids = inputs["input_ids"]
        model.generate(input_ids=input_ids, max_new_tokens=MAX_NEW_TOKENS, repetition_penalty=REPETITION_PENALTY,
                       streamer=streamer)
        history += [(query, streamer.response)]
    print("感谢您试用 BlueLM-7B 模型")


if __name__ == "__main__":
    main()
