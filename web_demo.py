# streamlit run web_demo.py --server.port 8080
import torch
import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

DEVICE = "cuda:0"
MODEL_ID = "vivo-ai/BlueLM-7B-Chat"

st.set_page_config(
    page_title="BlueLM-7B Demo",
    page_icon=":robot:",
    layout="wide"
)

st.title("BlueLM-7B")


@st.cache_resource
def get_model():
    print("Begin to Load BlueLM Model")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map=DEVICE, torch_dtype=torch.bfloat16,
                                                 trust_remote_code=True)
    model = model.eval()
    print("BlueLM Model is Ready!")
    return tokenizer, model


tokenizer, model = get_model()


def build_prompt(history, prompt):
    res = ""
    for query, response in history:
        res += f"[|Human|]:{query}[|AI|]:{response}</s>"
    res += f"[|Human|]:{prompt}[|AI|]:"
    return res


class BlueLMStreamer(TextStreamer):
    def __init__(self, tokenizer: "AutoTokenizer", message_placeholder):
        self.tokenizer = tokenizer
        self.tokenIds = []
        self.prompt = ""
        self.response = ""
        self.first = True
        self.message_placeholder = message_placeholder

    def put(self, value):
        if self.first:
            self.first = False
            return
        self.tokenIds.append(value.item())
        text = tokenizer.decode(self.tokenIds, skip_special_tokens=True)
        if text and text[-1] != "�":
            self.message_placeholder.markdown(text)

    def end(self):
        self.first = True
        text = tokenizer.decode(self.tokenIds, skip_special_tokens=True)
        self.response = text
        self.message_placeholder.markdown(text)
        self.tokenIds = []


max_new_tokens = st.sidebar.slider("max_new_tokens", 0, 2048, 512, step=1)
top_p = st.sidebar.slider("top_p", 0.0, 1.0, 0.8, step=0.01)
top_k = st.sidebar.slider("top_k", 0, 100, 50, step=1)
temperature = st.sidebar.slider("temperature", 0.0, 2.0, 1.0, step=0.01)
do_sample = st.sidebar.checkbox("do_sample", value=True)

if "history" not in st.session_state:
    st.session_state.history = []

for i, (query, response) in enumerate(st.session_state.history):
    with st.chat_message(name="user", avatar="user"):
        st.markdown(query)
    with st.chat_message(name="assistant", avatar="assistant"):
        st.markdown(response)
with st.chat_message(name="user", avatar="user"):
    input_placeholder = st.empty()
with st.chat_message(name="assistant", avatar="assistant"):
    message_placeholder = st.empty()

prompt_text = st.text_area(label="用户命令输入",
                           height=100,
                           placeholder="请在这儿输入您的命令",
                           key="input_text_area")

button = st.button("发送", key="predict")

if button:
    prompt_text = prompt_text.strip()
    input_placeholder.markdown(prompt_text)
    history = st.session_state.history
    streamer = BlueLMStreamer(tokenizer=tokenizer, message_placeholder=message_placeholder)
    prompt = build_prompt(history=history, prompt=prompt_text)
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = inputs.to(DEVICE)
    input_ids = inputs["input_ids"]
    model.generate(input_ids=input_ids, max_new_tokens=max_new_tokens, do_sample=do_sample, top_p=top_p, top_k=top_k,
                   temperature=temperature, streamer=streamer)
    history += [(prompt_text, streamer.response)]
    st.session_state.history = history
