import json
import time
import asyncio
from queue import Queue
from threading import Thread
from contextlib import asynccontextmanager
from typing import List, Literal, Optional, Union

import torch
import uvicorn
from loguru import logger
from pydantic import BaseModel, Field

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
from transformers.generation.logits_process import LogitsProcessor

from sse_starlette.sse import EventSourceResponse


EventSourceResponse.DEFAULT_PING_INTERVAL = 1000


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

streamer_queue = Queue()


class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "owner"
    root: Optional[str] = None
    parent: Optional[str] = None
    permission: Optional[list] = None


class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelCard] = []


class FunctionCallResponse(BaseModel):
    name: Optional[str] = None
    arguments: Optional[str] = None


class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system", "function"]
    content: str = None
    tool_calls: Optional[list] = None


class DeltaMessage(BaseModel):
    role: Optional[Literal["user", "assistant", "system"]] = None
    content: Optional[str] = None
    tool_calls: Optional[list] = None


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    total_tokens: int = 0
    completion_tokens: Optional[int] = 0


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.8
    top_p: Optional[float] = 0.8
    max_tokens: Optional[int] = None
    tools: Optional[Union[dict, List[dict]]] = None
    repetition_penalty: Optional[float] = 1.1
    stream: Optional[bool] = False


class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Literal["stop", "length", "tool_calls"]


class ChatCompletionResponseStreamChoice(BaseModel):
    index: int
    delta: DeltaMessage
    finish_reason: Optional[Literal["stop", "length", "tool_calls"]]


class ChatCompletionResponse(BaseModel):
    id: str
    model: str
    object: Literal["chat.completion", "chat.completion.chunk"]
    choices: List[Union[ChatCompletionResponseChoice, ChatCompletionResponseStreamChoice]]
    created: Optional[int] = Field(default_factory=lambda: int(time.time()))
    usage: Optional[UsageInfo] = None


class InvalidScoreLogitsProcessor(LogitsProcessor):
    def __call__(
            self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        if torch.isnan(scores).any() or torch.isinf(scores).any():
            scores.zero_()
            scores[..., 5] = 5e4
        return scores


class CustomStreamer(TextStreamer):
    def __init__(self, queue, tokenizer, skip_prompt, **decode_kwargs) -> None:
        super().__init__(tokenizer, skip_prompt, **decode_kwargs)
        self._queue = queue
        self.stop_signal = None
        self.timeout = 1

    def on_finalized_text(self, text: str, stream_end: bool = False):
        if text != "":
            self._queue.put(text)
        if stream_end:
            self._queue.put(self.stop_signal)


def process_bluelm_messages(messages, tools=None):
    _messages = messages
    messages = []
    if tools:
        tool_system_content = "你是一个AI助手，尽你所能回答用户的问题，你可以使用的工具如下:\n<APIs>\n- "
        tool_system_content += str("\n- ".join([str(i) for i in tools]))
        tool_system_content += '\n</APIs>\n你需要根据用户问题，选择合适的工具，输出的格式为：\n{"answer":"给用户的回复","function_calls":[{"name":"函数名","parameters":{"参数名":"参数"}}]}\n如果不需要额外回复或者没有合适工具，则对应字段输出空。\n'

        messages.append(
            {
                "role": "system",
                "content": tool_system_content,
                "tools": tools
            }
        )
    for m in _messages:
        role, content, tool_calls = m.role, m.content, m.tool_calls
        if role == "function":
            messages.append(
                {
                    "role": "observation",
                    "content": content
                }
            )
        elif role == "assistant" and tool_calls is not None:
            rewrite_function_calls = []
            for tool_call in tool_calls:
                rewrite_function_calls.append({
                    "name": tool_call["name"],
                    "parameters": tool_call["arguments"]
                })
            content = json.dumps(
                {
                    "answer": None,
                    "function_calls": rewrite_function_calls
                },
                ensure_ascii=False
            )
            messages.append({"role": role, "content": content})
        else:
            messages.append({"role": role, "content": content})
    return messages


def build_chat_input(query, history, role):
    if history is None:
        history = []
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

    if role == "observation":
        prompt += "[|Function|]:"
    else:
        prompt += "[|Human|]:"
    prompt += query
    prompt += "[|AI|]:"
    return prompt


@torch.inference_mode()
def start_generation(model, tokenizer, params):
    messages = params["messages"]
    temperature = float(params.get("temperature", 1.0))
    top_p = float(params.get("top_p", 1.0))
    max_new_tokens = int(params.get("max_tokens", 256))
    echo = params.get("echo", True)
    repetition_penalty = float(params.get("repetition_penalty", 1.0))
    tools = params["tools"]
    messages = process_bluelm_messages(messages, tools=tools)

    query, role = messages[-1]["content"], messages[-1]["role"]
    prompt = build_chat_input(query=query, history=messages[:-1], role=role)
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = inputs.to(model.device)

    eos_token_id = [
        tokenizer.eos_token_id,
        tokenizer._convert_token_to_id("[|Human|]:"),
    ]

    streamer = CustomStreamer(streamer_queue, tokenizer, True, skip_special_tokens=True)

    gen_kwargs = dict(
        inputs,
        streamer=streamer,
        max_new_tokens=max_new_tokens,
        do_sample=True if temperature > 1e-5 else False,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        logits_processor=[InvalidScoreLogitsProcessor()],
        eos_token_id=eos_token_id,
        temperature=temperature,
    )

    thread = Thread(target=model.generate, kwargs=gen_kwargs)
    thread.start()


async def stream_generator(model, tokenizer, params):
    start_generation(model, tokenizer, params)

    model_id = params.get("model_id", "bluelm-7b")

    while True:
        value = streamer_queue.get()
        if value is None:
            choice_data = ChatCompletionResponseStreamChoice(
                index=0,
                delta=DeltaMessage(role="assistant"),
                finish_reason="stop"
            )
            chunk = ChatCompletionResponse(
                model=model_id,
                id="",
                choices=[choice_data],
                created=int(time.time()),
                object="chat.completion.chunk"
            )
            yield "{}".format(chunk.json(exclude_unset=True))
            yield "[DONE]"
            break

        message = DeltaMessage(
            content=value,
            role="assistant",
            tool_calls=None,
        )
        choice_data = ChatCompletionResponseStreamChoice(
            index=0,
            delta=message,
            finish_reason=None
        )
        chunk = ChatCompletionResponse(
            model=model_id,
            id="",
            choices=[choice_data],
            created=int(time.time()),
            object="chat.completion.chunk"
        )
        yield "{}".format(chunk.json(exclude_unset=True))

        streamer_queue.task_done()
        await asyncio.sleep(0.1)


def generator(model, tokenizer, params):
    messages = params["messages"]
    temperature = float(params.get("temperature", 1.0))
    top_p = float(params.get("top_p", 1.0))
    max_new_tokens = int(params.get("max_tokens", 256))
    echo = params.get("echo", True)
    repetition_penalty = float(params.get("repetition_penalty", 1.0))
    tools = params["tools"]
    messages = process_bluelm_messages(messages, tools=tools)

    query, role = messages[-1]["content"], messages[-1]["role"]
    prompt = build_chat_input(query=query, history=messages[:-1], role=role)
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = inputs.to(model.device)
    input_echo_len = len(inputs["input_ids"][0])

    eos_token_id = [
        tokenizer.eos_token_id,
        tokenizer._convert_token_to_id("[|Human|]:"),
    ]

    gen_kwargs = dict(
        inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True if temperature > 1e-5 else False,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        logits_processor=[InvalidScoreLogitsProcessor()],
        eos_token_id=eos_token_id,
        temperature=temperature,
    )

    output = model.generate(**gen_kwargs).cpu()
    total_len = len(output[0])
    if echo:
        text = tokenizer.decode(output[0], skip_special_tokens=True)
    else:
        text = tokenizer.decode(output[0][input_echo_len:], skip_special_tokens=True)

    return {
        "text": text,
        "usage": {
            "prompt_tokens": input_echo_len,
            "completion_tokens": total_len - input_echo_len,
            "total_tokens": total_len
        }
    }


def parse_response(response):
    try:
        text_json = json.loads(response["text"])
        if "function_calls" in text_json and len(text_json["function_calls"]) > 0:
            text = text_json["answer"]
            function_call_lst = text_json["function_calls"]
            tool_calls = []
            for item in function_call_lst:
                tool_calls.append(
                    FunctionCallResponse(
                        name=item["name"],
                        arguments=item["parameters"]
                    )
                )

            return text, tool_calls
    except:
        return response["text"], None


@app.get("/v1/models", response_model=ModelList)
async def list_models():
    model_card = ModelCard(id="bluelm-7b")
    return ModelList(data=[model_card])


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(request: ChatCompletionRequest):
    global model, tokenizer

    if len(request.messages) < 1 or request.messages[-1].role == "assistant":
        raise HTTPException(status_code=400, detail="Invalid request")

    gen_params = dict(
        messages=request.messages,
        temperature=request.temperature,
        top_p=request.top_p,
        max_tokens=request.max_tokens or 1024,
        echo=False,
        stream=request.stream,
        repetition_penalty=request.repetition_penalty,
        tools=request.tools,
        model_id=request.model,
    )
    logger.debug(f"==== request ====\n{gen_params}")

    if request.stream:
        if request.tools:
            raise HTTPException(
                status_code=400,
                detail=
                "Invalid request: Function calling is not yet implemented for stream mode.",
            )
        return EventSourceResponse(stream_generator(model, tokenizer, gen_params), media_type="text/event-stream")

    else:
        gen_result = generator(model, tokenizer, gen_params)
        tool_calls, finish_reason = None, "stop"
        text = gen_result["text"]
        if request.tools:
            text, tool_calls = parse_response(gen_result)
        if tool_calls is not None and len(tool_calls) > 0:
            finish_reason = "tool_calls"

        message = ChatMessage(
            role="assistant",
            content=text,
            tool_calls=tool_calls
        )

        choice_data = ChatCompletionResponseChoice(
            index=0,
            message=message,
            finish_reason=finish_reason,
        )

        usage = UsageInfo(**gen_result["usage"])

        return ChatCompletionResponse(
            model=request.model,
            id="",
            choices=[choice_data],
            object="chat.completion",
            usage=usage
        )


if __name__ == "__main__":
    MODEL_ID = "vivo-ai/BlueLM-7B-Chat-32K"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map="cuda:0", torch_dtype=torch.bfloat16, trust_remote_code=True)
    model = model.eval()
    uvicorn.run(app, host="0.0.0.0", port=7776, workers=1)
