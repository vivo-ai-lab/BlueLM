from vllm.entrypoints.llm import LLM
from vllm.sampling_params import SamplingParams

prompts = [
    "[|Human|]:三国演义的作者是谁？[|AI|]:",
]

MODEL_PATH = "<PATH_TO_MODEL>"

sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

llm = LLM(model=MODEL_PATH, trust_remote_code=True)
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
