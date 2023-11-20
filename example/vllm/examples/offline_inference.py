from vllm import LLM, SamplingParams

# Sample prompts.
# "[|Human|]:Hello, what's your name[|AI|]:",
prompts = [
    "[|Human|]:Hello[|AI|]:",
]
# Create a sampling params object.
# sampling_params = SamplingParams(temperature=0,max_tokens=1024) # for greedy_sampling
sampling_params = SamplingParams(temperature=0.8,top_k= 1, top_p=0.01,max_tokens=128)

# Create an LLM.
# llm = LLM(model="facebook/opt-125m")
llm = LLM(model="/data/vjuicefs_ai_gpt/public_data/SFT/model/open_model/BlueLM-7B-Chat-32K",trust_remote_code=True)
#llm = LLM(model="/data/vjuicefs_ai_gpt/public_data/11119859/checkpoint/v0728_vivo_qjj/HF3000")
# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
outputs = llm.generate(prompts, sampling_params)
# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
