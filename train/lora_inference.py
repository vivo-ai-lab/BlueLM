from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

model_dir = ""  # <Path-To-FinetuneAdapterModel>
model = AutoPeftModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True, device_map='auto')
tokenizer_dir = model.peft_config['default'].base_model_name_or_path
tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, trust_remote_code=True, use_fast=False)
model = model.eval()
inputs = tokenizer("[|Human|]:请告诉我如何煮嫩牛肉。[|AI|]:", return_tensors="pt")
inputs = inputs.to("cuda:0")
pred = model.generate(**inputs, max_new_tokens=512, repetition_penalty=1.1)
print(tokenizer.decode(pred.cpu()[0], skip_special_tokens=True))
