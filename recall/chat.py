from modelscope import AutoModelForCausalLM, AutoTokenizer
from transformers import TextStreamer

model_name = "./Qwen2.5-7B-Instruct-with-special-tokens-memory-trained"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="cuda:5"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

prompt = "Give me a short introduction to large language model."
messages = [
    {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

# 创建流式输出器
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

print("AI回复：")
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=512,
    streamer=streamer  # 添加流式输出
)