from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers
from datetime import datetime
import json
from pathlib import Path

print(f"Transformers version: {transformers.__version__}")

model_name = "LGAI-EXAONE/K-EXAONE-236B-A23B"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype="bfloat16",
    device_map="auto",
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

messages = [
    {"role": "system", "content": "You are K-EXAONE, a large language model developed by LG AI Research in South Korea, built to serve as a helpful and reliable assistant."},
    {"role": "user", "content": "Which one is bigger, 3.9 vs 3.12?"}
]
input_ids = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
    enable_thinking=True,
)

generated_ids = model.generate(
    **input_ids.to(model.device),
    max_new_tokens=16384,
    temperature=1.0,
    top_p=0.95,
)
output_ids = generated_ids[0][input_ids['input_ids'].shape[-1]:]
answer_text = tokenizer.decode(output_ids, skip_special_tokens=True)
print(answer_text)

question_text = next(m["content"] for m in reversed(messages) if m["role"] == "user")

record = {
    "created_at": datetime.now().isoformat(timespec="seconds"),
    "model": model_name,
    "transformers_version": transformers.__version__,
    "question": question_text,
    "answer": answer_text,
    "messages": messages,
}

out_path = Path("results/exaone_qa.json")
out_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
