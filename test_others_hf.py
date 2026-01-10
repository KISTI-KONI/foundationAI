import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datetime import datetime
import json
from pathlib import Path

MODEL_ID= "skt/A.X-K1"
MODEL_ID= "naver-hyperclovax/HyperCLOVAX-SEED-Think-32B" 
MODEL_ID= "NC-AI-consortium-VAETKI/VAETKI"
MODEL_ID= "upstage/Solar-Open-100B"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)

messages = [{"role": "user", "content": "Which one is bigger, 3.9 vs 3.12?"}]
inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt",
)
inputs = inputs.to(model.device)

generated_ids = model.generate(
    **inputs,
    max_new_tokens=4096,
    temperature=0.8,
    top_p=0.95,
    top_k=50,
    do_sample=True,
)
answer_text = tokenizer.decode(generated_ids[0][inputs.input_ids.shape[1]:])
print(answer_text)

question_text = next(m["content"] for m in reversed(messages) if m["role"] == "user")

record = {
    "created_at": datetime.now().isoformat(timespec="seconds"),
    "model": MODEL_ID,
    "question": question_text,
    "answer": answer_text,
    "messages": messages,
}

out_path = Path("qa.json")
out_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")