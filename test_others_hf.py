"""Unified inference script for SK AX-K1 and other models."""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import torch
import torch.distributed as dist
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Unified inference script")
    parser.add_argument("--model_id", type=str, required=True, help="Model ID or path")
    parser.add_argument("--prompt", type=str, required=True, help="Input prompt")
    parser.add_argument("--max_new_tokens", type=int, default=4096, help="Max new tokens")
    parser.add_argument("--output_path", type=str, default="inference_qa.json", help="Output JSON path")

    args = parser.parse_args()  # ← parse 실행
    print(f"Model: {args.model_id}")  # ← 올바른 위치로 이동
    
    return args

@torch.inference_mode()
def main():
    args = parse_args()
    is_ax = "A.X-K1" in args.model_id or "skt" in args.model_id.lower()
    is_hyperclovax = "hyperclovax" in args.model_id.lower() or "hcx" in args.model_id.lower()
    is_vaetki = "vaetki" in args.model_id.lower() or "nc" in args.model_id.lower()
    is_solar = "solar" in args.model_id.lower() or "upstage" in args.model_id.lower()
    
    # naver-HyperCLOVA X (thinking)
    if is_hyperclovax:
        tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )

        messages = [{"role": "user", "content": args.prompt}]
        
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            thinking=True,
        ).to(model.device)

        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            temperature=0.8,
            top_p=0.95,
            top_k=50,
            do_sample=True,
            # thinking_token_budget=5000,
        )

        answer_text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        print(answer_text)

        record = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "model": args.model_id,
            "question": args.prompt,
            "answer": answer_text,
            "messages": messages,
            "thinking_enabled": True,
        }
        Path(args.output_path).write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")

    # skt-AX (thinking)
    elif is_ax:
        # AX (허깅페이스 분산) - device_map auto 추가, 공식 inference와 다름.
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            # config=config,
            device_map="auto",
            dtype=torch.bfloat16,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        ).eval()
        tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)

        messages = [{"role": "user", "content": args.prompt}]
        
        inputs = tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            return_dict=True,
            add_generation_prompt=True,
            enable_thinking=True, # thinking
        ).to(model.device)

        # token_type_ids 제거
        inputs.pop("token_type_ids", None)

        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=0.6,
            top_p=0.95,
            use_cache=False,
        )

        answer_text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        print(answer_text)

        record = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "model": args.model_id,
            "question": args.prompt,
            "answer": answer_text,
            "messages": messages,
        }
        Path(args.output_path).write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")

    # upstage-Solar
    elif is_solar:
        tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )

        messages = [{"role": "user", "content": args.prompt}]
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            reasoning_effort="high",  # thinking / "low", "minimal", "high" (기본값)
        ).to(model.device)

        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            temperature=0.8,
            top_p=0.95,
            top_k=50,
            do_sample=True,
        )

        answer_text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        print(answer_text)

        record = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "model": args.model_id,
            "question": args.prompt,
            "answer": answer_text,
            "messages": messages,
        }
        Path(args.output_path).write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")

    # NC VAETKI (thinking)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
        )

        messages = [{"role": "user", "content": args.prompt}]
        input_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt", # 별도 thinking 파라미터 없는거같음
        ).to(model.device)

        outputs = model.generate(
            input_ids=input_ids,
            max_new_tokens=args.max_new_tokens,
            temperature=0.8,
            top_p=0.95,
            top_k=50,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        answer_text = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
        print(answer_text)

        record = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "model": args.model_id,
            "question": args.prompt,
            "answer": answer_text,
            "messages": messages,
        }
        Path(args.output_path).write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()