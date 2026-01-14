# 노드 이름 (인자로 받거나 기본값 gpu47)
GPU=${1:-gpu47}

curl -X POST http://${GPU}:8000/v1/chat/completions \
-H "Content-Type: application/json" \
-d '{
  "model": "LGAI-EXAONE/K-EXAONE-236B-A23B",
  "messages": [
    {"role": "user", "content": "How many r'\''s in \"strawberry\"?"}
  ],
  "max_tokens": 16384,
  "temperature": 1.0,
  "top_p": 0.95,
  "chat_template_kwargs": {"enable_thinking": true}
}'