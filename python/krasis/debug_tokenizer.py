"""Debug script: check what apply_chat_template returns on this system."""
import sys
import transformers
print("transformers:", transformers.__version__)
print("python:", sys.version)

from transformers import AutoTokenizer

# Use first arg as model path, or default
model_path = sys.argv[1] if len(sys.argv) > 1 else "~/.krasis/models/Qwen3-Coder-Next"
print("model_path:", model_path)

tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
print("tokenizer class:", type(tok).__name__)
print("has chat_template:", hasattr(tok, "chat_template") and tok.chat_template is not None)

result = tok.apply_chat_template(
    [{"role": "user", "content": "hello"}],
    add_generation_prompt=True,
    tokenize=True,
)
print("result type:", type(result))
print("result repr:", repr(result)[:200])
if isinstance(result, list) and len(result) > 0:
    print("element type:", type(result[0]))
    print("first 5:", result[:5])
