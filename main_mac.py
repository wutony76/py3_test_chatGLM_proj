from __future__ import print_function
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
import threading
from pathlib import Path

MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # smaller & faster for quick replies
LOCAL_MODEL_DIR = Path(__file__).resolve().parent / "models" / "TinyLlama-1.1B-Chat-v1.0"


def main():
  print("start. main")

  if torch.backends.mps.is_available():
    device = torch.device("mps")
  elif torch.cuda.is_available():
    device = torch.device("cuda")
  else:
    device = torch.device("cpu")
  print(f"device: {device}")

  model_source = str(LOCAL_MODEL_DIR) if LOCAL_MODEL_DIR.exists() else MODEL_ID
  print(f"model source: {model_source}")

  tokenizer = AutoTokenizer.from_pretrained(model_source, use_fast=True)
  dtype = torch.float16 if device.type in ["mps", "cuda"] else torch.float32
  model = AutoModelForCausalLM.from_pretrained(
    model_source,
    torch_dtype=dtype,
    ignore_mismatched_sizes=True
  ).to(device)
  model.eval()

  if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

  history = []  # list of {"role": "user"|"assistant", "content": str}
  while True:
      query = input("你: ")
      if query.lower() in ["exit", "quit"]:
          break

      messages = history + [{"role": "user", "content": query}]
      if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
      else:
        # Fallback prompt format
        parts = []
        for m in messages[:-1]:
          if m["role"] == "user":
            parts.append(f"User: {m['content']}\nAssistant:")
          else:
            parts.append(m["content"])
        parts.append(f"User: {query}\nAssistant:")
        prompt_text = "\n".join(parts)

      inputs = tokenizer(prompt_text, return_tensors="pt").to(device)

      streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
      gen_kwargs = dict(
        input_ids=inputs["input_ids"],
        max_new_tokens=256,
        do_sample=False,
        top_p=0.7,
        temperature=0.7,
        streamer=streamer,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
      )

      print("ChatGLM: ", end="", flush=True)
      thread = threading.Thread(target=model.generate, kwargs=gen_kwargs)
      thread.start()

      assistant_text = ""
      for piece in streamer:
        print(piece, end="", flush=True)
        assistant_text += piece
      print()

      history.append({"role": "user", "content": query})
      history.append({"role": "assistant", "content": assistant_text})

if __name__ == "__main__":
    main()