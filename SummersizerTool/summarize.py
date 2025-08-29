import argparse
import sys
from typing import Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def build_summarizer(model_name: str) -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
    """
    Load tokenizer and model for the given Hugging Face LLaMA chat model.
    """
    # Prefer bfloat16 if available (common on newer GPUs/CPUs), else float16
    if torch.cuda.is_available():
        torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        # On CPU, bf16 may be supported by some CPUs under accelerate; default to float32 for safety
        torch_dtype = torch.float32

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

    # Some LLaMA models need a pad token; fallback to eos token if missing
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map="auto",
    )

    return tokenizer, model


def summarize_text(
    summarizer: Tuple[AutoTokenizer, AutoModelForCausalLM],
    text: str,
    max_tokens: int = 200,
    temperature: float = 0.7,
) -> str:
    """
    Generate a concise summary for the given text using the chat template.
    """
    tokenizer, model = summarizer

    system_prompt = "You are a helpful assistant that writes extremely concise, faithful summaries."
    user_prompt = f"Summarize the following text concisely in 2-4 sentences.\n\n{text}"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    # Build chat-formatted prompt string
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = tokenizer(prompt_text, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    do_sample = temperature is not None and temperature > 0.0

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        do_sample=do_sample,
        temperature=temperature if do_sample else None,
        top_p=0.95 if do_sample else None,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Heuristic: the generated string contains the prompt followed by the assistant reply
    if decoded.startswith(prompt_text):
        summary = decoded[len(prompt_text) :].strip()
    else:
        # Fallback: try to split on assistant tag if present; else use the tail
        summary = decoded.split("</s>")[-1].strip()

    return summary


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LLaMA Summarizer CLI: generate a concise summary for input text",
    )
    parser.add_argument(
        "text",
        type=str,
        help="Input text to summarize (quote the string)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-2-7b-chat-hf",
        help="Hugging Face model name (default: meta-llama/Llama-2-7b-chat-hf)",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=200,
        help="Maximum new tokens in the summary (default: 200)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature for creativity (default: 0.7)",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = _parse_args(argv)

    tokenizer, model = build_summarizer(args.model)

    summary = summarize_text(
        (tokenizer, model), args.text, max_tokens=args.max_tokens, temperature=args.temperature
    )

    print("\n=== Summary ===\n")
    print(summary)
    print("\n===============\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))


