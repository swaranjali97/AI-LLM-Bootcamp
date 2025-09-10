import argparse
import sys
from typing import List, Tuple

import numpy as np
from huggingface_hub import login
import os

# Read token from environment variable
token = os.getenv("HUGGINGFACE_TOKEN")
if token:
    print("✅ Hugging Face token found, logging in...")
    login(token)
else:
    raise RuntimeError("❌ Hugging Face token not found. Please set HUGGINGFACE_TOKEN")

try:
    import faiss  # type: ignore
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "faiss-cpu is required. Install with: pip install faiss-cpu"
    ) from exc

import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer


# Small, hardcoded corpus of documents for the demo
DOCS: List[str] = [
    "Retrieval-Augmented Generation (RAG) combines information retrieval with text generation.",
    "FAISS is a library for efficient similarity search and clustering of dense vectors.",
    "Sentence-Transformers provides easy-to-use models to compute sentence embeddings.",
    "DistilGPT2 is a smaller, faster variant of GPT-2 for text generation tasks.",
    "Cosine similarity is commonly used to compare embedding vectors for semantic similarity.",
    "MiniLM models offer a strong balance of speed and accuracy for embeddings.",
]


def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():  # Apple Silicon
        return torch.device("mps")
    return torch.device("cpu")


def embed_docs(texts: List[str], model: SentenceTransformer) -> np.ndarray:
    """Compute L2-normalized embeddings for a list of documents.

    Returns a 2D numpy array of shape (num_docs, dim).
    """
    # SentenceTransformer handles batching internally
    embeddings = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
    return embeddings.astype("float32", copy=False)


def build_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """Build a FAISS index for inner-product (cosine if vectors are normalized)."""
    if embeddings.ndim != 2:
        raise ValueError("embeddings must be a 2D array [num_docs, dim]")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index


def retrieve(
    query: str,
    model: SentenceTransformer,
    index: faiss.IndexFlatIP,
    texts: List[str],
    top_k: int = 3,
) -> List[Tuple[str, float]]:
    """Embed the query, search FAISS, and return top_k (doc, score) pairs."""
    if top_k < 1:
        raise ValueError("top_k must be >= 1")
    # Normalize to use cosine similarity via inner product
    q_emb = model.encode([query], normalize_embeddings=True, convert_to_numpy=True).astype("float32")
    scores, indices = index.search(q_emb, min(top_k, len(texts)))
    top_docs: List[Tuple[str, float]] = []
    for idx, score in zip(indices[0], scores[0]):
        if idx == -1:
            continue
        top_docs.append((texts[idx], float(score)))
    return top_docs


def generate_answer(
    context: str,
    question: str,
    model_name: str = "distilgpt2",
    max_new_tokens: int = 128,
    temperature: float = 0.7,
    top_p: float = 0.95,
    top_k: int = 50,
) -> str:
    """Generate an answer conditioned on the provided context and question."""
    device = _get_device()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        # GPT2-like models often have no pad token; set it to eos
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(device)
    model.eval()

    prompt = (
        "Context:\n" + context.strip() + "\n\n" +
        "Question: " + question.strip() + "\n" +
        "Answer:"
    )

    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            do_sample=True,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract everything after the last occurrence of "Answer:" to be safe
    answer = full_text.split("Answer:")[-1].strip()
    return answer


def main() -> None:
    parser = argparse.ArgumentParser(description="Mini RAG Q&A Bot (single-file demo)")
    parser.add_argument("--top_k", type=int, default=3, help="Number of documents to retrieve")
    parser.add_argument(
        "--embed_model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Sentence-Transformers model to use for embeddings",
    )
    parser.add_argument(
        "--gen_model",
        type=str,
        default="distilgpt2",
        help="HF Transformers causal LM to use for generation",
    )
    args = parser.parse_args()

    device = _get_device()
    print(f"Using device: {device}")

    # Load embedding model
    print("Loading embedding model...", flush=True)
    emb_model = SentenceTransformer(args.embed_model, device=str(device))

    # Embed documents and build FAISS index
    print("Embedding documents...", flush=True)
    doc_embeddings = embed_docs(DOCS, emb_model)
    print(f"Built {doc_embeddings.shape[0]} embeddings of dimension {doc_embeddings.shape[1]}")

    print("Building FAISS index...", flush=True)
    index = build_index(doc_embeddings)

    # Read a single user query from stdin
    try:
        question = input("Enter your question: ").strip()
    except EOFError:
        print("No input received. Exiting.")
        sys.exit(0)

    if not question:
        print("Empty question. Exiting.")
        sys.exit(0)

    # Retrieve top documents
    print("Retrieving relevant documents...", flush=True)
    top_docs = retrieve(question, emb_model, index, DOCS, top_k=args.top_k)

    # Build context string
    context_lines = []
    for i, (doc, score) in enumerate(top_docs, start=1):
        context_lines.append(f"[{i}] (score={score:.3f}) {doc}")
    context_str = "\n".join(context_lines)

    print("\nRetrieved context:")
    print(context_str)

    # Generate answer
    print("\nGenerating answer...", flush=True)
    answer = generate_answer(context=context_str, question=question, model_name=args.gen_model)

    print("\nAnswer:")
    print(answer)


if __name__ == "__main__":
    main()


