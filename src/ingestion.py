 # PDF/text chunking & embedding
# src/ingestion.py

import os
import pickle
from argparse import ArgumentParser

from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings


def load_pdf(path: str) -> str:
    """Extract raw text from a PDF file."""
    reader = PdfReader(path)
    pages = [page.extract_text() for page in reader.pages]
    return "\n".join(pages)


def chunk_text(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> list[str]:
    """Split text into overlapping chunks for better context."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_text(text)


def embed_chunks(
    chunks: list[str],
    api_key: str | None = None
) -> list[list[float]]:
    """Generate embeddings for each text chunk via OpenAI."""
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    embedder = OpenAIEmbeddings()
    return embedder.embed_documents(chunks)


def main():
    p = ArgumentParser()
    p.add_argument("--file", required=True, help="Path to PDF (or .txt) file")
    p.add_argument(
        "--api_key",
        required=False,
        help="Your OpenAI API key (or set OPENAI_API_KEY in env)"
    )
    p.add_argument(
        "--output",
        default="data/chunks.pkl",
        help="Where to save chunks+embeddings"
    )
    args = p.parse_args()

    # 1. Load text
    text = load_pdf(args.file)
    print(f"[+] Loaded {len(text)} characters from {args.file}")

    # 2. Chunk it
    chunks = chunk_text(text)
    print(f"[+] Split into {len(chunks)} chunks (≈1000 chars each)")

    # 3. Embed
    embeddings = embed_chunks(chunks, args.api_key)
    print(f"[+] Generated embeddings for all chunks")

    # 4. Persist
    with open(args.output, "wb") as f:
        pickle.dump({"chunks": chunks, "embeddings": embeddings}, f)
    print(f"[+] Saved chunks + embeddings → {args.output}")


if __name__ == "__main__":
    main()
