
# Ingestion: read every file in docs/, cut it into small overlapping
# chunks, and store those chunks in a persistent ChromaDB collection.
#
# Why chunk? Documents are too big to hand the model whole, and search
# works better over small, focused pieces. Why overlap? So a sentence
# that falls right on a chunk boundary doesn't get awkwardly cut in half
# with no surrounding context.
#
# Why store the source filename with each chunk? So that later, when we
# answer a question, we can tell the user exactly which document(s) the
# answer came from (a citation).

import os

import chromadb
from chromadb.utils import embedding_functions
from pypdf import PdfReader

DOCS_DIR = "docs"
CHROMA_DIR = "./chroma"
COLLECTION_NAME = "documents"

# ~800 characters per chunk, with a 100-character overlap between
# consecutive chunks. These numbers are a reasonable starting point,
# not a hard rule -- feel free to tune them later.
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100


def read_text_file(path):
    """Read a .txt or .md file as plain text."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def read_pdf_file(path):
    """Read a .pdf file by extracting the text of every page."""
    reader = PdfReader(path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """
    Split one long string into a list of overlapping chunks.

    Example with chunk_size=10, overlap=3 on "ABCDEFGHIJKLMNOP":
        "ABCDEFGHIJ", "HIJKLMNOP", ...
    Each chunk starts (chunk_size - overlap) characters after the
    previous one, so the last few characters of one chunk reappear at
    the start of the next.
    """
    chunks = []
    start = 0
    step = chunk_size - overlap
    while start < len(text):
        chunk = text[start : start + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
        start += step
    return chunks


def load_and_chunk_documents():
    """
    Read every supported file in docs/ and return a list of
    (filename, chunk_text) pairs, one entry per chunk.
    """
    results = []
    for filename in sorted(os.listdir(DOCS_DIR)):
        path = os.path.join(DOCS_DIR, filename)
        if not os.path.isfile(path):
            continue

        extension = filename.lower().rsplit(".", 1)[-1]
        if extension in ("txt", "md"):
            text = read_text_file(path)
        elif extension == "pdf":
            text = read_pdf_file(path)
        else:
            continue  # skip anything that isn't a doc we know how to read

        for chunk in chunk_text(text):
            results.append((filename, chunk))
    return results


def build_index():
    """
    Read docs/, chunk it, and (re)build the persistent Chroma collection
    from scratch. Returns (chunk_count, file_count).

    This is the reusable core of ingestion -- both the command-line
    `python ingest.py` and app.py (which needs to build the index itself
    the first time it runs on a fresh deployment) call this.
    """
    chunks = load_and_chunk_documents()
    if not chunks:
        return 0, 0

    # Store the index on disk at ./chroma so it survives between runs.
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    # Rebuild the collection from scratch each time, so the index always
    # reflects exactly what's currently in docs/ -- no stale or duplicate
    # chunks left over from a previous run.
    if COLLECTION_NAME in [c.name for c in client.list_collections()]:
        client.delete_collection(COLLECTION_NAME)

    # Chroma's default embedding function runs locally on your machine --
    # no API key needed. It turns each chunk's text into a vector so we
    # can later search by meaning.
    embedding_function = embedding_functions.DefaultEmbeddingFunction()
    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_function,
    )

    ids = [f"{filename}-{i}" for i, (filename, _) in enumerate(chunks)]
    documents = [chunk for _, chunk in chunks]
    metadatas = [{"source": filename} for filename, _ in chunks]

    collection.add(ids=ids, documents=documents, metadatas=metadatas)

    file_count = len({filename for filename, _ in chunks})
    return len(chunks), file_count


def main():
    chunk_count, file_count = build_index()
    if chunk_count == 0:
        print(f"No .txt, .md, or .pdf files found in {DOCS_DIR}/. Add some and try again.")
        return
    print(f"Indexed {chunk_count} chunks from {file_count} files")


if __name__ == "__main__":
    main()
