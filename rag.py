# The core of the whole project: Retrieval-Augmented Generation (RAG).
#
# Instead of asking Gemini to answer from its own memory (which can lead
# to confidently made-up answers -- "hallucinations"), we:
#   1. RETRIEVE the most relevant chunks from our indexed documents.
#   2. GROUND the model by handing it those chunks as context.
#   3. Instruct it to answer ONLY from that context, and to say so
#      honestly when the answer isn't there.
#
# This is what "answer()" below does.

import chromadb
from chromadb.utils import embedding_functions

from llm import ask as ask_llm

CHROMA_DIR = "./chroma"
COLLECTION_NAME = "documents"

# How many chunks to retrieve per question. 4 is a good default: enough
# context to answer most questions, without overwhelming the prompt.
TOP_K = 4

SYSTEM_INSTRUCTION = (
    "You are a Q&A assistant that answers questions using ONLY the context "
    "provided in the prompt. Never use outside knowledge or things you "
    "already know.\n"
    "If the context does not contain the answer, say clearly that you "
    "don't know -- do not guess or make anything up.\n"
    "When you do answer, mention which source file(s) the information came "
    "from, using the [Source: ...] labels in the context."
)

# Connect to the same persistent collection that ingest.py built. We use
# the same embedding function so questions get embedded the same way the
# document chunks were.
_client = chromadb.PersistentClient(path=CHROMA_DIR)
_embedding_function = embedding_functions.DefaultEmbeddingFunction()

try:
    _collection = _client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=_embedding_function,
    )
except Exception:
    raise RuntimeError(
        "No document index found. Run 'python ingest.py' first to build it "
        "from the files in docs/."
    )


def answer(question):
    """
    Answer a question using RAG: retrieve relevant chunks, then ask
    Gemini to answer using only those chunks.

    Returns a tuple: (answer_text, list_of_source_filenames)
    """
    results = _collection.query(query_texts=[question], n_results=TOP_K)

    chunks = results["documents"][0]
    metadatas = results["metadatas"][0]
    sources = [metadata["source"] for metadata in metadatas]

    # Build the context block the model will answer from, labeling each
    # chunk with the file it came from so the model can cite it.
    context = "\n\n---\n\n".join(
        f"[Source: {source}]\n{chunk}" for chunk, source in zip(chunks, sources)
    )

    prompt = (
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer the question using ONLY the context above."
    )

    # Low temperature: we want a focused, consistent answer here, not a
    # creative one.
    answer_text = ask_llm(prompt, system_instruction=SYSTEM_INSTRUCTION, temperature=0.2)

    # De-duplicate sources (a question might pull multiple chunks from
    # the same file) while keeping the list readable.
    unique_sources = sorted(set(sources))

    return answer_text, unique_sources
