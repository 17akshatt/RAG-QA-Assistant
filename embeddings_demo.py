# A hands-on demo of what an "embedding" actually is.
#
# An embedding turns a piece of text into a list of numbers (a "vector")
# that represents its MEANING. Texts with similar meaning get similar
# numbers -- even when they share no words at all.
#
# This script embeds a few sentences and prints how similar each pair is,
# so you can SEE that meaning (not word overlap) is what's being measured.

from chromadb.utils import embedding_functions

# ChromaDB ships with a built-in embedding model (all-MiniLM-L6-v2).
# It runs locally on your machine -- no API key, no internet call, no cost.
# The first time you ever use it, Chroma downloads the model (~79MB) once
# and caches it; after that it's instant.
embed = embedding_functions.DefaultEmbeddingFunction()


def cosine_similarity(a, b):
    """
    Measure how "close in meaning" two embedding vectors are.

    Cosine similarity looks at the ANGLE between two vectors, giving a
    score from about -1 to 1:
        ~1.0  = almost the same meaning
        ~0.0  = unrelated
    We compute it with plain Python so there's no magic to hide behind:
    it's the dot product of the two vectors divided by their lengths.
    """
    dot = sum(x * y for x, y in zip(a, b))
    length_a = sum(x * x for x in a) ** 0.5
    length_b = sum(y * y for y in b) ** 0.5
    return dot / (length_a * length_b)


# Four sentences. The first two mean nearly the same thing but share no
# important words. The last two are about completely different topics.
sentences = [
    "I love dogs",
    "puppies are wonderful",
    "the stock market crashed",
    "share prices fell sharply today",
]

# Turn every sentence into its embedding vector. embed(...) takes a list
# of strings and returns a list of vectors, one per sentence.
vectors = embed(sentences)

# Compare every unique pair of sentences and print their similarity score.
print("Similarity between each pair of sentences")
print("(closer to 1.0 = more similar in meaning)\n")

for i in range(len(sentences)):
    for j in range(i + 1, len(sentences)):
        score = cosine_similarity(vectors[i], vectors[j])
        print(f"  {score:5.2f}   \"{sentences[i]}\"  <->  \"{sentences[j]}\"")

print(
    "\nNotice: the two dog sentences (and the two market sentences) score"
    "\nHIGH even though they use different words, while a dog sentence vs. a"
    "\nmarket sentence scores LOW. That's meaning-based search -- the idea"
    "\nthat makes the whole RAG app work."
)
