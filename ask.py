# A tiny command-line tool for testing the RAG pipeline.
#
# Usage:
#   python ask.py "your question here"

import sys

from rag import answer


def main():
    if len(sys.argv) < 2:
        print('Usage: python ask.py "your question here"')
        return

    question = " ".join(sys.argv[1:])
    answer_text, sources = answer(question)

    print(answer_text)
    print()
    if sources:
        print("Sources:", ", ".join(sources))


if __name__ == "__main__":
    main()
