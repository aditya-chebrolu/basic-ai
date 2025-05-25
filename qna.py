from transformers import pipeline

from utils import read_file


def answer_question(context, question):
    nlp = pipeline("question-answering", model="deepset/roberta-base-squad2")
    result = nlp(question=question, context=context)
    return result["answer"]


def main():
    context = read_file("context.txt")
    # input question
    question = input("Enter your question: ")
    answer = answer_question(context, question)
    print(f"Answer: {answer}")


if __name__ == "__main__":
    main()
