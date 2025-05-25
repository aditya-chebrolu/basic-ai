from transformers import pipeline

from utils import read_file


def summarize_text(text, max_length=130, min_length=30):
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    summary = summarizer(
        text,
        max_length=max_length,
        min_length=min_length,
        do_sample=False,
    )
    return summary[0]["summary_text"]


def main():
    text = read_file("para.txt")
    summary = summarize_text(text)
    print(f"Summary:\n {summary}")


if __name__ == "__main__":
    main()
