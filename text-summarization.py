from transformers import pipeline


def read_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()


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
    file_path = "input.txt"  # Path to your input text file
    text = read_file(file_path)

    summary = summarize_text(text)

    print(f"Summary:\n {summary}")


if __name__ == "__main__":
    main()
