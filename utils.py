def read_file(file_name):
    with open(f"input/{file_name}", "r", encoding="utf-8") as file:
        return file.read()
