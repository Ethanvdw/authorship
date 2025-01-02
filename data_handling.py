import os

def read_file_content(file_path):
    """Reads the content of a text file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return None

def gather_text_data(base_folder):
    """Gathers text data from specified folders for each author."""
    texts = []
    authors = []

    for author_folder in os.listdir(base_folder):
         author_path = os.path.join(base_folder, author_folder)
         if os.path.isdir(author_path):
            for filename in os.listdir(author_path):
                if filename.endswith(".txt"):
                     file_path = os.path.join(author_path,filename)
                     text = read_file_content(file_path)
                     if text:
                        texts.append(text)
                        authors.append(author_folder)
    return texts, authors
