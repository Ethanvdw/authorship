import os
import requests
import time
from bs4 import BeautifulSoup

def get_author_books(author_url, base_folder, author_name, max_books = 20):
    """Retrieves all the books for a given author, and downloads them"""
    try:
        response = requests.get(author_url)
        response.raise_for_status() # Raise an exception for bad status codes.
        
        soup = BeautifulSoup(response.content, 'html.parser')

        book_links = []
        for link in soup.find_all('a', class_="link"):
           book_links.append(link["href"])

        num_books = 0
        for link in book_links:
            if num_books >= max_books:
              break

            if link.startswith("/ebooks/"):
              ebook_id = link.split("/")[-1]
              download_link = f"https://www.gutenberg.org/files/{ebook_id}/{ebook_id}-0.txt"
              download_book(download_link, base_folder, author_name, ebook_id)
              num_books+=1
    except Exception as e:
        print(f"Error processing {author_url}: {e}")
        return

def download_book(download_link, base_folder, author_name, ebook_id):
    """Downloads a book from a given link to the folder for the given author"""
    author_folder = os.path.join(base_folder,author_name) # Changed: constructs the full folder name here
    os.makedirs(author_folder, exist_ok = True)

    try:
        response = requests.get(download_link)
        response.raise_for_status()
        
        file_path = os.path.join(author_folder,f"{ebook_id}.txt")
        with open(file_path,'w', encoding="utf-8") as f:
            f.write(response.text)
        print(f"Downloaded {download_link} to {file_path}")
    except Exception as e:
       print(f"Error Downloading {download_link}: {e}")


def get_gutenberg_data(base_folder, authors, max_books = 20):
    """Gathers data from Project Gutenberg for multiple authors.
    """
    for author_name, author_url in authors.items():
        author_path = os.path.join(base_folder, author_name) # Construct the path before use
        get_author_books(author_url, base_folder, author_name, max_books=max_books)
        time.sleep(1) # Add a small wait so the site doesn't get overwhelmed


# Main script
if __name__ == "__main__":
    base_folder = "texts"
    if not os.path.isdir(base_folder):
      os.mkdir(base_folder)
    authors = {
    "Jane Austen": "https://www.gutenberg.org/ebooks/author/68",
    "Charles Dickens": "https://www.gutenberg.org/ebooks/author/37",
    "Edgar Allan Poe": "https://www.gutenberg.org/ebooks/author/48" ,
    "Nathaniel Hawthorne": "https://www.gutenberg.org/ebooks/author/28" ,
    "H.G. Wells": "https://www.gutenberg.org/ebooks/author/30" ,
    "Jules Verne": "https://www.gutenberg.org/ebooks/author/60" 
   }

    max_books_per_author = 20

    get_gutenberg_data(base_folder, authors, max_books = max_books_per_author)
    print("Download complete!")
