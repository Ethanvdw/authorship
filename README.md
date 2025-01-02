# Authorship Attribution

This project is designed to attribute authorship to texts using machine learning models.

## Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Ethanvdw/authorship.git
   cd authorship
   ```

2. **Install dependencies:**
   Make sure you have UV installed. Then, install the required Python packages using pip:
   ```bash
   uv pip install -r requirements.txt
   ```

## Running the Program

To run the program, use the following command:
```bash
uv run main.py <string> <string>
```
example:
```bash
uv run main.py "It is a truth universally acknowledged, that a single man in possession of a good fortune, must b
e in want of a wife." "War is peace. Freedom is slavery. Ignorance is strength."
```

## Training a New Model

To train a new model, you'll need to place text files in a `texts` folder. You can use the `downloader.py` script to download files from Project Gutenberg.

1. **Place text files:**
   Ensure that your text files are placed in the `texts` folder. The structure should be:
   ```
   texts/
   ├── Author1/
   │   ├── file1.txt
   │   └── file2.txt
   ├── Author2/
   │   ├── file1.txt
   │   └── file2.txt
   ```

2. **Download files using the downloader script:**
   You can use the `downloader.py` script to download files from Project Gutenberg:
   ```bash
   uv python downloader.py
   ```

3. **Train the model:**
  To train a new model, delete `authorship_model.joblib` and then run the program again.

## Notes

- Ensure that your text files are properly formatted and cleaned before training the model.
- The `requirements.txt` file contains all the necessary Python packages needed to run the program.

For any issues or contributions, please open an issue or submit a pull request on the GitHub repository.
