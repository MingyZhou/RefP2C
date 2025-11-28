import logging
import os


class PaperLoader:
    def __init__(self, paper_file_path):
        self.paper_file_path = paper_file_path

    def load(self):
        """
        Load Markdown Paper.
        """
        if not os.path.isfile(self.paper_file_path):
            logging.error(
                f"File not found at the specified path: {self.paper_file_path}"
            )
            return ""
        try:
            with open(self.paper_file_path, "r", encoding="utf-8") as f:
                content = f.read()
            logging.info("Paper loaded successfully.")
            return content
        except Exception as e:
            logging.error(f"An error occurred while reading the file: {e}")
            return ""
