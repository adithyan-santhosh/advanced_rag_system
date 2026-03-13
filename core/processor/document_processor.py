from pypdf import PdfReader
import os


class DocumentProcessor:

    def extract_text(self, file_path):

        if file_path.endswith(".pdf"):
            return self.extract_pdf(file_path)

        elif file_path.endswith(".txt"):
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()

        else:
            raise Exception("Unsupported file format")


    def extract_pdf(self, file_path):

        reader = PdfReader(file_path)
        text = ""

        for page in reader.pages:
            extracted = page.extract_text()

            if extracted:
                text += extracted + "\n"

        return text