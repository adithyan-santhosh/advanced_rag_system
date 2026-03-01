import os


class DocumentLoader:

    def __init__(self, data_folder="data"):
        self.data_folder = data_folder


    def load_documents(self):
        documents = []

        for file in os.listdir(self.data_folder):
            path = os.path.join(self.data_folder, file)
            print(path)
            if file.endswith(".txt"):
                with open(path, "r", encoding="utf-8") as f:
                    text = f.read()
                    documents.append({
                        "text": text,
                        "source": file
                    })

        return documents