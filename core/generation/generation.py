import requests


class OllamaLLM:
    def __init__(self, model_name="mistral"):
        self.model_name = model_name
        self.url = "http://localhost:11434/api/generate"

    def generate(self, prompt: str) -> str:
        response = requests.post(
            self.url,
            json={
                "model": self.model_name,
                "prompt": prompt,
                "stream": False
            }
        )

        return response.json()["response"]
