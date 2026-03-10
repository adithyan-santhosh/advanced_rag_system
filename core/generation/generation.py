import requests
import json

class OllamaLLM:
    def __init__(self, model_name="phi3:mini"):
        self.model_name = model_name
        self.url = "http://localhost:11434/api/generate"

    def generate(self, prompt: str) -> str:
        response = requests.post(
            self.url,
            json={
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.2,
                    "num_predict": 200,
                    "num_ctx": 4096,
                    "top_p": 0.9
                }
            }
        )

        return response.json()["response"]
    
    def generate_stream(self, prompt: str):

        response = requests.post(
            self.url,
            json={
                "model": self.model_name,
                "prompt": prompt,
                "stream": True,
                "options": {
                    "temperature": 0.2,
                    "num_predict": 200,
                    "num_ctx": 4096
                }
            },
            stream=True
        )

        for line in response.iter_lines():
            if line:
                data = json.loads(line.decode("utf-8"))
                if "response" in data:
                    yield data["response"]
