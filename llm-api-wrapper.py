from fastapi import FastAPI
from pydantic import BaseModel
from langchain_ollama import OllamaLLM

app = FastAPI(title="Llama 3.2 API Wrapper")

# Initialize your local model
llm = OllamaLLM(model="llama3.2")

# Define what the incoming request looks like
class Query(BaseModel):
    prompt: str
    temperature: float = 0.7

@app.post("/generate")
async def generate_response(query: Query):
    # Logic to talk to your local LLM
    response = llm.invoke(query.prompt)
    return {"status": "success", "response": response}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)