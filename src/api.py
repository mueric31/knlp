from fastapi import FastAPI
from chatbot import get_response
import uvicorn

app = FastAPI()

@app.post("/chat/")
async def chat(query: str):
    response = get_response(query)
    return {"response": response}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
