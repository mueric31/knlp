from fastapi import FastAPI
from .chat import get_response

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Welcome to the Chatbot API"}

@app.get("/chat")
def chat(query: str):
    return {"response": get_response(query)}

# optional: POST endpoint
from pydantic import BaseModel
class ChatIn(BaseModel):
    query: str

@app.post("/chat")
def chat_post(body: ChatIn):
    return {"response": get_response(body.query)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=1000)
