from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from langchain_core.messages import HumanMessage

load_dotenv()

app = FastAPI(title="Simple Chatbot")

llm = ChatGoogleGenerativeAI(model="gemma-4-31b-it", google_api_key=os.getenv("GEMINI_API_KEY"), temperature=0.6)

class TopicRequest(BaseModel):
    topic: str

@app.post("/explain")
def explain_topic(request: TopicRequest):
    prompt = f"Explain this topic in simple terms: {request.topic}"
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"topic": request.topic, "explanation": response.content}

@app.get("/")
def serve_frontend():
    return FileResponse("index.html")