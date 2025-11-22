import gradio as gr
import requests
import os
from main import app as fastapi_app  # Your existing FastAPI app
import uvicorn
from threading import Thread

# Start FastAPI in background
def run_fastapi():
    uvicorn.run(fastapi_app, host="0.0.0.0", port=7860)

Thread(target=run_fastapi, daemon=True).start()

# Gradio Interface
def upload_and_ask(file, question):
    if file is None:
        return "Please upload a document first!"
    
    # Upload document
    files = {"file": open(file.name, "rb")}
    upload_response = requests.post("http://localhost:7860/upload", files=files)
    
    if upload_response.status_code != 200:
        return f"Upload failed: {upload_response.text}"
    
    # Ask question
    query_response = requests.post(
        "http://localhost:7860/query",
        json={"query": question}
    )
    
    if query_response.status_code == 200:
        return query_response.json().get("answer", "No answer found")
    return f"Query failed: {query_response.text}"

# Create Gradio interface
demo = gr.Interface(
    fn=upload_and_ask,
    inputs=[
        gr.File(label="Upload Document (PDF, TXT, CSV, DOCX)", file_types=[".pdf", ".txt", ".csv", ".docx"]),
        gr.Textbox(label="Ask a Question", placeholder="What would you like to know about the document?")
    ],
    outputs=gr.Textbox(label="Answer"),
    title="📚 RAG Document Q&A System",
    description="Upload a document and ask questions based on its content!",
    examples=[
        [None, "What is the main topic of this document?"],
        [None, "Summarize the key points"],
    ]
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)