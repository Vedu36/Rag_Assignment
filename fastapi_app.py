import os
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import shutil
from dotenv import load_dotenv
import PyPDF2
import docx
import pandas as pd
from rag_engine import RAGEngine

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="RAG Assistant", description="Document Q&A System using RAG")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG Engine
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in environment variables")

rag_engine = RAGEngine(groq_api_key=GROQ_API_KEY)

# Create uploads directory
os.makedirs("uploads", exist_ok=True)

# File processing functions
def extract_text_from_txt(file_path: str) -> str:
    """Extract text from TXT file"""
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        return f.read()

def extract_text_from_pdf(file_path: str) -> str:
    """Extract text from PDF file"""
    text = ""
    with open(file_path, 'rb') as f:
        pdf_reader = PyPDF2.PdfReader(f)
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
    return text

def extract_text_from_docx(file_path: str) -> str:
    """Extract text from DOCX file"""
    doc = docx.Document(file_path)
    return "\n".join([paragraph.text for paragraph in doc.paragraphs])

def extract_text_from_csv(file_path: str) -> str:
    """Extract text from CSV file"""
    df = pd.read_csv(file_path)
    return df.to_string()

def process_file(file_path: str, filename: str) -> dict:
    """Process uploaded file and extract text"""
    extension = filename.lower().split('.')[-1]
    
    try:
        if extension == 'txt':
            text = extract_text_from_txt(file_path)
        elif extension == 'pdf':
            text = extract_text_from_pdf(file_path)
        elif extension == 'docx':
            text = extract_text_from_docx(file_path)
        elif extension == 'csv':
            text = extract_text_from_csv(file_path)
        else:
            raise ValueError(f"Unsupported file type: {extension}")
        
        return {'text': text, 'filename': filename}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing {filename}: {str(e)}")

# API Routes
@app.get("/", response_class=HTMLResponse)
async def home():
    """Serve the main HTML interface"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>RAG Assistant - Document Q&A</title>
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                padding: 20px;
            }
            .container {
                max-width: 1200px;
                margin: 0 auto;
                background: white;
                border-radius: 12px;
                box-shadow: 0 20px 60px rgba(0,0,0,0.3);
                overflow: hidden;
            }
            .header {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 30px;
                text-align: center;
            }
            .header h1 { font-size: 32px; margin-bottom: 10px; }
            .header p { opacity: 0.9; }
            .content {
                padding: 30px;
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 30px;
            }
            .section {
                background: #f8f9fa;
                padding: 25px;
                border-radius: 8px;
                border: 1px solid #e0e0e0;
            }
            .section h2 {
                color: #667eea;
                margin-bottom: 20px;
                font-size: 20px;
            }
            .upload-area {
                border: 2px dashed #667eea;
                border-radius: 8px;
                padding: 30px;
                text-align: center;
                background: white;
                cursor: pointer;
                transition: all 0.3s;
            }
            .upload-area:hover {
                background: #f0f4ff;
                border-color: #764ba2;
            }
            input[type="file"] { display: none; }
            .btn {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border: none;
                padding: 12px 30px;
                border-radius: 6px;
                cursor: pointer;
                font-size: 16px;
                font-weight: 600;
                margin-top: 15px;
                transition: transform 0.2s;
            }
            .btn:hover { transform: translateY(-2px); }
            .btn:disabled {
                background: #ccc;
                cursor: not-allowed;
                transform: none;
            }
            textarea {
                width: 100%;
                padding: 15px;
                border: 2px solid #e0e0e0;
                border-radius: 6px;
                font-size: 14px;
                resize: vertical;
                min-height: 100px;
                font-family: inherit;
            }
            textarea:focus {
                outline: none;
                border-color: #667eea;
            }
            .result {
                background: white;
                padding: 20px;
                border-radius: 8px;
                margin-top: 20px;
                border-left: 4px solid #667eea;
            }
            .result h3 {
                color: #667eea;
                margin-bottom: 10px;
            }
            .source {
                background: #f8f9fa;
                padding: 15px;
                border-radius: 6px;
                margin-top: 10px;
                border-left: 3px solid #764ba2;
            }
            .source strong { color: #764ba2; }
            .stats {
                display: flex;
                gap: 20px;
                margin-bottom: 20px;
            }
            .stat-card {
                flex: 1;
                background: white;
                padding: 15px;
                border-radius: 6px;
                text-align: center;
            }
            .stat-card .number {
                font-size: 32px;
                font-weight: bold;
                color: #667eea;
            }
            .stat-card .label {
                color: #666;
                margin-top: 5px;
            }
            .file-list {
                background: white;
                padding: 15px;
                border-radius: 6px;
                margin-top: 15px;
                max-height: 200px;
                overflow-y: auto;
            }
            .file-item {
                padding: 8px;
                background: #f8f9fa;
                margin-bottom: 5px;
                border-radius: 4px;
                font-size: 14px;
            }
            .loading {
                display: none;
                text-align: center;
                margin-top: 20px;
            }
            .loading.active { display: block; }
            .spinner {
                border: 3px solid #f3f3f3;
                border-top: 3px solid #667eea;
                border-radius: 50%;
                width: 40px;
                height: 40px;
                animation: spin 1s linear infinite;
                margin: 0 auto;
            }
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
            @media (max-width: 768px) {
                .content { grid-template-columns: 1fr; }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🤖 RAG Assistant</h1>
                <p>Upload documents and ask questions - Powered by Groq & FAISS</p>
            </div>
            
            <div class="content">
                <!-- Upload Section -->
                <div class="section">
                    <h2>📁 Upload Documents</h2>
                    <div class="stats">
                        <div class="stat-card">
                            <div class="number" id="docCount">0</div>
                            <div class="label">Documents</div>
                        </div>
                        <div class="stat-card">
                            <div class="number" id="chunkCount">0</div>
                            <div class="label">Chunks</div>
                        </div>
                    </div>
                    
                    <div class="upload-area" onclick="document.getElementById('fileInput').click()">
                        <p style="font-size: 48px; margin-bottom: 10px;">📄</p>
                        <p style="font-weight: 600; color: #667eea;">Click to select files</p>
                        <p style="color: #666; font-size: 14px; margin-top: 5px;">
                            Supports: TXT, PDF, CSV, DOCX
                        </p>
                    </div>
                    <input type="file" id="fileInput" multiple 
                           accept=".txt,.pdf,.csv,.docx" onchange="uploadFiles()">
                    
                    <div class="file-list" id="fileList" style="display: none;">
                        <strong>Selected files:</strong>
                        <div id="selectedFiles"></div>
                    </div>
                    
                    <button class="btn" onclick="clearIndex()" style="background: #dc3545; margin-top: 20px;">
                        Clear All Documents
                    </button>
                </div>
                
                <!-- Query Section -->
                <div class="section">
                    <h2>💬 Ask Questions</h2>
                    <textarea id="question" placeholder="Type your question here..."></textarea>
                    <button class="btn" onclick="askQuestion()">Ask Question</button>
                    
                    <div class="loading" id="loading">
                        <div class="spinner"></div>
                        <p style="margin-top: 10px; color: #667eea;">Processing...</p>
                    </div>
                    
                    <div id="answer"></div>
                </div>
            </div>
        </div>
        
        <script>
            // Load stats on page load
            loadStats();
            
            async function loadStats() {
                try {
                    const response = await fetch('/stats');
                    const data = await response.json();
                    document.getElementById('docCount').textContent = data.total_documents;
                    document.getElementById('chunkCount').textContent = data.total_chunks;
                } catch (error) {
                    console.error('Error loading stats:', error);
                }
            }
            
            async function uploadFiles() {
                const fileInput = document.getElementById('fileInput');
                const files = fileInput.files;
                
                if (files.length === 0) return;
                
                // Show selected files
                const fileList = document.getElementById('fileList');
                const selectedFiles = document.getElementById('selectedFiles');
                selectedFiles.innerHTML = '';
                for (let file of files) {
                    const div = document.createElement('div');
                    div.className = 'file-item';
                    div.textContent = file.name;
                    selectedFiles.appendChild(div);
                }
                fileList.style.display = 'block';
                
                const formData = new FormData();
                for (let file of files) {
                    formData.append('files', file);
                }
                
                try {
                    const response = await fetch('/upload', {
                        method: 'POST',
                        body: formData
                    });
                    
                    const result = await response.json();
                    if (response.ok) {
                        alert(`✅ Successfully uploaded ${result.uploaded_count} file(s)`);
                        fileInput.value = '';
                        fileList.style.display = 'none';
                        loadStats();
                    } else {
                        alert('❌ Error: ' + result.detail);
                    }
                } catch (error) {
                    alert('❌ Error uploading files: ' + error);
                }
            }
            
            async function askQuestion() {
                const question = document.getElementById('question').value.trim();
                if (!question) {
                    alert('Please enter a question');
                    return;
                }
                
                const loading = document.getElementById('loading');
                const answerDiv = document.getElementById('answer');
                loading.classList.add('active');
                answerDiv.innerHTML = '';
                
                try {
                    const response = await fetch('/query', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ question })
                    });
                    
                    const result = await response.json();
                    loading.classList.remove('active');
                    
                    if (response.ok) {
                        let html = `
                            <div class="result">
                                <h3>Answer:</h3>
                                <p>${result.answer}</p>
                            </div>
                        `;
                        
                        if (result.sources && result.sources.length > 0) {
                            html += '<div class="result"><h3>Sources:</h3>';
                            result.sources.forEach((source, index) => {
                                html += `
                                    <div class="source">
                                        <strong>Source ${index + 1}:</strong> ${source.filename}<br>
                                        <small>${source.text_snippet}</small>
                                    </div>
                                `;
                            });
                            html += '</div>';
                        }
                        
                        answerDiv.innerHTML = html;
                    } else {
                        answerDiv.innerHTML = `<div class="result" style="border-color: #dc3545;">
                            <h3 style="color: #dc3545;">Error:</h3>
                            <p>${result.detail}</p>
                        </div>`;
                    }
                } catch (error) {
                    loading.classList.remove('active');
                    answerDiv.innerHTML = `<div class="result" style="border-color: #dc3545;">
                        <h3 style="color: #dc3545;">Error:</h3>
                        <p>${error}</p>
                    </div>`;
                }
            }
            
            async function clearIndex() {
                if (!confirm('Are you sure you want to clear all documents?')) return;
                
                try {
                    const response = await fetch('/clear', { method: 'POST' });
                    const result = await response.json();
                    alert(result.message);
                    loadStats();
                    document.getElementById('answer').innerHTML = '';
                } catch (error) {
                    alert('Error clearing index: ' + error);
                }
            }
            
            // Allow Enter key to submit question (Shift+Enter for new line)
            document.getElementById('question').addEventListener('keydown', function(e) {
                if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    askQuestion();
                }
            });
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.post("/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    """Upload and process multiple files"""
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    
    documents = []
    uploaded_count = 0
    
    for file in files:
        # Check file extension
        allowed_extensions = ['txt', 'pdf', 'csv', 'docx']
        file_ext = file.filename.split('.')[-1].lower()
        
        if file_ext not in allowed_extensions:
            continue
        
        # Save file temporarily
        file_path = f"uploads/{file.filename}"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Process file
        try:
            doc = process_file(file_path, file.filename)
            documents.append(doc)
            uploaded_count += 1
        except Exception as e:
            print(f"Error processing {file.filename}: {str(e)}")
        finally:
            # Clean up
            if os.path.exists(file_path):
                os.remove(file_path)
    
    if documents:
        rag_engine.add_documents(documents)
    
    return JSONResponse({
        "message": f"Successfully uploaded {uploaded_count} file(s)",
        "uploaded_count": uploaded_count
    })

@app.post("/query")
async def query(request: dict):
    """Query the RAG system"""
    question = request.get("question", "").strip()
    
    if not question:
        raise HTTPException(status_code=400, detail="Question is required")
    
    result = rag_engine.query(question)
    return JSONResponse(result)

@app.get("/stats")
async def get_stats():
    """Get system statistics"""
    stats = rag_engine.get_stats()
    return JSONResponse(stats)

@app.post("/clear")
async def clear_index():
    """Clear all documents from the index"""
    rag_engine.clear_index()
    return JSONResponse({"message": "All documents cleared successfully"})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)