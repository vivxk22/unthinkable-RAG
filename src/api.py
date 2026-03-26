import os
import shutil
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from src.search import RAGSearch

app = FastAPI(title="RAG Backend API")

DATA_DIR = "data"
FAISS_DIR = "faiss_store"

os.makedirs(DATA_DIR, exist_ok=True)

# Initialize RAG Engine globally
try:
    rag_engine = RAGSearch(persist_dir=FAISS_DIR)
except Exception as e:
    rag_engine = None
    print(f"Failed to initialize RAGEngine on startup: {e}")

class Message(BaseModel):
    role: str
    content: str
    context: Optional[List[str]] = None

class QueryRequest(BaseModel):
    query: str
    chat_history: Optional[List[Message]] = []

@app.post("/api/ingest")
async def ingest_documents(files: List[UploadFile] = File(...)):
    global rag_engine
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")
        
    for file in files:
        file_path = os.path.join(DATA_DIR, file.filename)
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
            
    # Remove old index to force rebuild
    if os.path.exists(FAISS_DIR):
        shutil.rmtree(FAISS_DIR)
        
    try:
        # This will trigger a re-build of the vectorstore
        rag_engine = RAGSearch(persist_dir=FAISS_DIR)
        return {"message": f"Successfully ingested {len(files)} files and rebuilt index."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to rebuild index: {str(e)}")

@app.post("/api/query")
async def query_documents(request: QueryRequest):
    if rag_engine is None:
        raise HTTPException(status_code=503, detail="RAG Engine not initialized or is still building.")
        
    # Convert Pydantic Models to dicts as expected by RAGSearch
    history_dicts = [{"role": msg.role, "content": msg.content} for msg in request.chat_history]
    
    try:
        response, context_chunks = rag_engine.search_and_summarize(
            query=request.query, 
            chat_history=history_dicts
        )
        return {
            "response": response,
            "context": context_chunks
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/files")
async def list_files():
    valid_extensions = ('.pdf', '.txt', '.csv', '.docx', '.xlsx', '.json', '.md')
    try:
        files = os.listdir(DATA_DIR)
        filtered_files = [f for f in files if f.endswith(valid_extensions)]
        return {"files": filtered_files}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not list directory: {str(e)}")

@app.delete("/api/files/{filename}")
async def delete_file(filename: str):
    global rag_engine
    file_path = os.path.join(DATA_DIR, filename)
    
    # Check for path traversal attacks or if file doesn't exist
    if not os.path.exists(file_path) or not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail="File not found")
        
    try:
        # Delete the actual file
        os.remove(file_path)
        
        # Rebuild vector database index
        if os.path.exists(FAISS_DIR):
            shutil.rmtree(FAISS_DIR)
            
        rag_engine = RAGSearch(persist_dir=FAISS_DIR)
        return {"message": f"Successfully deleted {filename} and rebuilt index."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete file and rebuild index: {str(e)}")
