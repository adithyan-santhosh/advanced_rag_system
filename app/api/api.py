from fastapi import FastAPI, UploadFile, File, APIRouter
from pydantic import BaseModel
import shutil
import os

from core.pipeline.rag_pipeline import RAGPipeline
from core.processor.document_processor import DocumentProcessor


# FastAPI App
app = FastAPI(
    title="Adaptive Hybrid RAG API",
    description="Production-ready Adaptive Hybrid RAG System",
    version="1.0",

)

# Router with Base Path
router = APIRouter(prefix="/rag-system/v1")


# Paths
DATA_FOLDER = "data"
STORAGE_FOLDER = "storage"

os.makedirs(DATA_FOLDER, exist_ok=True)
os.makedirs(STORAGE_FOLDER, exist_ok=True)



# Initialize Pipeline
rag = RAGPipeline(
    data_folder=DATA_FOLDER,
    storage_path=STORAGE_FOLDER
)

processor = DocumentProcessor()



# Request Model
class QueryRequest(BaseModel):
    question: str



# ASK Endpoint

@router.post("/ask", tags=["RAG Query Engine"])
def ask_question(request: QueryRequest):

    result = rag.generate_answer(
        query=request.question,
        top_k=3,
        confidence_threshold=0.0
    )

    return result



# UPLOAD Endpoint

@router.post("/upload", tags=["Document Management"])
async def upload_file(file: UploadFile = File(...)):

    filename = file.filename

    save_path = os.path.join(
        DATA_FOLDER,
        filename
    )

    # Save uploaded file

    with open(save_path, "wb") as buffer:

        shutil.copyfileobj(
            file.file,
            buffer
        )

    # Extract text

    try:
        text = processor.extract_text(save_path)
    except Exception as e:
        return {
            "error": str(e)
        }

    # Save extracted text

    txt_filename = filename + ".txt"

    txt_path = os.path.join(
        DATA_FOLDER,
        txt_filename
    )

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(text)

    # Remove PDF after extraction

    if save_path.endswith(".pdf"):
        os.remove(save_path)

    # Rebuild index

    rag.rebuild_index()
    return {
        "message": "Document uploaded successfully",
        "saved_as": txt_filename
    }

# HEALTH Endpoint

@router.get("/health")
def health_check():

    return {
        "status": "running",
        "service": "rag-system",
        "documents_folder": DATA_FOLDER,
        "storage_folder": STORAGE_FOLDER
    }

# Register Router
app.include_router(router)