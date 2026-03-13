from fastapi import FastAPI, UploadFile, File, APIRouter
from pydantic import BaseModel
from core.logger.logger import logger
import time
import shutil
import os
from fastapi.responses import StreamingResponse

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

    start_time = time.time()
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

    # ---------- Process File ----------
    if filename.endswith(".pdf"):
        text = processor.extract_text(save_path)
        txt_filename = filename.replace(".pdf", ".txt")

        txt_path = os.path.join(
            DATA_FOLDER,
            txt_filename
        )

        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(text)

        os.remove(save_path)
        final_filename = txt_filename

    else:
        # Already TXT
        final_filename = filename

    # ---------- Rebuild Index ----------

    rag.rebuild_index()

    latency = round(time.time() - start_time, 3)

    # ---------- Logging ----------

    logger.info(
        f"UPLOAD | "
        f"OriginalFile={filename} | "
        f"StoredAs={final_filename} | "
        f"Mode={'HYBRID' if rag.use_hybrid else 'VECTOR'} | "
        f"DocumentsIndexed={len(os.listdir(DATA_FOLDER))} | "
        f"RebuildTime={latency}s"
    )

    return {
        "message": "Document uploaded successfully",
        "saved_as": final_filename
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

@router.post("/ask-stream", tags=["RAG Query Engine"])
async def ask_stream(request: QueryRequest):

    question = request.question
    
    def stream():
        for token in rag.generate_stream(question):
            yield token

    return StreamingResponse(
        stream(),
        media_type="text/plain"
    )

# Register Router
app.include_router(router)