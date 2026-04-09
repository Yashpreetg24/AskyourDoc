import os
import shutil
import logging
from pathlib import Path
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# --------------- state ---------------
UPLOAD_DIR = Path("/tmp/uploads") if os.environ.get("RENDER") else Path("uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Global variables for models
vector_store = None
embeddings_model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models and check configuration at startup."""
    global embeddings_model
    
    logger.info("Starting up application...")
    
    # Check for API Key
    if not os.getenv("GROQ_API_KEY"):
        logger.warning("GROQ_API_KEY not found in environment variables!")
    
    try:
        logger.info("Pre-loading embeddings model: all-MiniLM-L6-v2...")
        # Pre-loading the model at startup to avoid request-time timeouts
        embeddings_model = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )
        logger.info("Embeddings model loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load embeddings model: {e}")
        # We don't raise here so the app can still start and show 
        # errors during runtime instead of failing to deploy.

    yield
    logger.info("Shutting down application...")


app = FastAPI(title="RAG Application", lifespan=lifespan)


# --------------- helpers ---------------

def load_document(file_path: str):
    """Load a PDF or text file and return LangChain Documents."""
    # TODO 
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith(".txt"):
        loader = TextLoader(file_path, encoding="utf-8")
    else:
        raise ValueError("Only .pdf and .txt files are supported.")
    
    return loader.load()


def build_vector_store(documents):
    """Split documents into chunks and build a FAISS vector store."""
    global embeddings_model
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(documents)

    # Use the pre-loaded model if available, otherwise initialize
    if embeddings_model is None:
        logger.info("Initializing embeddings model (it wasn't pre-loaded)...")
        embeddings_model = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )

    store = FAISS.from_documents(chunks, embeddings_model)
    return store


def get_qa_chain(store):
    """Create a RetrievalQA chain from the vector store."""
    # TODO 
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0
    )

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=store.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True
    )

    return chain

# --------------- routes ---------------

@app.get("/", response_class=HTMLResponse)
async def home():
    # TODO 
    return Path("static/index.html").read_text()


class QueryRequest(BaseModel):
    question: str


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    global vector_store

    if not file.filename:
        logger.error("Upload attempt with no filename")
        raise HTTPException(status_code=400, detail="No file uploaded.")

    ext = Path(file.filename).suffix.lower()
    if ext not in [".pdf", ".txt"]:
        logger.warning(f"Unsupported file type attempted: {ext}")
        raise HTTPException(status_code=400, detail="Only .pdf and .txt files are allowed.")

    safe_name = Path(file.filename).name
    file_path = UPLOAD_DIR / safe_name

    logger.info(f"Saving uploaded file: {safe_name}")
    try:
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
    except Exception as e:
        logger.error(f"Failed to save file: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")

    try:
        logger.info(f"Loading document: {safe_name}")
        documents = load_document(str(file_path))
        
        logger.info(f"Building vector store for {len(documents)} pages/docs")
        vector_store = build_vector_store(documents)
        
        logger.info("Vector store built successfully")
    except Exception as e:
        logger.error(f"Error during document processing: {e}")
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

    return {
        "message": f"'{safe_name}' uploaded and indexed successfully.",
        "pages": len(documents)
    }


@app.post("/query")
async def query_document(req: QueryRequest):
    #TODO 
    global vector_store

    if vector_store is None:
        raise HTTPException(
            status_code=400,
            detail="No document uploaded yet. Please upload a document first."
        )

    chain = get_qa_chain(vector_store)

    result = chain.invoke({"query": req.question})

    sources = []
    for doc in result.get("source_documents", []):
        sources.append({
            "content": doc.page_content[:300],
            "metadata": doc.metadata
        })

    return {
        "answer": result["result"],
        "sources": sources
    }

port = int(os.environ.get("PORT", 10000))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=port)