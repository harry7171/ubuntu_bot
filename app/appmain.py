from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from data_ingestion.data_pipeline import DataIngestion
from data_ingestion.chroma_crud import ChromaVectorStore
from helpers.utils import build_prompt, construct_messages, generate
import uvicorn
from dotenv import load_dotenv
import os
import warnings

warnings.filterwarnings("ignore")
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

# Retrieve environment variables
applicationPort = int(os.getenv("PORT", 8000))  # Default to 8000 if not set
vs_name = os.getenv(
    "VECTOR_STORE_NAME", "ubuntu_store"
)  # Default to "ubuntu_store" if not set


app = FastAPI()


class RetrievalRequest(BaseModel):
    query: str


@app.post("/ingest-data")
async def ingest_data_endpoint():
    """
    Endpoint to trigger data ingestion.
    """
    try:
        # Specify the root folder to search for .md files
        root_folder = "c:\\Personal\\demo_bot_data\\demo_bot_data\\ubuntu_bot\\app"
        data_obj = DataIngestion()
        data_obj.ingest_data(path=root_folder, vector_store_name=vs_name)
        return {"message": "Data ingestion completed successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/qna")
async def retrieval_endpoint(request: RetrievalRequest):
    """
    Endpoint to perform retrieval from the vector store.
    """
    try:
        vs_name = "ubuntu_store"
        loaded_vector_store = ChromaVectorStore.load_vs(
            persist_directory="vector_stores/chroma_db", collection_name=vs_name
        )
        top_chunks = loaded_vector_store.search(request.query, top_k=10)
        rag_system_prompt = build_prompt("ubuntuBot")
        messages = construct_messages(
            system_prompt=rag_system_prompt, query=request.query, contexts=top_chunks
        )
        answer = generate(messages=messages, stream=False)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Run the application
if __name__ == "__main__":
    uvicorn.run(
        "appmain:app",
        host="0.0.0.0",
        port=applicationPort,
        reload=True,
        workers=1,
    )
