from data_ingestion.chroma_crud import ChromaVectorStore
from rag_flow.qna import handle_conversation
import warnings
import logging
from dotenv import load_dotenv
import os

warnings.filterwarnings("ignore")
# Configure logging
logging.basicConfig(
    filename="app.log",  # Log file name
    level=logging.INFO,  # Logging level
    format="%(asctime)s - %(levelname)s - %(message)s",  # Log format
)

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

vs_name = os.getenv("VECTOR_STORE_NAME", "ubuntu_store")


if __name__ == "__main__":
    try:
        print("Starting Demo Bot..")
        # Load the vector store
        loaded_vector_store = ChromaVectorStore.load_vs(
            persist_directory="vector_stores/chroma_db", collection_name=vs_name
        )
        print(f"Started Demo bot.")
        handle_conversation(loaded_vector_store)
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}", exc_info=True)
        print(f"An unexpected error occurred. Please check the logs for details. - {e}")
