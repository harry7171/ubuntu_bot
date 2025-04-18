import os
from data_ingestion.chunking_md import MarkdownChunker
from data_ingestion.chroma_crud import ChromaVectorStore
from helpers.utils import get_all_md_files, load_md_to_text
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)


class DataIngestion:
    def __init__(self) -> None:
        print(f"Ingesting Data now")

    def ingest_data(self, path, vector_store_name):
        """
        Ingest data from the specified path and return a Create a vector store.
        """
        try:
            if not os.path.exists(path):
                raise FileNotFoundError(f"The path {path} does not exist.")

            md_file_paths = get_all_md_files(path)

            # Initialize the ChromaVectorStore
            vector_store = ChromaVectorStore(
                persist_directory="vector_stores/chroma_db",
                collection_name=vector_store_name,
            )

            chunks_list = []

            for md_file_path in md_file_paths:
                try:
                    logging.info(f"Processing: {md_file_path}")
                    file_name = os.path.basename(
                        md_file_path
                    )  # Extract the file name from the path
                    text = load_md_to_text(md_file_path)
                    chunker = MarkdownChunker()
                    chunks = chunker.chunk_markdown(
                        markdown_text=text, file_name=file_name
                    )

                    # Extract texts and metadata from chunks
                    texts = [chunk.page_content for chunk in chunks]
                    metadatas = [chunk.metadata for chunk in chunks]

                    # Add texts and metadata to the vector store
                    vector_store.add_texts(texts=texts, metadatas=metadatas)

                    chunks_list.extend(chunks)
                    print(f"Chunking done successfully")
                except Exception as e:
                    logging.error(f"Error processing {md_file_path}: {e}")

            # Persist the vector store
            vector_store.persist()

            print(f"total chunks - {len(chunks_list)}")
        except FileNotFoundError as fnf_error:
            logging.error(fnf_error)
            raise
        except Exception as e:
            logging.error(f"An unexpected error occurred during data ingestion: {e}")
            raise
