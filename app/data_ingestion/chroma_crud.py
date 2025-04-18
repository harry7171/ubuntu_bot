from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import logging


class ChromaVectorStore:
    def __init__(
        self,
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        persist_directory="./chroma_db",
        collection_name="default_collection",
    ):
        """
        Initialize the ChromaVectorStore with a Hugging Face model and LangChain's ChromaDB integration.
        :param model_name: Hugging Face model name for embeddings.
        :param persist_directory: Directory to persist the ChromaDB data.
        :param collection_name: Name of the ChromaDB collection.
        """
        try:
            self.embeddings = HuggingFaceEmbeddings(model_name=model_name)
            self.persist_directory = persist_directory
            self.collection_name = collection_name
            self.vector_store = Chroma(
                embedding_function=self.embeddings,
                persist_directory=self.persist_directory,
                collection_name=self.collection_name,
            )
        except Exception as e:
            logging.error(f"Failed to initialize ChromaVectorStore: {e}")
            raise

    def add_texts(self, texts, metadatas=None):
        """
        Add texts to the ChromaDB collection along with their metadata.
        :param texts: List of strings to be added.
        :param metadatas: List of metadata dictionaries corresponding to each text.
        """
        try:
            if metadatas is None:
                metadatas = [{} for _ in range(len(texts))]
            self.vector_store.add_texts(texts=texts, metadatas=metadatas)
        except Exception as e:
            logging.error(f"Failed to add texts to the vector store: {e}")
            raise

    def search(self, query, top_k=5):
        """
        Search for the most similar texts to the query.
        :param query: Query string.
        :param top_k: Number of top results to return.
        :return: List of tuples (text, metadata, score).
        """
        try:
            results = self.vector_store.similarity_search_with_score(query, k=top_k)
            return [
                (result[0].page_content, result[0].metadata, result[1])
                for result in results
            ]
        except Exception as e:
            logging.error(f"Search failed for query: '{query}': {e}")
            raise

    def persist(self):
        """
        Persist the ChromaDB data to the specified directory.
        """
        try:
            self.vector_store.persist()
        except Exception as e:
            logging.error(f"Failed to persist the vector store: {e}")
            raise

    @staticmethod
    def load_vs(
        persist_directory,
        collection_name,
        model_name="sentence-transformers/all-MiniLM-L6-v2",
    ):
        """
        Load an existing ChromaDB collection by name.
        :param persist_directory: Directory where the ChromaDB data is stored.
        :param collection_name: Name of the ChromaDB collection to load.
        :param model_name: Hugging Face model name for embeddings.
        :return: An instance of ChromaVectorStore.
        """
        try:
            embeddings = HuggingFaceEmbeddings(model_name=model_name)
            vector_store = Chroma(
                embedding_function=embeddings,
                persist_directory=persist_directory,
                collection_name=collection_name,
            )
            return ChromaVectorStore(
                model_name=model_name,
                persist_directory=persist_directory,
                collection_name=collection_name,
            )
        except Exception as e:
            logging.error(f"Failed to load vector store: {e}")
            raise

