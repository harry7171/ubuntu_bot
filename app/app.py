import warnings
import logging
from dotenv import load_dotenv
import os
from data_ingestion.data_pipeline import DataIngestion

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

vs_name = os.getenv("VECTOR_STORE_NAME", "ubuntu_store")


if __name__ == "__main__":
    try:
        
        root_folder = "c:\\Personal\\demo_bot_data\\demo_bot_data\\ubuntu_bot\\app"
        data_obj = DataIngestion()
        data_obj.ingest_data(path=root_folder, vector_store_name=vs_name)
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}", exc_info=True)
        print(f"An unexpected error occurred. Please check the logs for details. - {e}")
