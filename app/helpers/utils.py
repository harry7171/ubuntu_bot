import os
from pathlib import Path
import yaml
from huggingface_hub import InferenceClient
import time
from dotenv import load_dotenv
import openai

env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path)
HF_API_KEY = os.getenv("HF_API_KEY")


def get_all_md_files(root_folder):
    """
    Recursively find all .md files in the given root folder.
    """
    md_files = []
    for dirpath, _, filenames in os.walk(root_folder):
        print(f"Searching in: {dirpath}")
        print(f"Files found: {filenames}")
        for file in filenames:
            if file.endswith(".md"):
                md_files.append(os.path.join(dirpath, file))
        print(f"============================")
    return md_files


def load_md_to_text(file_path):
    """
    Load a Markdown file and return its content as plain text.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")

    with open(file_path, "r", encoding="utf-8") as md_file:
        plain_text = md_file.read()

    return plain_text
