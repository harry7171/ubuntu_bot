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


def build_prompt(prompt_template_name: str) -> str:
    """
    Builds a formatted prompt based on a template and provided keyword arguments.

    Args:
        prompt_template_name (str): The name of the prompt template to use.

    Returns:
        str: A prompt string based on the specified template and provided arguments.

    Example:
        >>> prompt = build_prompt("greeting")
        >>> print(prompt)
        "Hello! Welcome to our service."
    """
    root_path = Path(__file__).parent.parent / "prompts"
    # root_path = Path(__file__).parent
    with open(f"{root_path}\\prompts.yaml", encoding="utf8") as prompts_yaml:
        prompts = yaml.safe_load(prompts_yaml)
    prompt_template = prompts[prompt_template_name]
    return prompt_template


def construct_messages(system_prompt, query, contexts):
    """
    Construct messages for an LLM based on the system prompt, user query, and retrieved results.
    :param system_prompt: The system's initial instruction or context.
    :param query: The user's query string.
    :param results: A list of tuples (content, metadata, score) retrieved from the vector store.
    :return: A list of messages formatted for the LLM.
    """
    # Construct the system message
    messages = [{"role": "system", "content": system_prompt}]

    # Add the user's query as a user message
    messages.append({"role": "user", "content": query})

    # Add the retrieved results' content as assistant messages
    for idx, (content, metadata, score) in enumerate(contexts, start=1):
        messages.append({"role": "user", "content": f"Context {idx}: {content}"})

    return messages


def generate(messages, stream=True):

    client = InferenceClient(
        provider="sambanova",
        api_key=HF_API_KEY,
    )
    completion = client.chat.completions.create(
        model="meta-llama/Llama-3.1-8B-Instruct",
        messages=messages,
        max_tokens=512,
        stream=stream,
    )
    # print(completion.choices[0].message)
    if not stream:
        answer = completion.choices[0].message.content
        return answer
    else:
        for chunk in completion:
            if chunk.choices[0].delta.content:
                print(chunk.choices[0].delta.content, end="", flush=True)
                time.sleep(0.5)
        print("\n")
    return None
