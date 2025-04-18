import warnings
from helpers.utils import build_prompt, construct_messages, generate
import logging

# Suppress warnings and import messages
warnings.filterwarnings("ignore")

# # Configure logging
# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s - %(levelname)s - %(message)s",
#     handlers=[logging.StreamHandler()],
# )


def handle_conversation(vector_store):
    """
    Handles the conversation loop with the user.
    :param vector_store: The loaded vector store for searching.
    """
    conversation_context = []  # To store the conversation history if needed

    while True:
        try:
            query = input("Enter your query : ").strip()
            print(f"Query: {query}")
            if query.lower() == "exit":
                print("Exiting the QnA system. Goodbye!")
                break

            # Call the vector store to search for the query
            try:
                top_chunks = vector_store.search(query, top_k=10)
            except Exception as e:
                logging.error(f"Error during vector store search: {e}")
                print("An error occurred while searching. Please try again.")
                continue

            rag_system_prompt = build_prompt("ubuntuBot")

            # Add previous context if continuing the conversation
            if conversation_context:
                previous_context = "\n".join(
                    [
                        f"User: {ctx['query']}\nAssistant: {ctx['answer']}"
                        for ctx in conversation_context
                    ]
                )
                rag_system_prompt += f"\nPrevious Context:\n{previous_context}"

            # Construct messages for the LLM
            messages = construct_messages(
                system_prompt=rag_system_prompt, query=query, contexts=top_chunks
            )
            answer = generate(messages=messages)
            if answer:
                print(answer)

            # Extract unique file names from top_chunks
            unique_file_names = {
                chunk[1].get("file_name")
                for chunk in top_chunks
                if "file_name" in chunk[1]
            }
            print(f"Source files: {unique_file_names}")

            # Append the current query and answer to the conversation context
            conversation_context.append({"query": query, "answer": answer})

            # Prompt the user for the next action
            while True:
                next_action = (
                    input("What would you like to do next? (start/ continue / stop): ")
                    .strip()
                    .lower()
                )
                if next_action == "start":
                    print("Starting a new conversation...")
                    conversation_context = []  # Clear the conversation context
                    break
                elif next_action == "continue":
                    print("Continuing the current conversation...")
                    break
                elif next_action == "stop":
                    print("Exiting the Chatbot. Goodbye!")
                    exit()
                else:
                    print("Invalid input. Please type 'start', 'continue', or 'stop'.")
        except Exception as e:
            logging.error(f"An unexpected error occurred in the conversation loop: {e}")
            print(f"An unexpected error occurred. Please try again.- {e}")
