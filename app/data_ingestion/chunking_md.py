from langchain.text_splitter import MarkdownHeaderTextSplitter


class MarkdownChunker:
    def __init__(self):
        self.headers_to_split_on = [
            ("#", "header_1"),
            ("##", "header_2"),
            ("###", "header_3"),
            ("####", "header_4"),
        ]
        self.splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=self.headers_to_split_on
        )

    def chunk_markdown(self, markdown_text, file_name):
        """
        Splits the given markdown text into chunks based on headers.

        Args:
            markdown_text (str): The markdown text to split.

        Returns:
            list: A list of chunks with their content and metadata.
        """
        docs = self.splitter.split_text(markdown_text)
        for i, doc in enumerate(docs):
            headings = doc.metadata
            finalhead = ""
            for key, value in headings.items():
                finalhead += value + "\n"
            doc.page_content = finalhead + "\n" + doc.page_content
            doc.metadata["file_name"] = file_name
            print(f"--- Chunk {i+1} ---")
            print(doc.page_content)
            print(doc.metadata)

        return docs
