import re
import streamlit as st

from io import BytesIO
from typing import List, Dict, Any
from pypdf import PdfReader

from langchain.llms import OpenAI
from langchain.docstore.document import Document
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import VectorStore
from langchain.vectorstores.faiss import FAISS

from embeddings import OpenAIEmbeddings


def parse_pdf(file: BytesIO) -> List[str]:
    """
    Parses a PDF file into a list of strings, one for each page.
    """
    pdf = PdfReader(file)
    output = []
    for page in pdf.pages:
        text = page.extract_text()
        # Merge hyphenated words
        text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)
        # Fix newlines in the middle of sentences
        text = re.sub(r"(?<!\n\s)\n(?!\s\n)", " ", text.strip())
        # Remove multiple newlines
        text = re.sub(r"\n\s*\n", "\n\n", text)

        output.append(text)

    return output


def text_to_docs(text: str | List[str]) -> List[Document]:
    """
    Converts a string or list of strings to a list of Documents
    with metadata.
    """
    if isinstance(text, str):
        # Take a single string as one page
        text = [text]
    page_docs = [Document(page_content=page) for page in text]

    # Add page numbers as metadata
    for i, doc in enumerate(page_docs):
        doc.metadata["page"] = i + 1

    # Split pages into chunks
    doc_chunks = []

    for doc in page_docs:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
            chunk_overlap=0,
        )
        chunks = text_splitter.split_text(doc.page_content)
        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk, metadata={"page": doc.metadata["page"], "chunk": i}
            )
            # Add sources a metadata
            doc.metadata["source"] = f"{doc.metadata['page']}-{doc.metadata['chunk']}"
            doc_chunks.append(doc)
    return doc_chunks


def embed_docs(docs: List[Document]) -> VectorStore:
    """
    Embeds a list of Documents and returns a FAISS index.
    """
    # Embed the chunks
    embeddings = OpenAIEmbeddings(
        openai_api_key=st.session_state.get("OPENAI_API_KEY")
    )  # type: ignore
    index = FAISS.from_documents(docs, embeddings)

    return index


def search_docs(index: VectorStore, query: str) -> List[Document]:
    """
    Searches a FAISS index for similar chunks to
    the query and returns a list of Documents.
    """

    # Search for similar chunks
    docs = index.similarity_search(query, k=5)
    return docs


def get_answer(_docs: List[Document], _query: str) -> Dict[str, Any]:
    """
    Gets an answer to a question from a list of Documents.
    """

    # Get the answer
    chain = load_qa_with_sources_chain(
        OpenAI(
            temperature=0,
            openai_api_key=st.session_state.get("OPENAI_API_KEY")
        ),
        chain_type="stuff",
    )
    answer = chain(
        {
            "input_documents": _docs,
            "question": _query
        },
        return_only_outputs=True
    )
    return answer


def get_sources(_answer: Dict[str, Any], _docs: List[Document]) -> List[Document]:
    """
    Gets the source documents for an answer.
    """

    # Get sources for the answer
    source_keys = [s for s in _answer["output_text"].split("SOURCES: ")[-1].split(", ")]

    source_docs = []
    for doc in _docs:
        if doc.metadata["source"] in source_keys:
            source_docs.append(doc)

    return source_docs

