from openai.error import OpenAIError
from utils import (
    parse_pdf,
    text_to_docs,
    embed_docs,
    search_docs,
    get_answer,
    get_sources,
)
import os
import streamlit as st

st.set_page_config(page_title="DataQuill🪶", page_icon="🔍", layout="wide")


# OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

QUERY = """Please provide information on the dataset and data
collection methods used in the research paper. Additionally,
identify and fetch the citations or references that are specifically
related to the dataset."""


st.title("🔍 DataQuill🪶")


def clear_submit():
    st.session_state["submit"] = False


with st.sidebar:
    st.markdown("# About")
    st.markdown("""
        DataQuill is an AI-powered tool using Large Language Models(LLMs)
        to extract datasets from research papers.
        """)
    st.markdown("---")
    st.markdown("""
        Source code:
        [vchrombie/dataquill](https://github.com/vchrombie/dataquill)
        """)
    st.markdown("""
        Developed DataQuill tool as a part of my participation in the seminar-styled
        course CS-GY 9223: Special Topics - [Dataset Search and Discovery](https://vida-nyu.github.io/dataset-search-and-discovery-seminar/)
        under [Prof. Juliana Freire](https://engineering.nyu.edu/faculty/juliana-freire)
        and [NYU VIDA](https://vida.engineering.nyu.edu/).
        """)
    st.markdown("📍 Made at NYU Tandon!")
    st.markdown("---")
    st.markdown("Built with 🦜️🔗 LangChain, 🤖️ OpenAI, and 🐍️ Streamlit.")
    st.markdown("---")

uploaded_file = st.file_uploader(
    "Upload a research paper (.pdf)",
    type=["pdf"],
    help="Scanned documents are not supported yet!",
    on_change=clear_submit,
)

index = None
doc = None
if uploaded_file is not None:
    if uploaded_file.name.endswith(".pdf"):
        doc = parse_pdf(uploaded_file)
    else:
        raise ValueError("File type not supported!")

    text = text_to_docs(doc)

    try:
        with st.spinner("Indexing document... ⏳"):
            index = embed_docs(text)
        st.session_state["api_key_configured"] = True
    except OpenAIError as e:
        st.error(e._message)

if not index:
    st.error("Please upload a document!!!")
else:
    st.session_state["submit"] = True

    dataset, mentions = st.columns(2)
    with st.spinner("Generating results... ⏳"):
        sources = search_docs(index, QUERY)

    try:
        with st.spinner("Writing the results... ⏳"):
            answer = get_answer(sources, QUERY)
            sources = get_sources(answer, sources)

        with dataset:
            st.markdown("#### Dataset")
            st.markdown(answer["output_text"].split("SOURCES: ")[0])

        with mentions:
            st.markdown("#### Mentions")
            for source in sources:
                st.markdown(source.page_content)
                st.markdown(source.metadata["source"])
                st.markdown("---")

    except OpenAIError as e:
        st.error(e._message)
