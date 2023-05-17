import streamlit as st

st.set_page_config(page_title="📊🪶 DataQuill", layout="wide")

st.title("📊🪶 DataQuill")

def clear_submit():
    st.session_state["submit"] = False

with st.sidebar:
    st.markdown("# About")
    st.markdown(
        "DataQuill is an AI-powered tool using Large Language Models (LLMs) to extract datasets from research papers. It transforms unstructured information into structured data, streamlining research, and making data analysis simpler, faster, and more reliable for everyone."
    )
    st.markdown("---")
    st.markdown(
        "Source code: [vchrombie/dataquill](https://github.com/vchrombie/dataquill) "
    )
    st.markdown("---")

uploaded_file = st.file_uploader(
    "Upload a research paper (.pdf)",
    type=["pdf"],
    help="Scanned documents are not supported yet!",
    on_change=clear_submit,
)
