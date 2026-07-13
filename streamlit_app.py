# pyrefly: ignore [missing-import]
import streamlit as st

from app.document_processor import DocumentProcessor
from app.rag_pipeline import RAGPipeline


st.set_page_config(
    page_title="BookWise AI",
    page_icon="📚"
)


st.title("📚 BookWise AI")
st.write(
    "Ask questions about your uploaded documents."
)


# Initialize pipeline once
@st.cache_resource
def load_pipeline():
    return RAGPipeline()


try:
    pipeline = load_pipeline()

except Exception as e:
    st.error(f"Unable to initialize RAG pipeline: {e}")
    st.stop()


uploaded_files = st.file_uploader(
    "Upload PDFs",
    type=["pdf"],
    accept_multiple_files=True
)


if uploaded_files:

    if st.button("Index Document"):

        with st.spinner("Processing documents..."):

            try:

                # Clear previous index so we only index the currently uploaded files
                pipeline.clear_index()

                for uploaded_file in uploaded_files:
                    file_path = (
                        f"data/{uploaded_file.name}"
                    )

                    with open(
                        file_path,
                        "wb"
                    ) as f:
                        f.write(
                            uploaded_file.getbuffer()
                        )


                    processor = DocumentProcessor()


                    # FIXED: extract_pages -> extract_text
                    pages = processor.extract_text(
                        file_path
                    )


                    pipeline.index_document(
                        pages,
                        uploaded_file.name
                    )


                st.success(
                    "Documents indexed successfully!"
                )


            except Exception as e:

                st.error(
                    f"Document processing failed: {e}"
                )



question = st.text_input(
    "Ask a question"
)


if st.button("Ask"):

    if question:

        with st.spinner("Searching..."):

            try:

                response = pipeline.ask(
                    question
                )


                st.subheader("Answer")

                st.write(
                    response["answer"]
                )


                st.subheader(
                    "Source Pages"
                )

                pages = response.get(
                    "pages",
                    []
                )


                if pages:

                    st.write(
                        pages
                    )

                else:

                    st.write(
                        "No sources found"
                    )


            except Exception as e:

                st.error(
                    f"Unable to answer question: {e}"
                )