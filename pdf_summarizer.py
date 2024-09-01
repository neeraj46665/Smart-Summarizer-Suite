import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.llms import Cohere
from langchain.schema import Document
from langchain.chains.summarize import load_summarize_chain

def pdf_summarizer():
    st.title('Text & PDF Summarizer')

    # Text input or PDF upload options
    st.subheader("Choose input method:")
    input_method = st.radio("Select an option:", ("Upload PDF", "Enter Text"))

    # Initialize variables
    pdf_text = ''
    input_text = ''

    if input_method == "Upload PDF":
        # Upload PDF
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
        if uploaded_file is not None:
            try:
                # Read PDF
                pdf_reader = PdfReader(uploaded_file)
                for page in pdf_reader.pages:
                    text = page.extract_text()
                    if text:
                        pdf_text += text
            except Exception as e:
                st.error(f"An error occurred while reading the PDF: {e}")
    elif input_method == "Enter Text":
        # Text input
        input_text = st.text_area("Enter the text you want to summarize")

    # Combine text sources
    text_to_summarize = pdf_text if pdf_text else input_text

    if text_to_summarize:
        # Initialize text splitter
        text_splitter = RecursiveCharacterTextSplitter()
        chunks = text_splitter.split_text(text_to_summarize)
        
        # Convert text chunks into Document objects
        documents = [Document(page_content=chunk) for chunk in chunks]
        
        # Initialize Cohere LLM
        llm = Cohere(temperature=0.8)

        # Define the summarization chain
        summarize_chain = load_summarize_chain(llm=llm, chain_type="refine", verbose=True)

        # Create an empty string for the summary
        full_summary = ""

        # Process each document chunk with spinner
        with st.spinner('Generating summary...'):
            try:
                for doc in documents:
                    summary = summarize_chain.run([doc])
                    full_summary += summary + "\n\n"
                
                # Display the full summary
                st.subheader("Summary:")
                st.success(full_summary)
            except Exception as e:
                st.error(f"An error occurred during summarization: {e}")

    else:
        st.warning("Please upload a PDF or enter text to summarize.")
