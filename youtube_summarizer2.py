import streamlit as st
from pytube import YouTube
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import YoutubeLoader
from langchain_community.llms import Cohere
from langchain.chains.summarize import load_summarize_chain
from langchain.schema import Document
from langchain_google_genai import GoogleGenerativeAI


def youtube_summarizer():
    st.title("YouTube Video Summarizer")

    # Input YouTube URL
    youtube_url = st.text_input("Enter YouTube URL:")
    full_summary = ""

    # Button to generate summary
    if st.button("Generate Summary"):
        if youtube_url:
            with st.spinner("Loading transcript, fetching thumbnail, and generating summary..."):
                try:
                    # Get video details using PyTube
                    yt = YouTube(youtube_url)
                    video_title = yt.title
                    thumbnail_url = yt.thumbnail_url

                    # Display video title
                    st.subheader(video_title)

                    # Load Transcript
                    loader = YoutubeLoader.from_youtube_url(youtube_url, language=["hi", "en", "en-US"])
                    transcript = loader.load()

                    # Split Transcript into chunks
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=3500, chunk_overlap=200, add_start_index=True)
                    chunks = text_splitter.split_documents(transcript)
                    

                    # Ensure chunks are strings
                    string_chunks = [chunk.page_content for chunk in chunks if isinstance(chunk.page_content, str)]

                    # Set up LLM with temperature
                    # llm = Cohere(temperature=0.5)
                    llm = GoogleGenerativeAI(model='gemini-1.5-pro')

                    # Define the summarization chain
                    summarize_chain = load_summarize_chain(llm=llm, chain_type="refine", verbose=True)

                    # Process each chunk with spinner and error handling
                    for chunk in string_chunks:
                        chunk_summary = summarize_chain.run([Document(page_content=chunk)])
                        full_summary += chunk_summary + "\n\n"

                    # Display video thumbnail
                    st.image(thumbnail_url, use_column_width=True)

                    # Display the full summary
                    st.subheader("Summary:")
                    st.success(full_summary)

                except Exception as e:
                    st.error(f"An error occurred: {e}")
        else:
            st.warning("Please enter a valid YouTube URL.")

if __name__ == "__main__":
    youtube_summarizer()
