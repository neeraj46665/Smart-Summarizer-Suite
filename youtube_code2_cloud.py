import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate


from pytube import YouTube

import re
from langchain_core.documents import Document

import os
from dotenv import load_dotenv


# Initialize the loader with the video URL
import requests
from bs4 import BeautifulSoup
import time



def get_transcript(youtube_url):
    # Extract the video ID from the YouTube URL
    video_id = get_video_id(youtube_url)

    # Construct the transcript URL
    transcript_url = f"https://youtubetotranscript.com/transcript?v={video_id}&current_language_code=en"

    # Send a request to the webpage
    response = requests.get(transcript_url)

    # Parse the page content using BeautifulSoup
    soup = BeautifulSoup(response.content, 'html.parser')

    # Find all p tags with the specific class
    transcript = soup.find_all('p', class_="inline NA text-primary-content")

    # Store transcript in a string
    transcript_text = ""
    for p in transcript:
        transcript_text += p.text + "\n"
    # st.write(transcript_text)

    # Return the transcript text
    return transcript_text

import http.client

def get_video_title(video_id):
    conn = http.client.HTTPSConnection("www.youtube.com")
    conn.request("GET", f"/watch?v={video_id}")
    res = conn.getresponse()
    html_content = res.read().decode()

    # Use a regular expression to find the title in the HTML content
    match = re.search(r'<title>(.*?)</title>', html_content)
    if match:
        return match.group(1).replace(" - YouTube", "").strip()
    else:
        return "Title not found"

# Load environment variables
load_dotenv()

# Set up the language model with API key
os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')
llm = ChatGroq(model='llama-3.1-70b-versatile')

# List of top 10 most spoken languages with their codes
top_languages = [
    ("English", "en"),
    ("Mandarin Chinese", "zh"),
    ("Hindi", "hi"),
    ("Spanish", "es"),
    ("French", "fr"),
    ("Standard Arabic", "ar"),
    ("Bengali", "bn"),
    ("Portuguese", "pt"),
    ("Russian", "ru"),
    ("Japanese", "ja"),
]

def get_video_id(youtube_url):
    # Check for YouTube video URL and extract video ID
    video_id = None
    # Regular expression to match different YouTube URL formats
    match = re.search(r"(?:https?://)?(?:www\.)?(?:youtube\.com/watch\?v=|youtu\.be/)([a-zA-Z0-9_-]{11})", youtube_url)
    if match:
        video_id = match.group(1)
    return video_id

# Main function for YouTube summarizer
def youtube_summarizer():
    st.title("YouTube Video Summarizer")

    # Input YouTube URL
    youtube_url = st.text_input("Enter YouTube URL:")

    # Dropdown for language selection
    language = st.selectbox(
        "Select the language for summarization:",
        [lang_name for lang_name, _ in top_languages]
    )

    # Button to generate summary
    if st.button("Generate Summary"):
        if youtube_url:
            with st.spinner("Loading transcript, fetching thumbnail, and generating summary..."):
                video_id = get_video_id(youtube_url)                    
                video_title = get_video_title(video_id)
                # Display video title
                st.subheader(video_title)
                transcript = get_transcript(youtube_url)

                # Load Transcript in the selected language
                documents = [
                    Document(
                        page_content=transcript,
                        metadata={"source": youtube_url, "title": video_title}
                    )
                ]
                
                transcript = documents
                thumbnail_url = f"https://i.ytimg.com/vi/{video_id}/hq720.jpg"

                # Check if transcript is available
                if not transcript:
                    st.warning("No transcript available in the selected language.")
                    return

                # Split Transcript into chunks
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=100000, chunk_overlap=10000, add_start_index=True)
                chunks = text_splitter.split_documents(transcript)

                # Prepare prompts
                chunk_prompt = "Break down the following text into key points and highlight the most critical information for each section: Text: '{text}' Key Points:"
                map_prompt_template = PromptTemplate(input_variables=['text'], template=chunk_prompt)

                final_prompt = '''
Provide a comprehensive summary of the key points in the specified language ({language}). First, list the key points as a breakdown, then synthesize these points into a clear and concise summary. TEXT: {text}
'''
                final_prompt_template = PromptTemplate(input_variables=['text', 'language'], template=final_prompt)

                # Load summarization chain
                chain = load_summarize_chain(
                    llm=llm,
                    chain_type='map_reduce',
                    map_prompt=map_prompt_template,
                    combine_prompt=final_prompt_template,
                    verbose=False
                )

                # Display video thumbnail
                st.image(thumbnail_url, use_column_width=True)

                # # Generate and display the full summary
                summary = chain.run({"input_documents": chunks, "language": language})
                st.subheader(f"Summary (in {language}):")
                st.success(summary)

        else:
            st.warning("Please enter a valid YouTube URL.")

# Run the app
if __name__ == "__main__":
    youtube_summarizer()
