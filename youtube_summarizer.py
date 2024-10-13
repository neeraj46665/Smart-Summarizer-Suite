import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
# from langchain.document_loaders import YoutubeLoader
from langchain_community.document_loaders import YoutubeLoader

from pytube import YouTube
from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound
import re
from langchain_core.documents import Document

import os
from dotenv import load_dotenv


from langchain_community.document_loaders import YoutubeLoader



# Initialize the loader with the video URL










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

def get_youtube_video_details(youtube_url):
    try:
        video_id = get_video_id(youtube_url)
        if not video_id:
            return None, None, "Invalid YouTube URL."

        # Fetch video details using pytube
        yt = YouTube(youtube_url)
        title = yt.title
        thumbnail_url = yt.thumbnail_url
        
        # Fetch transcript using youtube-transcript-api
        transcript_text = ""
        languages = ['en', 'hi']  # Hardcoded languages: English and Hindi
        
        for lang in languages:
            try:
                transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=[lang])
                # Join all text parts together without timestamps
                transcript_text += f"Transcript in {lang.upper()}:\n" + "\n".join([t['text'] for t in transcript]) + "\n\n"
            except NoTranscriptFound:
                transcript_text += f"No transcript available for this video in '{lang}' language.\n\n"
            except Exception as e:
                transcript_text += str(e) + "\n\n"

        return title, thumbnail_url, transcript_text.strip()
    except Exception as e:
        return None, None, str(e)


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
                
                    

                    # loader = YoutubeLoader('O0dRSA8b5tk',language=["en", "hi"])
                    loader = YoutubeLoader.from_youtube_url(youtube_url,language=["en", "hi"], add_video_info=True)
                    # Load the transcript
                    transcript = loader.load()
                    
                    video_title=transcript[0].metadata['title']
                    thumbnail_url=transcript[0].metadata['thumbnail_url']
                    
                    # video_title, thumbnail_url, transcript = get_youtube_video_details(youtube_url)
                    # Display video title
                    st.subheader(video_title)

                    # Load Transcript in the selected language
                    
                    # documents = [
                    #                 Document(
                    #                     page_content=transcript,
                    #                     metadata={"source": youtube_url, "title": video_title}
                    #                 )
                                    
                    #             ]
                    # transcript = documents
               

                    # Check if transcript is available
                    if not transcript:
                        st.warning("No transcript available in the selected language.")
                        return

#                     # Split Transcript into chunks
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

                    # Generate and display the full summary
                    summary = chain.run({"input_documents": chunks, "language": language})
                 
                    st.subheader(f"Summary (in {language}):")
                    st.success(summary)

               
        else:
            st.warning("Please enter a valid YouTube URL.")


# Run the app
if __name__ == "__main__":
    youtube_summarizer()
