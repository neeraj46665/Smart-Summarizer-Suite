import os
import streamlit as st
from pdf_summarizer import pdf_summarizer
from news_summarizer import news_summarizer
# from youtube_summarizer import youtube_summarizer
from t import youtube_summarizer
from dotenv import load_dotenv
load_dotenv()



os.environ["COHERE_API_KEY"]= os.getenv('COHERE_API_KEY')
os.environ['GROQ_API_KEY']=os.getenv('GROQ_API_KEY')





def main():
    # Sidebar with unique key
    selected_option = st.sidebar.radio(
        "Choose a summarizer:",
        ("📄 PDF/Text Summarizer", "📰 News Summarizer", "🎥 YouTube Summarizer"),
        key='summarizer_selector'
    )

    # Get the current theme color
    background_color = "#e6f2ff" 

    # Apply styling to sidebar
    st.sidebar.markdown(
        f"""
        <style>
        .sidebar .sidebar-content {{
            background-color: {background_color};
            padding: 10px;
            border-radius: 10px;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

    # Main title with a custom style
    st.markdown(
        f"""
        <h1 style='text-align: center; color: #4B89DC; font-weight: bold;'>📝 Summarizer App</h1>
        """,
        unsafe_allow_html=True
    )

    # Display welcome message and app description
    if selected_option == "📄 PDF/Text Summarizer":
        st.markdown(
            f"""
            <div style='background-color: {background_color}; padding: 15px; border-radius: 10px;color:black'>
            <h2 style='color:black' >Welcome to the Summarizer App!</h2>
            <p >This tool helps you quickly summarize PDFs, news articles, and YouTube videos using advanced AI models.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Sidebar with options and additional information
    st.sidebar.title("Navigation")

    # Sidebar section for About
    st.sidebar.header("About")
    st.sidebar.markdown(
        f"""
        <div style='background-color: {background_color}; padding: 15px; border-radius: 10px;color:black''>
        <p >This app was made by <strong>Neeraj Singh</strong> and <strong>Faisal Hussain</strong>.</p>
        <p >It uses state-of-the-art AI models from <strong>Cohere</strong> and <strong>Google</strong> to provide concise summaries of text, news articles, and video content.</p>
        <p >Simply select the content type from the options above, upload or paste your content, and get a summary in seconds!</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Set background color variable
    background_color = "#e6f2ff"  # Light blue color for background

    # Sidebar section for User Guide with background styling
    st.sidebar.header("User Guide")
    st.sidebar.markdown(
        f"""
        <div style='background-color: {background_color}; padding: 15px; border-radius: 10px; color:black;'>
            <p><strong>How to Use:</strong></p>
            <p>1. <strong>Select Input Method:</strong> Choose between uploading a PDF or entering text.</p>
            <p>2. <strong>Generate Summary:</strong> Click on 'Generate Summary' to process the input.</p>
            <p>3. <strong>View Output:</strong> The summarized text will appear below the input section.</p>
        </div>
        """,
        unsafe_allow_html=True
    )




    # Display the appropriate summarizer based on the selected option
    if selected_option == "📄 PDF/Text Summarizer":
        st.subheader("📄 PDF/Text Summarizer")
        st.markdown("Upload a PDF file or paste your text, and get a summarized version of the content.")
        pdf_summarizer()
    elif selected_option == "📰 News Summarizer":
        st.subheader("📰 News Summarizer")
        st.markdown(
            f"""
            <div style='background-color: {background_color}; padding: 15px; border-radius: 10px; color:black''>
            Fetch the latest news articles and get a summarized overview.
            </div>
            """,
            unsafe_allow_html=True
        )
        news_summarizer()
    elif selected_option == "🎥 YouTube Summarizer":
        st.subheader("🎥 YouTube Summarizer")
        st.markdown(
            f"""
            <div style='background-color: {background_color}; padding: 15px; border-radius: 10px;color:black''>
            Paste a YouTube video URL and get a summarized version of the content.
            </div>
            """,
            unsafe_allow_html=True
        )
        youtube_summarizer()

    # Footer with contact information and credits
    st.markdown(
        """
        <hr style="border-top: 1px solid #e6e6e6;">
        <div style='text-align: center;'>
            <p >Made with ❤️ by Neeraj Singh and Faisal Hussain</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
