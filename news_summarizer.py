import streamlit as st
import requests
from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.llms import Cohere
from langchain.schema import Document
from langchain.chains.summarize import load_summarize_chain



def scrape_news_from_class(homepage_url):
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        # Fetch the homepage content with User-Agent header
        response = requests.get(homepage_url, headers=headers)
        response.raise_for_status()
        
        # Parse the HTML content
        soup = BeautifulSoup(response.text, 'html.parser')
        
        news_list = []
        
        # First news with different class
        first_article = soup.find('div', class_='cartHolder bigCart track timeAgo articleClick')
        if first_article:
            news_list.append(parse_article(first_article))
        
        # Rest of the news with common class
        articles = soup.find_all('div', class_='cartHolder listView track timeAgo articleClick')
        for article in articles:
            news_list.append(parse_article(article))
        
        return news_list
    
    except requests.RequestException as e:
        st.write(f"Error fetching or processing the page: {e}")
        return []

def parse_article(article):
    # Extract article title
    title = article.get('data-vars-story-title', 'No title found')
    
    # Extract article URL
    article_url = article.get('data-weburl', '#')
    
    # Extract image URL from the figure tag
    img_tag = article.find('figure').find('img') if article.find('figure') else None
    if img_tag:
        img_url = img_tag.get('data-src', img_tag.get('src', 'No image found'))
        img_url = img_url.replace('148x111', '550x309')  # Replace the size
    else:
        img_url = 'No image found'
    
    # Extract publish time
    publish_time = article.get('data-vars-story-time', 'No time found')
    
    # Extract section
    section = article.get('data-vars-section', 'No section found')
    
    return {
        'title': title,
        'url': article_url,
        'image_url': img_url,
        'publish_time': publish_time,
        'section': section
    }

def scrape_content(url):
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Referer': 'https://www.hindustantimes.com/'
        }
        
        response = requests.get(url, headers=headers)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = soup.find_all('p')
        content = ' '.join(paragraph.get_text(strip=True) for paragraph in paragraphs)

        return content
    
    except requests.RequestException as e:
        st.error(f"Error fetching or processing the page: {e}")
        return None

def summarize_text(text):
    text_splitter = RecursiveCharacterTextSplitter()
    chunks = text_splitter.split_text(text)
    documents = [Document(page_content=chunk) for chunk in chunks]

    llm = Cohere(temperature=0.8)
    summarize_chain = load_summarize_chain(llm=llm, chain_type="refine", verbose=True)
    summary = summarize_chain.run(documents)
    return summary

def news_summarizer():
    st.title('Hindustan Times News Summarizer')

    homepage_url = 'https://www.hindustantimes.com'
    
    with st.spinner('Fetching news articles...'):
        news_details = scrape_news_from_class(homepage_url)

    if news_details:
        for i, news in enumerate(news_details):
            st.subheader(news['title'])
        
            if news['image_url'] and news['image_url'] != 'No image found':
                st.image(news['image_url'], use_column_width=True)
            else:
                st.write("No image available")

            st.info(
    f"[Read Full Article]({news['url']}),    "
    
    f"{news['publish_time']},    "
    
    f"{news['section']}"
)


            
            
            

            if st.button(f"Summarize Article {i+1}", key=f"summarize_{i}"):
                with st.spinner('Fetching article content...'):
                    content = scrape_content(news['url'])
                
                if content:
                    c = 'Summarize the following content into bullet points and ask no questions, it should be only summary.'
                    c += content
                    with st.spinner('Generating summary...'):
                        summary = summarize_text(c)
                        st.markdown("Summary:")
                        st.success(summary)
                else:
                    st.write("Failed to retrieve content.")
            
            st.write("---")
    else:
        st.write("No news articles found.")
