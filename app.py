import streamlit as st
import requests
import pandas as pd
import time
import nltk
from transformers import pipeline
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import seaborn as sns

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize the sentiment analysis pipeline
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
revision = "af0f99b"

@st.cache
def load_pipeline(model_name, revision):
    return pipeline("sentiment-analysis", model=model_name, revision=revision)

sentiment_pipeline = load_pipeline(model_name, revision)

# Preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

# Function to fetch YouTube comments
def fetch_comments(video_id, api_key, page_limit=5):
    comments = []
    page_token = None
    page_count = 0

    while page_count < page_limit:
        url = f'https://www.googleapis.com/youtube/v3/commentThreads?key={api_key}&textFormat=plainText&part=snippet&videoId={video_id}&maxResults=100'
        if page_token:
            url += f'&pageToken={page_token}'
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()

            for item in data.get('items', []):
                top_comment = item['snippet']['topLevelComment']['snippet']
                comments.append({
                    'Author': top_comment.get('authorDisplayName', 'N/A'),
                    'Comment': top_comment.get('textDisplay', 'N/A'),
                    'Likes': top_comment.get('likeCount', 0),
                    'Published At': top_comment.get('publishedAt', 'N/A')
                })

                if 'replies' in item:
                    for reply in item['replies']['comments']:
                        reply_comment = reply['snippet']
                        comments.append({
                            'Author': reply_comment.get('authorDisplayName', 'N/A'),
                            'Comment': reply_comment.get('textDisplay', 'N/A'),
                            'Likes': reply_comment.get('likeCount', 0),
                            'Published At': reply_comment.get('publishedAt', 'N/A')
                        })

            next_page_token = data.get('nextPageToken')
            if not next_page_token:
                break

            page_token = next_page_token
            page_count += 1
            time.sleep(1)  # To avoid hitting API rate limits

        except requests.exceptions.RequestException as e:
            st.error(f"An error occurred: {e}")
            break

    return comments

# Streamlit interface
st.title('YouTube Comment Sentiment Analysis')

api_key = st.text_input("Enter your YouTube API Key:", type="password")
video = st.text_input("Enter YouTube Video Link:")
if 'v=' in video:   
    video_id = video.split("v=")[1]

if st.button("Fetch and Analyze Comments"):
    if not api_key or not video_id:
        st.error("Please provide both YouTube API Key and Video ID.")
    else:
        st.info("Fetching comments...")
        comments = fetch_comments(video_id, api_key)
        
        if comments:
            st.success(f"Fetched {len(comments)} comments.")
            st.info("Preprocessing comments...")

            comment_texts = [comment['Comment'] for comment in comments]
            preprocessed_comments = [preprocess_text(text) for text in comment_texts]

            st.info("Performing sentiment analysis...")

            results = []
            sentiments = []
            for comment, preprocessed_comment in zip(comments, preprocessed_comments):
                sentiment = sentiment_pipeline(preprocessed_comment)
                sentiments.append(sentiment[0]['label'])
                results.append({
                    'Author': comment['Author'],
                    'Comment': comment['Comment'],
                    'Preprocessed Comment': preprocessed_comment,
                    'Sentiment': sentiment[0]['label']
                })

            df = pd.DataFrame(results)
            st.write(df)

            # Generate word cloud for positive and negative comments
            st.info("Generating word cloud...")
            positive_comments = ' '.join(df[df['Sentiment'] == 'POSITIVE']['Comment'])
            negative_comments = ' '.join(df[df['Sentiment'] == 'NEGATIVE']['Comment'])
            combined_wordcloud = WordCloud(width=800, height=400, background_color=None, mode='RGBA',
                                           colormap='coolwarm', stopwords=stopwords).generate(positive_comments + negative_comments)
            plt.figure(figsize=(10, 6))
            plt.imshow(combined_wordcloud, interpolation='bilinear')
            plt.title('Word Cloud for Positive and Negative Words', fontsize=14)
            plt.axis('off')
            st.pyplot(plt)

            # Save the word cloud image
            plt.savefig('wordcloud_image.png', bbox_inches='tight', pad_inches=0.1)

            # Emotion analysis using the EmoRoBERTa model
            st.info("Performing emotion analysis...")
            emotion_pipeline = pipeline('sentiment-analysis', model='arpanghoshal/EmoRoBERTa')

            def get_emotion_label(text):
                return emotion_pipeline(text)[0]['label']

            df['Emotion'] = df['Comment'].apply(get_emotion_label)
            df_filtered = df[df['Emotion'] != 'neutral']

            # Display emotion distribution
            st.info("Visualizing emotion distribution...")
            plt.figure(figsize=(10, 6))
            sns.countplot(data=df_filtered, y='Emotion').set_title("Emotion Distribution (Excluding Neutral)")
            st.pyplot(plt)

            # Optionally save the results to a CSV file
            df.to_csv('youtube_comments_with_sentiment_and_emotion.csv', index=False)
        else:
            st.warning("No comments fetched or an error occurred.")
