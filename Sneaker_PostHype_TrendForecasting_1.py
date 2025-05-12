#!/usr/bin/env python
# coding: utf-8

# # Sneaker & Streetwear Trend Evolution: Post-Hype Culture
# This project explores how sneaker and streetwear trends have evolved in the post-hype era, especially among Gen Z consumers. We use data from Google Trends, Reddit, and fashion blogs to uncover shifting interest from traditional hype silhouettes (like Yeezys and Jordans) to emerging aesthetics such as gorpcore, quiet luxury, and trailwear.

# In[1]:


get_ipython().system('pip install pytrends praw beautifulsoup4 matplotlib pandas scikit-learn')


# ## Step 1: Google Trends – Interest Over Time

# In[4]:


import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV (skip first row to get proper headers)
df = pd.read_csv("/Users/karishmamehta/Downloads/sneaker_google_trend.csv", skiprows=1)

# Rename columns for cleaner names
df.columns = ['Week', 'Yeezy', 'Air Jordan', 'Asics', 'Adidas Samba', 'New Balance']

# Convert '<1' values to 0 (you can also use 0.5 if you want to keep a low signal)
df.replace('<1', 0, inplace=True)

# Convert all columns (except 'Week') to numeric
for col in df.columns[1:]:
    df[col] = pd.to_numeric(df[col])

# Convert 'Week' column to datetime
df['Week'] = pd.to_datetime(df['Week'])

# Set the date column as index
df.set_index('Week', inplace=True)

# Plot the trends
plt.figure(figsize=(14, 7))
for col in df.columns:
    plt.plot(df.index, df[col], label=col)

plt.title("Sneaker Brand Interest Over Time (US - Google Trends)")
plt.xlabel("Date")
plt.ylabel("Search Interest")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()



# In[6]:


df_smooth = df.rolling(window=4).mean()
df_smooth.plot(figsize=(14, 7), title="4-Week Moving Average of Sneaker Trends")
plt.ylabel("Smoothed Interest")
plt.grid(True)
plt.show()


# ## Step 2: Reddit Discussion Scraping

# In[16]:


import praw
import pandas as pd
from datetime import datetime
import time


# Set up Reddit connection
reddit = praw.Reddit(
    client_id='j2s7NZ4RAKIEsfsUK0fjsw',
    client_secret='r6S8SrqjKkYsyiXG2fEpTs05GxsAxg',
    user_agent='fashion_g_trends_app',
    username='fashion_trends_17',
    password='Mahi05Kari17@'  # replace this
)

# Target sneaker subreddits
target_subreddits = [
    'Sneakers', 'FashionReps', 'FashionRepsBST',
    'SneakerMarket', 'SneakerBots', 'Repsneakers'
]

# Search terms (sneaker brands/trends)
search_terms = ['yeezy', 'air jordan', 'asics', 'new balance', 'adidas samba']

# Collect posts
all_posts = []

for subreddit in target_subreddits:
    for term in search_terms:
        print(f"🔎 Searching '{term}' in r/{subreddit}")
        try:
            for post in reddit.subreddit(subreddit).search(term, sort='new', limit=500):
                if not post.selftext: post.selftext = ""
                if len(post.title + post.selftext) < 30:
                    continue  # skip short/low-effort posts
                all_posts.append([
                    term,
                    post.title,
                    post.selftext,
                    post.score,
                    post.num_comments,
                    datetime.fromtimestamp(post.created_utc),
                    subreddit
                ])
            time.sleep(2)  # small pause between batches
        except Exception as e:
            print(f"❌ Error in {subreddit} - {term}: {e}")
            continue

# Save as DataFrame
df = pd.DataFrame(all_posts, columns=[
    'Keyword', 'Title', 'Body', 'Upvotes', 'Comments_Count', 'Date', 'Subreddit'
])

# Save to CSV
df.to_csv('reddit_filtered_sneaker_posts.csv', index=False)
print("Saved: reddit_filtered_sneaker_posts.csv")


# In[11]:


get_ipython().system('pip install bertopic')
get_ipython().system('pip install umap-learn')
get_ipython().system('pip install sentence-transformers')


# # STEP 3: BERT ANALYSIS

# ### BERT for Yeezy

# In[46]:


import pandas as pd
import re
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Load dataset
df = pd.read_csv("/Users/karishmamehta/Downloads/reddit_filtered_sneaker_posts.csv")

# Combine title and body
df['Text'] = df[['Title', 'Body']].fillna('').agg(' '.join, axis=1)

# Filter for Yeezy posts
yeezy_df = df[df['Keyword'] == 'yeezy'].copy()

# Clean text function
def clean_text(text):
    text = str(text).lower()
    text = text.encode('ascii', 'ignore').decode('utf-8')               # remove unicode
    text = re.sub(r'http\S+', '', text)                                 # remove URLs
    text = re.sub(r'\b(sz|tts|x200b|yezzy|yeezy)\b', '', text)  # normalize & remove yeezy variants
    text = re.sub(r'[^a-z\s]', '', text)                                # remove punctuation/numbers
    text = re.sub(r'\b\w{1,2}\b', '', text)                             # remove short words
    text = re.sub(r'\s+', ' ', text)                                    # normalize whitespace
    return text.strip()

# Apply cleaning
yeezy_df['CleanText'] = yeezy_df['Text'].apply(clean_text)

# Filter short entries
texts = yeezy_df['CleanText']
texts = [t for t in texts if len(t) > 30]

# Define custom stopwords
brand_stopwords = {'yeezy', 'yezzy', 'rnnr', 'slide', 'slides'}
custom_stopwords = list(ENGLISH_STOP_WORDS.union(brand_stopwords))

# TF-IDF Vectorizer
vectorizer_model = CountVectorizer(ngram_range=(1, 2), stop_words=custom_stopwords)

# Run BERTopic
topic_model = BERTopic(vectorizer_model=vectorizer_model, verbose=True)
topics, probs = topic_model.fit_transform(texts)

# Show top 10 topics
print(topic_model.get_topic_info().head(10))

# Visualize with custom title
fig = topic_model.visualize_barchart(top_n_topics=10)
fig.update_layout(
    title_text="Top Topics in Yeezy Reddit Posts )",
    title_font_size=18,
    title_x=0.5
)
fig.show()


# 🧠 Insights You Can Extract
# Product-Specific Trends: Yeezy Slides and 350 dominate discussion
# 
# Buyer Behavior: Lots of posts about WTS (want to sell), QC checks, and fit
# 
# Resale Ecosystem: Topics on bots and proxies highlight automation tools usage
# 
# Community Culture: "Feet today" shows lifestyle or flex-posts are common
# 
# 

# ### BERT for adidas samba

# In[42]:


import pandas as pd
import re
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Load dataset
df = pd.read_csv("/Users/karishmamehta/Downloads/reddit_filtered_sneaker_posts.csv")

# Combine title and body
df['Text'] = df[['Title', 'Body']].fillna('').agg(' '.join, axis=1)

# Filter for Adidas Samba posts
samba_df = df[df['Keyword'] == 'adidas samba'].copy()

# Clean text function
def clean_text(text):
    text = str(text).lower()
    text = text.encode('ascii', 'ignore').decode('utf-8')               # remove unicode
    text = re.sub(r'http\S+', '', text)                                 # remove URLs
    text = re.sub(r'\b(sz|tts|x200b)\b', 'size', text)                  # normalize
    text = re.sub(r'[^a-z\s]', '', text)                                # remove punctuation/numbers
    text = re.sub(r'\b\w{1,2}\b', '', text)                             # remove short words
    text = re.sub(r'\s+', ' ', text)                                    # normalize whitespace
    return text.strip()

# Apply cleaning
samba_df['CleanText'] = samba_df['Text'].apply(clean_text)

# Filter short entries
texts = samba_df['CleanText']
texts = [t for t in texts if len(t) > 30]

# Define custom stopwords
brand_stopwords = {'adidas', 'samba', 'sambas', 'adidassamba'}
custom_stopwords = list(ENGLISH_STOP_WORDS.union(brand_stopwords))

# TF-IDF Vectorizer with brand stopwords
vectorizer_model = CountVectorizer(ngram_range=(1, 2), stop_words=custom_stopwords)

# Run BERTopic
topic_model = BERTopic(vectorizer_model=vectorizer_model, verbose=True)
topics, probs = topic_model.fit_transform(texts)

# Show top 10 topics
print(topic_model.get_topic_info().head(10))

# Visualize with custom title
fig = topic_model.visualize_barchart(top_n_topics=10)
fig.update_layout(
    title_text="Top Topics in Cleaned Adidas Samba Reddit Posts (Excluding Brand Terms)",
    title_font_size=18,
    title_x=0.5
)
fig.show()


# 🧠 Insights
# There's a strong emphasis on fit reviews and variant (colorway) feedback for Sambas.
# 
# Discussions around quality and haul show they are often included in bulk or influencer-style posts.
# 
# Resale isn’t as dominant as in Yeezy, but it still shows up.

# ### BERT for air jordan

# In[44]:


import pandas as pd
import re
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Load dataset
df = pd.read_csv("/Users/karishmamehta/Downloads/reddit_filtered_sneaker_posts.csv")

# Combine title and body
df['Text'] = df[['Title', 'Body']].fillna('').agg(' '.join, axis=1)

# Filter for Air Jordan posts
jordan_df = df[df['Keyword'] == 'air jordan'].copy()

# Clean text function
def clean_text(text):
    text = str(text).lower()
    text = text.encode('ascii', 'ignore').decode('utf-8')               # remove unicode
    text = re.sub(r'http\S+', '', text)                                 # remove URLs
    text = re.sub(r'\b(sz|tts|x200b)\b', 'size', text)                  # normalize size variants
    text = re.sub(r'[^a-z\s]', '', text)                                # remove punctuation/numbers
    text = re.sub(r'\b\w{1,2}\b', '', text)                             # remove short words
    text = re.sub(r'\s+', ' ', text)                                    # normalize whitespace
    return text.strip()

# Apply cleaning
jordan_df['CleanText'] = jordan_df['Text'].apply(clean_text)

# Filter short entries
texts = jordan_df['CleanText']
texts = [t for t in texts if len(t) > 30]

# Define custom stopwords to exclude brand-related terms
brand_stopwords = {'air', 'jordan', 'aj', 'airjordan', 'jordans'}
custom_stopwords = list(ENGLISH_STOP_WORDS.union(brand_stopwords))

# TF-IDF Vectorizer with custom stopwords
vectorizer_model = CountVectorizer(ngram_range=(1, 2), stop_words=custom_stopwords)

# Run BERTopic
topic_model = BERTopic(vectorizer_model=vectorizer_model, verbose=True)
topics, probs = topic_model.fit_transform(texts)

# Show top 10 topics
print(topic_model.get_topic_info().head(10))

# Visualize with custom title
fig = topic_model.visualize_barchart(top_n_topics=10)
fig.update_layout(
    title_text="Top Topics in Air Jordan Reddit Posts",
    title_font_size=18,
    title_x=0.5
)
fig.show()


# ### BERT for Asics

# In[48]:


import pandas as pd
import re
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Load dataset
df = pd.read_csv("/Users/karishmamehta/Downloads/reddit_filtered_sneaker_posts.csv")

# Combine title and body
df['Text'] = df[['Title', 'Body']].fillna('').agg(' '.join, axis=1)

# Filter for Asics posts
asics_df = df[df['Keyword'] == 'asics'].copy()

# Clean text function
def clean_text(text):
    text = str(text).lower()
    text = text.encode('ascii', 'ignore').decode('utf-8')                  # remove unicode
    text = re.sub(r'http\S+', '', text)                                    # remove URLs
    text = re.sub(r'\b(sz|tts|x200b|asics)\b', '', text)  # normalize/rem brand variants
    text = re.sub(r'[^a-z\s]', '', text)                                   # remove punctuation/numbers
    text = re.sub(r'\b\w{1,2}\b', '', text)                                # remove short words
    text = re.sub(r'\s+', ' ', text)                                       # normalize whitespace
    return text.strip()

# Apply cleaning
asics_df['CleanText'] = asics_df['Text'].apply(clean_text)

# Filter short entries
texts = asics_df['CleanText']
texts = [t for t in texts if len(t) > 30]

# Define custom stopwords
brand_stopwords = {'asics', 'gel'}
custom_stopwords = list(ENGLISH_STOP_WORDS.union(brand_stopwords))

# TF-IDF Vectorizer
vectorizer_model = CountVectorizer(ngram_range=(1, 2), stop_words=custom_stopwords)

# Run BERTopic
topic_model = BERTopic(vectorizer_model=vectorizer_model, verbose=True)
topics, probs = topic_model.fit_transform(texts)

# Show top 10 topics
print(topic_model.get_topic_info().head(10))

# Visualize with custom title
fig = topic_model.visualize_barchart(top_n_topics=10)
fig.update_layout(
    title_text="Top Topics in Asics Reddit Posts",
    title_font_size=18,
    title_x=0.5
)
fig.show()


# ### BERT for new balance

# In[51]:


import pandas as pd
import re
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Load dataset
df = pd.read_csv("/Users/karishmamehta/Downloads/reddit_filtered_sneaker_posts.csv")

# Combine title and body
df['Text'] = df[['Title', 'Body']].fillna('').agg(' '.join, axis=1)

# Filter for New Balance posts
nb_df = df[df['Keyword'] == 'new balance'].copy()

# Clean text function
def clean_text(text):
    text = str(text).lower()
    text = text.encode('ascii', 'ignore').decode('utf-8')                        # remove unicode
    text = re.sub(r'http\S+', '', text)                                          # remove URLs
    text = re.sub(r'\b(sz|tts|x200b|newbalance|new|balance|nb)\b', '', text)  # remove brand terms
    text = re.sub(r'[^a-z\s]', '', text)                                         # remove punctuation/numbers
    text = re.sub(r'\b\w{1,2}\b', '', text)                                      # remove short words
    text = re.sub(r'\s+', ' ', text)                                             # normalize whitespace
    return text.strip()

# Apply cleaning
nb_df['CleanText'] = nb_df['Text'].apply(clean_text)

# Filter short entries
texts = nb_df['CleanText']
texts = [t for t in texts if len(t) > 30]

# Define custom stopwords
brand_stopwords = {'new', 'balance', 'nb', 'newbalance'}
custom_stopwords = list(ENGLISH_STOP_WORDS.union(brand_stopwords))

# TF-IDF Vectorizer
vectorizer_model = CountVectorizer(ngram_range=(1, 2), stop_words=custom_stopwords)

# Run BERTopic
topic_model = BERTopic(vectorizer_model=vectorizer_model, verbose=True)
topics, probs = topic_model.fit_transform(texts)

# Show top 10 topics
print(topic_model.get_topic_info().head(10))

# Visualize with custom title
fig = topic_model.visualize_barchart(top_n_topics=10)
fig.update_layout(
    title_text="Top Topics in New Balance Reddit Posts",
    title_font_size=18,
    title_x=0.5
)
fig.show()


# # ANALYSIS
# 
# 🟢 1. Adidas Samba – Key Insights
# Image: newplot (1).png
# 
# Popular Topics: "WTS" (Want To Sell), sizing and fit, resale shipping, and Wales Bonner collab.
# 
# Reviews: Heavy focus on colorways, haul quality, and fit feedback.
# 
# Insight: Adidas Samba conversations are heavily influenced by fashion collaborations, with Wales Bonner being a dominant sub-theme. Users also focus on resale and fit experiences.
# 
# 🟠 2. Yeezy – Key Insights
# Image: newplot.png
# 
# Popular Topics: Bot protection (AIO, proxies), WTS threads, resale tags, payment types (PayPal invoice), and foam runners.
# 
# Suspicious Activity: Some topics are dominated by WhatsApp codes, resale terms, and seller names like “lucasyeezy” – indicating potential scam awareness.
# 
# Insight: Yeezy discussions are transaction-heavy, focused on bots, buying/selling, and drop logistics. There's also discussion around model types like Foam Runners and Slides.
# 
# 🔵 3. Air Jordan – Key Insights
# Image: newplot (2).png
# 
# Popular Topics: Retro models, WTS/WTB (want to buy), “Travis Scott” collabs, pricing, and quality.
# 
# Emerging Subtopics: “Brick” (sneaker resale slang), batch reviews, and sizing debates.
# 
# Insight: Jordans attract hype-focused and resale-driven chatter with Travis Scott being a major influencer in the conversation.
# 
# 🟣 4. Asics – Key Insights
# Image: newplot (3).png
# 
# Popular Topics: Gel-Kayano, JJJJound collabs, Clifton models, worn/sold pairs.
# 
# Tone: More technical and performance-focused compared to others. Topics around comfort, heel/toe fit, and running use cases.
# 
# Insight: Asics posts are less resale-focused and more oriented toward utility and comfort, indicating a practical user base.
# 
# 🔴 5. New Balance – Key Insights
# Image: newplot (4).png
# 
# Popular Topics: Protection Pack, WTS posts, resale conditions, walking comfort, and collaborations (e.g., JJJJound, Loro Piana).
# 
# Community Focus: Sizing is still a big concern, alongside comfort and resale packaging (“haul”).
# 
# Insight: New Balance is a blend of comfort-seekers and fashion enthusiasts, with attention toward limited releases and refined aesthetics.
# 
# 
# | **Theme**                 | **Observed In**                       | **Commentary**                                                                 |
# | ------------------------- | ------------------------------------- | ------------------------------------------------------------------------------ |
# | 🔁 Resale (WTS/WTB)       | All brands, heavily in Yeezy & Jordan | A significant portion of Reddit activity revolves around **resale logistics**. |
# | 👟 Sizing & Fit           | All brands                            | The single most consistent user concern is **getting the right size**.         |
# | 🤝 Collabs & Hype Culture | Jordan, New Balance, Samba            | **Collaborations** like Travis Scott, JJJJound, Wales Bonner spark attention.  |
# | 🤖 Bot Usage              | Yeezy                                 | **Automation bots and proxies** dominate Yeezy discourse.                      |
# | 📦 Shipping + QC          | Samba, Jordan, Yeezy                  | Concerns around **product condition** and shipping are frequent.               |
# | 🏃‍♂️ Comfort + Utility   | Asics, New Balance                    | Brands like Asics and NB attract users interested in **functionality**.        |
# 

# # STEP 4: SENTIMENT ANALYSIS

# In[61]:


pip install transformers


# In[69]:


get_ipython().system('pip install torch')
get_ipython().system('pip install tqdm')


# In[91]:


import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax

# Load dataset
df = pd.read_csv("/Users/karishmamehta/Downloads/reddit_filtered_sneaker_posts.csv")
df['Text'] = df[['Title', 'Body']].fillna('').agg(' '.join, axis=1)

# Load CardiffNLP model and tokenizer
model_name = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
model.eval()

# Initialize
brands = df['Keyword'].unique()
summary_dict = {}
label_map = {0: 'NEGATIVE', 1: 'NEUTRAL', 2: 'POSITIVE'}

# Sentiment analysis per brand
for brand in brands:
    brand_df = df[df['Keyword'] == brand].copy()
    texts = brand_df['Text'].tolist()[:300]

    sentiments = []
    for i in tqdm(range(0, len(texts), 50), desc=f"Processing {brand}"):
        batch = texts[i:i+50]
        encoded = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**encoded)
        scores = softmax(outputs.logits.numpy(), axis=1)
        batch_labels = scores.argmax(axis=1)
        batch_sentiments = [label_map[label] for label in batch_labels]
        sentiments.extend(batch_sentiments)

    # Save results
    brand_df = brand_df.iloc[:300].copy()
    brand_df['Sentiment'] = sentiments
    summary = brand_df['Sentiment'].value_counts(normalize=True) * 100
    summary_dict[brand] = summary

# Compile summary
summary_df = pd.DataFrame(summary_dict).T.fillna(0)
summary_df = summary_df[['POSITIVE', 'NEUTRAL', 'NEGATIVE']]  # order columns

print(summary_df)

summary_df.to_csv("reddit_sentiment_results.csv")


# | Brand            | 👍 Positive | 😐 Neutral | 👎 Negative | Key Insight                                                            |
# | ---------------- | ----------- | ---------- | ----------- | ---------------------------------------------------------------------- |
# | **Yeezy**        | 36.7%       | 53.3%      | 10.0%       | Slightly more positive than average; lots of **neutral resale posts**. |
# | **Air Jordan**   | 36.0%       | 57.7%      | 6.3%        | Highest neutral %; tone is mostly transactional (WTS, sizing).         |
# | **Asics**        | 33.7%       | 54.7%      | 11.7%       | Strong practical focus — comfort, fit, performance dominate.           |
# | **New Balance**  | 38.7%       | 52.0%      | 9.3%        | Most positive of all; community tone is balanced and product-focused.  |
# | **Adidas Samba** | 27.0%       | 63.3%      | 9.7%        | Lowest positivity; many neutral WTS posts + reviews with mixed tone.   |
# 
# Neutral dominates — most Reddit sneaker posts are factual, resale-oriented, or just sharing info.
# 
# New Balance has the best sentiment — likely due to strong positive perception in fashion & comfort.
# 
# Samba’s low positivity might be due to oversaturation or mixed reviews in recent fashion hype.
# 
# Negative sentiment is low across the board — very few posts are emotionally negative, which makes sense in sneaker culture.
# 

# In[82]:


import pandas as pd
import matplotlib.pyplot as plt

color_map = {
    'POSITIVE': '#2ecc71',  # Emerald Green
    'NEUTRAL': '#f1c40f',   # Sunflower Yellow
    'NEGATIVE': '#e74c3c'   # Alizarin Red
}

# Generate pie charts per brand
for brand in summary_df.index:
    sentiment_values = summary_df.loc[brand]
    labels = sentiment_values.index
    values = sentiment_values.values
    colors = [color_map[label] for label in labels]

    plt.figure(figsize=(5, 5))
    plt.pie(
        values,
        labels=labels,
        autopct='%1.1f%%',
        startangle=140,
        colors=colors,
        textprops={'fontsize': 12}
    )
    plt.title(f"Sentiment Distribution for {brand.title()}", fontsize=14)
    plt.axis('equal')
    plt.tight_layout()
    plt.show()


# # TREND STRENGTH SCORECARD

# In[125]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# ----------------- Load and Clean All Data ------------------

def clean_brand(name):
    return name.lower().replace('(united states)', '').replace(':', '').strip()

# --- 1. GOOGLE TRENDS ---
google_df = pd.read_csv("/Users/karishmamehta/Downloads/sneaker_google_trend.csv", skiprows=1)
google_df.columns = [col.strip() for col in google_df.columns]
google_df.rename(columns={google_df.columns[0]: "Week"}, inplace=True)
google_df["Week"] = pd.to_datetime(google_df["Week"])
for col in google_df.columns[1:]:
    google_df[col] = pd.to_numeric(google_df[col].replace('<1', 0))
trend_momentum = google_df.tail(6).iloc[:, 1:].mean().to_frame(name="Google_Momentum")
trend_momentum.index = trend_momentum.index.map(clean_brand)

# --- 2. REDDIT POST VOLUME GROWTH ---
reddit_df = pd.read_csv("/Users/karishmamehta/Downloads/reddit_filtered_sneaker_posts.csv")
reddit_df['Date'] = pd.to_datetime(reddit_df['Date'])
cutoff = reddit_df['Date'].max()
recent = reddit_df[reddit_df['Date'] >= cutoff - pd.Timedelta(days=30)]
previous = reddit_df[(reddit_df['Date'] < cutoff - pd.Timedelta(days=30)) & 
                     (reddit_df['Date'] >= cutoff - pd.Timedelta(days=60))]
recent_counts = recent['Keyword'].value_counts()
prev_counts = previous['Keyword'].value_counts()
volume_growth = ((recent_counts - prev_counts) / prev_counts).fillna(0).to_frame(name='Reddit_Growth')
volume_growth.index = volume_growth.index.map(clean_brand)

# --- 3. SENTIMENT SCORE ---
sentiment_df = pd.read_csv("/Users/karishmamehta/Downloads/reddit_sentiment_results.csv", encoding="ISO-8859-1", index_col=0)
sentiment_df.columns = sentiment_df.columns.str.upper().str.strip()
sentiment_df.index = sentiment_df.index.map(clean_brand)
sentiment_df['Sentiment_Score'] = sentiment_df['POSITIVE'] - sentiment_df['NEGATIVE']

# --- 4. BERTOPIC TOPIC COUNT ---
topic_counts = {
    'yeezy': 7,
    'air jordan': 6,
    'asics': 5,
    'new balance': 7,
    'adidas samba': 6
}
topic_df = pd.DataFrame.from_dict(topic_counts, orient='index', columns=['Topic_Count'])
topic_df.index = topic_df.index.map(clean_brand)

# --- Merge all metrics ---
score_df = pd.concat([
    trend_momentum, 
    volume_growth, 
    sentiment_df['Sentiment_Score'].to_frame(), 
    topic_df
], axis=1)

# Drop rows with all NaNs
score_df.dropna(how='all', inplace=True)

# Normalize 0–1
scaler = MinMaxScaler()
score_df[['Google_Momentum', 'Reddit_Growth', 'Sentiment_Score', 'Topic_Count']] = scaler.fit_transform(
    score_df[['Google_Momentum', 'Reddit_Growth', 'Sentiment_Score', 'Topic_Count']]
)

# -------------------- Radar Chart Plot --------------------

import matplotlib.pyplot as plt
import numpy as np

# Radar chart metrics
metrics = ['Google_Momentum', 'Reddit_Growth', 'Sentiment_Score', 'Topic_Count']
num_vars = len(metrics)

# Compute angle for each axis
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]

# Plot setup
fig, ax = plt.subplots(figsize=(8,8), subplot_kw=dict(polar=True))

# Plot each brand
for brand in score_df.index:
    values = score_df.loc[brand, metrics].tolist()
    values += values[:1]
    ax.plot(angles, values, label=brand.title(), linewidth=2)
    ax.fill(angles, values, alpha=0.1)

# Rotate chart
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)

# Remove default labels
ax.set_xticks([])

# Add custom axis labels with smart alignment
for i, label in enumerate(metrics):
    angle_rad = angles[i]
    angle_deg = np.degrees(angle_rad)
    ha = "left" if 0 <= angle_deg <= 90 or 270 <= angle_deg <= 360 else "right"
    rotation = angle_deg if angle_deg <= 180 else angle_deg - 180

    ax.text(
        angle_rad,
        1.15,  # push label out further from center
        label.replace("_", " "),
        horizontalalignment=ha,
        verticalalignment="center",
        rotation=rotation,
        rotation_mode="anchor",
        fontsize=12,
        color="black"
    )

# Add y-axis circles and values
ax.set_rlabel_position(180 / num_vars)
plt.yticks([0.2, 0.5, 0.8], ["0.2", "0.5", "0.8"], color="grey", size=10)
ax.set_ylim(0, 1)

# Final touches
plt.suptitle("Trend Strength Radar Chart by Sneaker Brand", fontsize=14, y=1.15)
plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1.1))
plt.tight_layout()
plt.show()


# In[127]:


import pandas as pd
import matplotlib.pyplot as plt

# --- Assuming score_df is already prepared and normalized as before ---

# Select only the 4 base metrics (exclude final average Trend_Score)
metrics = ['Google_Momentum', 'Reddit_Growth', 'Sentiment_Score', 'Topic_Count']
stacked_data = score_df[metrics].copy()

# Plot
fig, ax = plt.subplots(figsize=(10, 6))
stacked_data.plot(kind='barh', stacked=True, ax=ax, colormap='Set3', width=0.7)

# Title and labels
ax.set_title("Stacked Trend Strength Comparison by Sneaker Brand", fontsize=14, pad=20)
ax.set_xlabel("Cumulative Normalized Score (0–4)", fontsize=12)
ax.set_ylabel("Sneaker Brand", fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.08), ncol=2, fontsize=9, frameon=False)
plt.grid(axis='x', linestyle='--', alpha=0.5)

# Annotate total trend score on bars
for i, (index, row) in enumerate(stacked_data.iterrows()):
    total = row.sum()
    ax.text(total + 0.05, i, f"{total:.2f}", va='center', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.show()


# # Analysis: 
# 🔍 Key Insights:
# New Balance has strong performance across all metrics, especially in Sentiment Score and Topic Count.
# 
# Adidas Samba leads on Reddit Growth, but lags in Google Momentum and Sentiment.
# 
# Air Jordan shows dominance in Google Trends and Sentiment, but Reddit growth is low.
# 
# Yeezy shows balanced strength, but none of the metrics is the absolute highest.
# 
# Asics is middling across most metrics with moderate Reddit activity and sentiment.

# # TIME Series Analysis

# In[134]:


get_ipython().system('pip install prophet')


# In[164]:


# YEEZY


# In[162]:


import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

# Load your CSVs
google_df = pd.read_csv("/Users/karishmamehta/Downloads/sneaker_google_trend.csv", skiprows=1)
reddit_df = pd.read_csv("/Users/karishmamehta/Downloads/reddit_filtered_sneaker_posts.csv")

# --- Google Trends Prep ---
google_df.columns = [col.strip().lower().replace(" (united states)", "") for col in google_df.columns]
google_df.rename(columns={google_df.columns[0]: "week"}, inplace=True)
google_df["week"] = pd.to_datetime(google_df["week"])
brand = "yeezy:"  # change to any brand

google_ts = google_df[["week", brand]].rename(columns={"week": "ds", brand: "y"})
google_ts['y'] = pd.to_numeric(google_ts['y'], errors='coerce').fillna(0)

# --- Forecast Google Trends ---
model_google = Prophet()
model_google.fit(google_ts)
future_google = model_google.make_future_dataframe(periods=12, freq='W')
forecast_google = model_google.predict(future_google)

import matplotlib.dates as mdates

# --- Plotting ---
plt.figure(figsize=(14, 6))

plt.plot(google_ts['ds'], google_ts['y'], label='Actual')
plt.plot(forecast_google['ds'], forecast_google['yhat'], label='Forecast')
plt.fill_between(forecast_google['ds'], forecast_google['yhat_lower'], forecast_google['yhat_upper'], alpha=0.2)

# Fix x-axis: ticks every 2 months, format as abbreviated month + year
ax = plt.gca()
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))

plt.title(f"Google Trends Forecast - {brand.strip(':').title()}")
plt.xlabel("Date")
plt.ylabel("Search Interest")
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()



# In[ ]:


# Air Jordan


# In[168]:


import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

# Load your CSVs
google_df = pd.read_csv("/Users/karishmamehta/Downloads/sneaker_google_trend.csv", skiprows=1)
reddit_df = pd.read_csv("/Users/karishmamehta/Downloads/reddit_filtered_sneaker_posts.csv")

# --- Google Trends Prep ---
google_df.columns = [col.strip().lower().replace(" (united states)", "") for col in google_df.columns]
google_df.rename(columns={google_df.columns[0]: "week"}, inplace=True)
google_df["week"] = pd.to_datetime(google_df["week"])
brand = "air jordan:"  # change to any brand

google_ts = google_df[["week", brand]].rename(columns={"week": "ds", brand: "y"})
google_ts['y'] = pd.to_numeric(google_ts['y'], errors='coerce').fillna(0)

# --- Forecast Google Trends ---
model_google = Prophet()
model_google.fit(google_ts)
future_google = model_google.make_future_dataframe(periods=12, freq='W')
forecast_google = model_google.predict(future_google)

import matplotlib.dates as mdates

# --- Plotting ---
plt.figure(figsize=(14, 6))

plt.plot(google_ts['ds'], google_ts['y'], label='Actual')
plt.plot(forecast_google['ds'], forecast_google['yhat'], label='Forecast')
plt.fill_between(forecast_google['ds'], forecast_google['yhat_lower'], forecast_google['yhat_upper'], alpha=0.2)

# Fix x-axis: ticks every 2 months, format as abbreviated month + year
ax = plt.gca()
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))

plt.title(f"Google Trends Forecast - {brand.strip(':').title()}")
plt.xlabel("Date")
plt.ylabel("Search Interest")
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()



# In[ ]:


# Asics


# In[170]:


import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

# Load your CSVs
google_df = pd.read_csv("/Users/karishmamehta/Downloads/sneaker_google_trend.csv", skiprows=1)
reddit_df = pd.read_csv("/Users/karishmamehta/Downloads/reddit_filtered_sneaker_posts.csv")

# --- Google Trends Prep ---
google_df.columns = [col.strip().lower().replace(" (united states)", "") for col in google_df.columns]
google_df.rename(columns={google_df.columns[0]: "week"}, inplace=True)
google_df["week"] = pd.to_datetime(google_df["week"])
brand = "asics:"  # change to any brand

google_ts = google_df[["week", brand]].rename(columns={"week": "ds", brand: "y"})
google_ts['y'] = pd.to_numeric(google_ts['y'], errors='coerce').fillna(0)

# --- Forecast Google Trends ---
model_google = Prophet()
model_google.fit(google_ts)
future_google = model_google.make_future_dataframe(periods=12, freq='W')
forecast_google = model_google.predict(future_google)

import matplotlib.dates as mdates

# --- Plotting ---
plt.figure(figsize=(14, 6))

plt.plot(google_ts['ds'], google_ts['y'], label='Actual')
plt.plot(forecast_google['ds'], forecast_google['yhat'], label='Forecast')
plt.fill_between(forecast_google['ds'], forecast_google['yhat_lower'], forecast_google['yhat_upper'], alpha=0.2)

# Fix x-axis: ticks every 2 months, format as abbreviated month + year
ax = plt.gca()
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))

plt.title(f"Google Trends Forecast - {brand.strip(':').title()}")
plt.xlabel("Date")
plt.ylabel("Search Interest")
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()


# In[ ]:


# New Balance


# In[172]:


import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

# Load your CSVs
google_df = pd.read_csv("/Users/karishmamehta/Downloads/sneaker_google_trend.csv", skiprows=1)
reddit_df = pd.read_csv("/Users/karishmamehta/Downloads/reddit_filtered_sneaker_posts.csv")

# --- Google Trends Prep ---
google_df.columns = [col.strip().lower().replace(" (united states)", "") for col in google_df.columns]
google_df.rename(columns={google_df.columns[0]: "week"}, inplace=True)
google_df["week"] = pd.to_datetime(google_df["week"])
brand = "new balance:"  # change to any brand

google_ts = google_df[["week", brand]].rename(columns={"week": "ds", brand: "y"})
google_ts['y'] = pd.to_numeric(google_ts['y'], errors='coerce').fillna(0)

# --- Forecast Google Trends ---
model_google = Prophet()
model_google.fit(google_ts)
future_google = model_google.make_future_dataframe(periods=12, freq='W')
forecast_google = model_google.predict(future_google)

import matplotlib.dates as mdates

# --- Plotting ---
plt.figure(figsize=(14, 6))

plt.plot(google_ts['ds'], google_ts['y'], label='Actual')
plt.plot(forecast_google['ds'], forecast_google['yhat'], label='Forecast')
plt.fill_between(forecast_google['ds'], forecast_google['yhat_lower'], forecast_google['yhat_upper'], alpha=0.2)

# Fix x-axis: ticks every 2 months, format as abbreviated month + year
ax = plt.gca()
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))

plt.title(f"Google Trends Forecast - {brand.strip(':').title()}")
plt.xlabel("Date")
plt.ylabel("Search Interest")
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()


# In[174]:


# ADIDAS SAMBA


# In[178]:


import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

# Load your CSVs
google_df = pd.read_csv("/Users/karishmamehta/Downloads/sneaker_google_trend.csv", skiprows=1)
reddit_df = pd.read_csv("/Users/karishmamehta/Downloads/reddit_filtered_sneaker_posts.csv")

# --- Google Trends Prep ---
google_df.columns = [col.strip().lower().replace(" (united states)", "") for col in google_df.columns]
google_df.rename(columns={google_df.columns[0]: "week"}, inplace=True)
google_df["week"] = pd.to_datetime(google_df["week"])
brand = "adidas samba:"  # change to any brand

google_ts = google_df[["week", brand]].rename(columns={"week": "ds", brand: "y"})
google_ts['y'] = pd.to_numeric(google_ts['y'], errors='coerce').fillna(0)

# --- Forecast Google Trends ---
model_google = Prophet()
model_google.fit(google_ts)
future_google = model_google.make_future_dataframe(periods=12, freq='W')
forecast_google = model_google.predict(future_google)

import matplotlib.dates as mdates

# --- Plotting ---
plt.figure(figsize=(14, 6))

plt.plot(google_ts['ds'], google_ts['y'], label='Actual')
plt.plot(forecast_google['ds'], forecast_google['yhat'], label='Forecast')
plt.fill_between(forecast_google['ds'], forecast_google['yhat_lower'], forecast_google['yhat_upper'], alpha=0.2)

# Fix x-axis: ticks every 2 months, format as abbreviated month + year
ax = plt.gca()
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))

plt.title(f"Google Trends Forecast - {brand.strip(':').title()}")
plt.xlabel("Date")
plt.ylabel("Search Interest")
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()


# # Collective forecasting

# In[185]:


import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

# Load and clean data
google_df = pd.read_csv("sneaker_google_trend.csv", skiprows=1)
google_df.columns = [col.strip().lower().replace(" (united states)", "") for col in google_df.columns]
google_df.rename(columns={google_df.columns[0]: "week"}, inplace=True)
google_df["week"] = pd.to_datetime(google_df["week"])

brands = ["yeezy:", "air jordan:", "asics:", "adidas samba:", "new balance:"]
all_forecasts = []

# Forecast loop for each brand
for brand in brands:
    ts = google_df[["week", brand]].rename(columns={"week": "ds", brand: "y"})
    ts["y"] = pd.to_numeric(ts["y"], errors='coerce').fillna(0)

    model = Prophet()
    model.fit(ts)

    future = model.make_future_dataframe(periods=12, freq='W')
    forecast = model.predict(future)
    forecast["brand"] = brand.replace(":", "").title()
    all_forecasts.append(forecast)

# Combine forecasts
brand_forecasts = pd.concat(all_forecasts)

# Plotting
plt.figure(figsize=(15, 7))
brands_clean = [b.replace(":", "").title() for b in brands]

for brand in brands_clean:
    brand_data = brand_forecasts[brand_forecasts["brand"] == brand]
    plt.plot(brand_data["ds"], brand_data["yhat"], label=brand)
    plt.fill_between(brand_data["ds"], brand_data["yhat_lower"], brand_data["yhat_upper"], alpha=0.15)

plt.title("Combined Google Trends Forecast with Confidence Intervals")
plt.xlabel("Date")
plt.ylabel("Forecasted Search Interest")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

