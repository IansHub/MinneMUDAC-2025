import pandas as pd
import numpy as np
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter


BBBS_df = pd.read_excel("./undergrade/Training-Restated.xlsx")
#print(BBBS_df["Match Support Contact Notes"].head())
strings_to_remove = ("Question:", "Activities:", "Answer:", "Child Safety:", "Child Development:", "Child/Volunteer Relationship development:", "Relationship with BBBS:", "Parent/Volunteer Concerns:", "Other Comments-List progress/activities in school and in JJ System:" "MSS Notes:", "Match Closing with bs via email:")
BBBS_df["Match Support Contact Notes Cleaned"] = BBBS_df["Match Support Contact Notes"].str.replace(strings_to_remove, "")
Match_Support_str_list = [str(i) for i in BBBS_df["Match Support Contact Notes"]]
#print(Match_Support_str_list)
Match_Support_text = "".join(Match_Support_str_list).lower()
BBBS_wordcloud = WordCloud(width=1000, height=1000).generate(Match_Support_text)
plt.imshow(BBBS_wordcloud)
plt.axis('off')
plt.title('Language Match Support Contact Notes')
plt.show()

nltk.download(['punkt_tab', 'stopwords', 'wordnet', 'omw-1.4', 'vader_lexicon'])
notes_series = BBBS_df["Match Support Contact Notes"]


def preprocess_text(text):
    text = str(text)
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    custom_stopwords = ["question", "answer"]
    filtered_tokens = [word for word in tokens
                       if word.isalpha()
                       and word not in stop_words
                       and word not in custom_stopwords
                       and len(word) > 2]
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]

    return lemmatized_tokens

notes_text = ' '.join(BBBS_df["Match Support Contact Notes"].dropna())

processed_tokens = preprocess_text(notes_text)

# Calculate word frequencies
word_freq = Counter(processed_tokens)

# Generate word cloud
wordcloud = WordCloud(width=800, height=400,
                      max_words=200).generate_from_frequencies(word_freq)

# Display
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation="bilinear")
plt.title(f"Match Support Notes Cloud", fontsize=14)
plt.axis("off")
plt.show()