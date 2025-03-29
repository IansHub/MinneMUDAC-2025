# -*- coding: utf-8 -*-
"""
Created on Mon Mar 24 16:22:42 2025

@author: Pyang
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

# Load your data
df = pd.read_excel("Training-Restated.xlsx")

# Clean column names
df.columns = df.columns.str.strip().str.replace(" ", "_").str.lower()

# Select only the fields we need
notes_df = df[['match_length', 'match_support_contact_notes']].dropna()

# Create a match_type label (long vs short based on median)
median_length = notes_df['match_length'].median()
notes_df['match_type'] = np.where(notes_df['match_length'] >= median_length, 'long', 'short')

# Sample 1000 from each group to reduce memory usage
sampled_df = notes_df.copy()

# Use CountVectorizer to tokenize words
vectorizer = CountVectorizer(stop_words='english', max_features=150)
X = vectorizer.fit_transform(sampled_df['match_support_contact_notes'])

# Convert to DataFrame
word_counts = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())
word_counts['match_type'] = sampled_df['match_type'].values

# Calculate average word frequency by group
word_summary = word_counts.groupby('match_type').mean().T
word_summary['difference'] = word_summary['long'] - word_summary['short']
word_summary_sorted = word_summary.sort_values('difference', ascending=False)

# Show results
print("\nTop 10 Keywords in Long Matches:")
print(word_summary_sorted.head(10))

print("\nTop 10 Keywords in Short Matches:")
print(word_summary_sorted.tail(10))
