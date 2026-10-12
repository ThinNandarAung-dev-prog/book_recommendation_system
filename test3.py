import pandas as pd
import json
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
import numpy as np

# Load the dataset
df = pd.read_csv('./amazon/books_data.csv', delimiter=',', encoding='utf-8', on_bad_lines='skip', low_memory=False)

# Remove rows with missing essential fields
df = df.dropna(subset=['Title', 'authors', 'publisher', 'categories', 'publishedDate','image'])

# Replace non-numeric values in 'Year' with NaN and convert the column to numeric
df['Year'] = pd.to_numeric(df['publishedDate'].str[:4], errors='coerce')

# Fill NaN values in 'Year' with the median of 'Year'
df['Year'] = df['Year'].fillna(df['Year'].median())

# Normalize 'Year' using MinMaxScaler
scaler = MinMaxScaler()
df['Year'] = scaler.fit_transform(df[['Year']])

# Sample a subset of the data for testing (adjust size as needed)
df_sample = df.sample(n=10000, random_state=1)

# Function to safely parse authors column
def parse_authors(authors_str):
    if pd.isna(authors_str):  # Handle NaN values
        return None
    try:
        # Convert string representation of list to actual list
        authors_list = json.loads(authors_str.replace("'", '"'))
        return authors_list[0] if isinstance(authors_list, list) and authors_list else None
    except (json.JSONDecodeError, TypeError):
        return None

# Function to safely parse categories column
def parse_categories(categories_str):
    if pd.isna(categories_str):  # Handle NaN values
        return None
    try:
        # Convert string representation of list to actual list
        categories_list = json.loads(categories_str.replace("'", '"'))
        return categories_list if isinstance(categories_list, list) else [categories_str]
    except (json.JSONDecodeError, TypeError):
        return [categories_str] if isinstance(categories_str, str) else None

# Apply the function to the authors and categories columns
df_sample['author'] = df_sample['authors'].apply(parse_authors)
df_sample['categories'] = df_sample['categories'].apply(parse_categories)
# Convert categories to a comma-separated string for easy writing
df_sample['categories_str'] = df_sample['categories'].apply(lambda x: ', '.join(x) if x else '')

# One-hot encode Author
df_author = pd.get_dummies(df_sample['author'])

# One-hot encode Publisher
df_publisher = pd.get_dummies(df_sample['publisher'])

# Use TF-IDF to convert Book-Title into vectorized form
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(max_features=10000, stop_words='english')  # Limiting to 10000 features to save memory
title_tfidf = tfidf.fit_transform(df_sample['Title']).toarray()

# Use TF-IDF for Categories
categories_tfidf = tfidf.fit_transform(df_sample['categories_str']).toarray()

# Combine all the features (Title, Author, Publisher, Categories, Year)
features = pd.concat([
    2*pd.DataFrame(title_tfidf, index=df_sample.index),  # Book title (TF-IDF)
    pd.DataFrame(categories_tfidf, index=df_sample.index),  # Categories (TF-IDF)
    df_author,                                         # Author (one-hot)
    df_publisher,                                      # Publisher (one-hot)
    df_sample[['Year']]                                # Year of publication (normalized)
], axis=1)

# Ensure all column names are strings
features.columns = features.columns.astype(str)

# Dimensionality reduction with PCA
pca = PCA(n_components=100)
features_reduced = pca.fit_transform(features)

# Batch processing for cosine similarity
def batch_cosine_similarity(features, batch_size=1000):
    n_samples = features.shape[0]
    similarity_matrix = np.zeros((n_samples, n_samples))  # Preallocate memory
    
    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        # Only calculate similarities for a batch at a time
        batch_similarity = cosine_similarity(features[start_idx:end_idx], features)
        similarity_matrix[start_idx:end_idx] = batch_similarity
    
    return similarity_matrix

# Calculate Cosine Similarity for the reduced feature set in batches
similarity_matrix = batch_cosine_similarity(features_reduced, batch_size=1000)

# Create a DataFrame to represent the similarity between books
similarity_df = pd.DataFrame(similarity_matrix, index=df_sample['Title'], columns=df_sample['Title'])

# Function to get the top N similar books for a given book title
def get_similar_books(title, top_n=10):
    # Normalize the input title
    normalized_title = title.strip().lower()
    
    # Normalize all titles in the similarity_df
    normalized_titles = similarity_df.index.str.strip().str.lower()
    
    # Check if the normalized title is in the dataset
    if normalized_title not in normalized_titles:
        return "Title not found in the dataset."
    
    # Find the actual title in the index
    actual_title = similarity_df.index[normalized_titles == normalized_title].tolist()[0]
    
    # Get the similarity scores for the given title
    sim_scores = similarity_df.loc[actual_title].sort_values(ascending=False)
    
    # Exclude the book itself from the results
    sim_scores = sim_scores[sim_scores.index != actual_title]
    
    # Get the top N similar books
    top_books = sim_scores.head(top_n).index.tolist()

    # Display more information about each similar book
    similar_books_info = []
    for book_title in top_books:
        book_row = df_sample[df_sample['Title'] == book_title].iloc[0]
        similar_books_info.append({
            "Title": book_title,
            "Image":book_row['image'] if book_row['image'] else "Unknown",
            "Author": book_row['author'] if book_row['author'] else "Unknown",
            "Publisher": book_row['publisher'] if pd.notna(book_row['publisher']) else "Unknown",
            "Year": book_row['Year'],
            "Categories": book_row['categories_str'] if pd.notna(book_row['categories_str']) else "Unknown"
        })
    
    # Return a list of books with their details
    return similar_books_info

# Write the titles, authors, year, and categories to a text file
with open('sampled_titles_info.txt', 'w', encoding='utf-8') as f:
    for index, row in df_sample.iterrows():
        title = row['Title']
        author = row['author'] if row['author'] else "Unknown"
        year = row['Year']
        categories = row['categories_str']
        publisher = row['publisher'] if row['publisher'] else "Unknown"
        
        f.write(f"Title: {title}\n")
        f.write(f"Author: {author}\n")
        f.write(f"Publisher: {publisher}\n")
        f.write(f"Year: {year}\n")
        f.write(f"Categories: {categories}\n")
        f.write("\n")  # Add a new line between entries
print("Sampled titles, authors, years, and categories have been saved to 'sampled_titles_info.txt'.")

# Print out the sample titles with additional info
#print("Sampled Titles with Info:")
#for index, row in df_sample.iterrows():
#    print(f"Title: {row['Title']}, Author: {row['author']}, Year: {row['Year']}, Categories: {row['categories_str']}")

# Example usage of similarity function
'''
book_title = input("Enter the book title to find similar ones: ")

# Ensure the input is valid
if book_title.strip() == "":
    print("Input cannot be empty.")
else:
    top_books = get_similar_books(book_title, top_n=10)
    
    # Display the result
    print(f"Top 10 books similar to '{book_title}':")
    for i, book in enumerate(top_books, 1):
        print(f"\n{i}. Title: {book['Title']}")
        print(f"   Author: {book['Author']}")
        print(f"   Publisher: {book['Publisher']}")
        print(f"   Year: {book['Year']}")
        print(f"   Categories: {book['Categories']}")
'''