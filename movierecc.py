import pandas as pd

# Load the dataset
file_path = 'C:/Users/HOME/python/movies.csv'
movies_df = pd.read_csv(file_path)

# Display the first few rows of the dataset
movies_df.head()

# Preprocess the data
# Fill NaN values with empty strings
movies_df['genres'] = movies_df['genres'].fillna('')
movies_df['keywords'] = movies_df['keywords'].fillna('')
movies_df['overview'] = movies_df['overview'].fillna('')
movies_df['cast'] = movies_df['cast'].fillna('')
movies_df['crew'] = movies_df['crew'].fillna('')
movies_df['director'] = movies_df['director'].fillna('')

# Combine features into a single string
def combine_features(row):
    return f"{row['genres']} {row['keywords']} {row['overview']} {row['cast']} {row['crew']} {row['director']}"

movies_df['combined_features'] = movies_df.apply(combine_features, axis=1)

# Display the combined features for the first few movies
movies_df['combined_features'].head()

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Initialize the TF-IDF Vectorizer
tfidf = TfidfVectorizer(stop_words='english')

# Fit and transform the data into a TF-IDF matrix
tfidf_matrix = tfidf.fit_transform(movies_df['combined_features'])

# Compute the cosine similarity matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Function to get movie recommendations
def get_recommendations(title, cosine_sim=cosine_sim):
    # Get the index of the movie that matches the title
    idx = movies_df[movies_df['title'] == title].index[0]

    # Get the pairwise similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar movies
    sim_scores = sim_scores[1:11]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar movies
    return movies_df['title'].iloc[movie_indices]

# Example usage: Get recommendations for a specific movie
print(get_recommendations('Avatar'))
