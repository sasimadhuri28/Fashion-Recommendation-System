import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load cleaned dataset
df = pd.read_csv("data/cleaned_fashion.csv")

print("Cleaned Dataset Loaded!")
print(df.head())

# Convert text (tags) into numerical vectors
tfidf = TfidfVectorizer(stop_words='english')

vectors = tfidf.fit_transform(df['tags'])

print("\nTF-IDF Vector Shape:")
print(vectors.shape)

# Calculate cosine similarity
similarity = cosine_similarity(vectors)

print("\nSimilarity Matrix Shape:")
print(similarity.shape)

# Recommendation function
def recommend(product_name):
    index = df[df['name'] == product_name].index[0]

    distances = list(enumerate(similarity[index]))

    distances = sorted(distances, key=lambda x: x[1], reverse=True)

    print("\nRecommended Products:\n")

    for i in distances[1:6]:
        print(df.iloc[i[0]]['name'])



# Test the system with one product
sample_product = df['name'][10]
print("\nSelected Product:")
print(sample_product)

recommend(sample_product)
print("\nChecking image column values:")
print(df['img'].head(10))



df = pd.read_csv("data/cleaned_fashion.csv")
print(df.columns)
