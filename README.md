# 🛍️ Fashion Recommendation System

An AI-powered fashion recommendation web app built with Python and Streamlit,
featuring a Myntra-style UI with smart product suggestions.

## ✨ Features

- 🔐 Login authentication
- 👩👨 Browse by Gender — Women & Men
- 🗂️ Category-wise product browsing (Saree, Kurta, Dress, Jeans, etc.)
- 🔍 Search, filter by brand, price range & sort options
- 🤖 ML-powered similar product recommendations (TF-IDF + Cosine Similarity)
- 🎨 Myntra-style luxury UI with gold & charcoal theme

## 🧠 How It Works

Product features (name, brand, colour, description) are combined into tags →
converted to TF-IDF vectors → cosine similarity finds the top 5 most similar
products in the same category.

## 🚀 Run Locally

```bash
git clone https://github.com/sasimadhuri28/Fashion-Recommendation-System
cd Fashion-Recommendation-System
pip install -r requirements.txt
streamlit run app.py
```

## 🔑 Login Credentials
- **Username:** admin
- **Password:** fashion123

## 🛠️ Tech Stack
- Python, Pandas, Scikit-learn
- Streamlit
- TF-IDF Vectorizer + Cosine Similarity