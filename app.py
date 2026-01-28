import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- PAGE CONFIG ----------------

st.set_page_config(page_title="Fashion Recommendation System", layout="wide")

# ---------------- LOGIN ----------------

USERNAME = "admin"
PASSWORD = "fashion123"

st.sidebar.title("Login")

username = st.sidebar.text_input("Username")
password = st.sidebar.text_input("Password", type="password")

if username != USERNAME or password != PASSWORD:
    st.sidebar.warning("Please login to continue")
    st.stop()

st.sidebar.success("Login Successful ✅")

# ---------------- LOAD DATA ----------------

df = pd.read_csv("data/fashion.csv")

# Remove unwanted column
if "Unnamed: 0" in df.columns:
    df.drop(columns=["Unnamed: 0"], inplace=True)

# Remove missing values
df.dropna(subset=["name", "brand", "colour", "description", "price", "avg_rating", "img"], inplace=True)

# IMPORTANT FIX: reset index after cleaning
df.reset_index(drop=True, inplace=True)

# ---------------- CATEGORY CLASSIFICATION ----------------

def get_category(row):

    text = (row["name"] + " " + row["description"]).lower()

    if "saree" in text:
        return "Saree"

    elif "kurta" in text or "kurti" in text:
        return "Kurta"

    elif "dress" in text or "gown" in text:
        return "Dress"

    elif "jeans" in text or "denim" in text:
        return "Jeans"

    elif "trouser" in text or "pant" in text or "palazzo" in text:
        return "Trousers"

    elif "shirt" in text:
        return "Shirt"

    elif "top" in text or "tunic" in text:
        return "Top"

    elif "jacket" in text or "coat" in text:
        return "Jacket"

    else:
        return "Others"


df["category"] = df.apply(get_category, axis=1)

# ---------------- CREATE TAGS ----------------

df["tags"] = df["name"] + " " + df["brand"] + " " + df["colour"] + " " + df["description"]

# ---------------- TF-IDF MODEL ----------------

tfidf = TfidfVectorizer(stop_words="english")
vectors = tfidf.fit_transform(df["tags"])
similarity = cosine_similarity(vectors)

# ---------------- RECOMMEND FUNCTION ----------------

def recommend(product_name):

    selected_index = df[df["name"] == product_name].index[0]
    selected_category = df.iloc[selected_index]["category"]

    distances = list(enumerate(similarity[selected_index]))
    distances = sorted(distances, key=lambda x: x[1], reverse=True)

    recommendations = []

    for i in distances:

        product = df.iloc[i[0]]

        # Skip same product
        if product["name"] == product_name:
            continue

        # Same category only
        if product["category"] == selected_category:
            recommendations.append(product)

        if len(recommendations) == 5:
            break

    return recommendations

# ---------------- SIDEBAR FILTERS ----------------

st.sidebar.header("Filters")

search_text = st.sidebar.text_input("Search Product")

min_price = int(df["price"].min())
max_price = int(df["price"].max())

price_range = st.sidebar.slider(
    "Select Price Range",
    min_price,
    max_price,
    (min_price, max_price)
)

category_list = ["All"] + sorted(df["category"].unique())
selected_category_filter = st.sidebar.selectbox("Select Category", category_list)

sort_rating = st.sidebar.checkbox("Sort by Top Rating")

# ---------------- APPLY FILTERS ----------------

filtered_df = df.copy()

# Search filter
if search_text:
    filtered_df = filtered_df[
        filtered_df["name"].str.contains(search_text, case=False, na=False)
    ]

# Price filter
filtered_df = filtered_df[
    (filtered_df["price"] >= price_range[0]) &
    (filtered_df["price"] <= price_range[1])
]

# Category filter
if selected_category_filter != "All":
    filtered_df = filtered_df[filtered_df["category"] == selected_category_filter]

# Rating sort
if sort_rating:
    filtered_df = filtered_df.sort_values(by="avg_rating", ascending=False)

# ---------------- MAIN UI ----------------

st.title("👗 Fashion Recommendation System")

if filtered_df.empty:
    st.warning("No products found with applied filters.")
    st.stop()

selected_product = st.selectbox(
    "Select a Fashion Product",
    filtered_df["name"].values
)

# ---------------- SHOW SELECTED PRODUCT ----------------

selected_index = df[df["name"] == selected_product].index[0]
selected_row = df.iloc[selected_index]

st.image(selected_row["img"], width=300)

st.subheader("Selected Product Details")

st.write("Brand:", selected_row["brand"])
st.write("Color:", selected_row["colour"])
st.write("Category:", selected_row["category"])
st.write("Price: ₹", int(selected_row["price"]))
st.write("Rating:", round(selected_row["avg_rating"], 2))

# ---------------- RECOMMEND BUTTON ----------------

if st.button("Recommend"):

    recommendations = recommend(selected_product)

    st.subheader("Recommended Products")

    for product in recommendations:

        col1, col2 = st.columns([1, 3])

        with col1:
            st.image(product["img"], width=160)

        with col2:
            st.write("###", product["name"])
            st.write("Brand:", product["brand"])
            st.write("Color:", product["colour"])
            st.write("Category:", product["category"])
            st.write("Price: ₹", int(product["price"]))
            st.write("Rating:", round(product["avg_rating"], 2))

        st.markdown("---")
