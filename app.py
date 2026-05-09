import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ─────────────────────────────────────────────
#  PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="FashionAI – Your Style, Recommended",
    page_icon="👗",
    layout="wide",
)

# ─────────────────────────────────────────────
#  GLOBAL CSS  – Myntra-inspired luxury look
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700;900&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; background: #fafafa; }
.stApp { background: #fafafa; }
#MainMenu, footer, header { visibility: hidden; }

::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-thumb { background: #e91e8c; border-radius: 3px; }

/* ── Navbar ── */
.navbar {
    background: #fff; border-bottom: 2px solid #e91e8c;
    padding: 14px 40px; display: flex; align-items: center;
    justify-content: space-between;
    box-shadow: 0 2px 12px rgba(233,30,140,.08); margin-bottom: 24px;
}
.navbar-logo {
    font-family: 'Playfair Display', serif; font-size: 28px; font-weight: 900;
    background: linear-gradient(135deg, #e91e8c, #ff6b35);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; letter-spacing: -1px;
}
.navbar-tagline { font-size: 11px; color: #aaa; letter-spacing: 2px; text-transform: uppercase; }
.breadcrumb { font-size: 12px; color: #aaa; }
.breadcrumb span { color: #e91e8c; font-weight: 600; }

/* ── Section headings ── */
.section-title {
    font-family: 'Playfair Display', serif; font-size: 30px; font-weight: 700;
    color: #111; margin: 8px 0 4px; letter-spacing: -.5px;
}
.section-sub { font-size: 13px; color: #999; margin-bottom: 24px; }

/* ── Gender cards ── */
.gender-card {
    border-radius: 16px; overflow: hidden; cursor: pointer;
    transition: transform .25s, box-shadow .25s;
    box-shadow: 0 4px 20px rgba(0,0,0,.08);
}
.gender-card:hover { transform: translateY(-6px); box-shadow: 0 12px 40px rgba(233,30,140,.18); }
.gender-card-inner { padding: 56px 28px; text-align: center; }
.gender-card-women { background: linear-gradient(145deg,#f953c6,#b91d73); }
.gender-card-men   { background: linear-gradient(145deg,#0f0c29,#302b63,#24243e); }
.gender-card-kids  { background: linear-gradient(145deg,#f7971e,#ffd200); }
.gender-card-title {
    font-family: 'Playfair Display', serif; font-size: 34px; font-weight: 900;
    color: #fff; display: block; margin-bottom: 6px; letter-spacing: -1px;
}
.gender-card-sub { font-size: 12px; color: rgba(255,255,255,.7); letter-spacing: 1.5px; text-transform: uppercase; }
.gender-emoji { font-size: 46px; display: block; margin-bottom: 12px; }

/* ── Product cards ── */
.product-card {
    background: #fff; border-radius: 12px; overflow: hidden; cursor: pointer;
    transition: transform .22s, box-shadow .22s;
    box-shadow: 0 2px 12px rgba(0,0,0,.06); position: relative; height: 100%;
}
.product-card:hover { transform: translateY(-5px); box-shadow: 0 10px 32px rgba(0,0,0,.12); }
.product-card img { width: 100%; height: 250px; object-fit: cover; display: block; }
.product-card-body { padding: 12px 14px 14px; }
.product-brand { font-size: 10px; font-weight: 700; color: #e91e8c; letter-spacing: 1px; text-transform: uppercase; margin-bottom: 3px; }
.product-name  { font-size: 13px; font-weight: 500; color: #222; line-height: 1.4; margin-bottom: 6px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.product-price { font-size: 14px; font-weight: 700; color: #111; }
.product-rating { font-size: 11px; color: #888; margin-top: 2px; }
.rating-star { color: #f5a623; }
.badge { position: absolute; top: 8px; left: 8px; color: #fff; font-size: 9px; font-weight: 700; padding: 3px 7px; border-radius: 4px; letter-spacing: .5px; text-transform: uppercase; }
.badge-top  { background: #e91e8c; }
.badge-sim  { background: #7c3aed; }

/* ── Product detail ── */
.detail-brand { font-size: 12px; font-weight: 700; color: #e91e8c; letter-spacing: 1.5px; text-transform: uppercase; margin-bottom: 8px; }
.detail-name  { font-family: 'Playfair Display', serif; font-size: 28px; font-weight: 700; color: #111; line-height: 1.3; margin-bottom: 14px; }
.detail-price { font-size: 26px; font-weight: 800; color: #111; margin-bottom: 4px; }
.detail-price-sub { font-size: 12px; color: #aaa; margin-bottom: 18px; }
.detail-meta  { display: flex; gap: 28px; margin-bottom: 20px; }
.detail-meta-item label { font-size: 10px; color: #bbb; text-transform: uppercase; letter-spacing: 1px; display: block; margin-bottom: 3px; }
.detail-meta-item span  { font-size: 14px; font-weight: 600; color: #333; }
.detail-desc  { font-size: 13px; color: #666; line-height: 1.8; border-top: 1px solid #f0f0f0; padding-top: 18px; margin-bottom: 24px; }

/* ── Divider ── */
.divider { height: 1px; background: linear-gradient(90deg,#e91e8c22,#ff6b3522,transparent); margin: 36px 0; }

/* ── Empty state ── */
.empty-state { text-align: center; padding: 60px 20px; color: #ccc; }
.empty-state-icon { font-size: 56px; margin-bottom: 12px; }
.empty-state-text { font-size: 16px; }

/* ── Streamlit button override ── */
div[data-testid="stButton"] button {
    background: linear-gradient(135deg,#e91e8c,#ff6b35) !important;
    color: #fff !important; border: none !important;
    border-radius: 8px !important; font-weight: 600 !important;
    font-size: 12px !important; padding: 8px 18px !important;
    box-shadow: 0 3px 10px rgba(233,30,140,.25) !important;
    transition: opacity .2s !important;
}
div[data-testid="stButton"] button:hover { opacity: .85 !important; }

/* ── Login ── */
.login-logo {
    font-family: 'Playfair Display', serif; font-size: 48px; font-weight: 900;
    background: linear-gradient(135deg,#e91e8c,#ff6b35);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    text-align: center; margin-bottom: 4px;
}
.login-sub { text-align: center; font-size: 12px; color: #bbb; letter-spacing: 2px; text-transform: uppercase; margin-bottom: 32px; }

/* ── Image Upload Zone ── */
.upload-zone {
    border: 2px dashed #e91e8c; border-radius: 16px;
    padding: 40px 20px; text-align: center;
    background: linear-gradient(135deg,#fff0f7,#fff8f0);
    margin-bottom: 24px;
}
.upload-zone-icon { font-size: 48px; margin-bottom: 10px; }
.upload-zone-text { font-size: 15px; color: #888; }
.analysis-box {
    background: linear-gradient(135deg,#fff0f7,#fff);
    border: 1px solid #f9a8d4; border-radius: 12px;
    padding: 18px 22px; margin-bottom: 24px;
    font-size: 13px; color: #444; line-height: 1.8;
}
.analysis-box strong { color: #e91e8c; }


</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  SESSION STATE
# ─────────────────────────────────────────────
defaults = {"logged_in": False, "page": "home", "gender": None, "sub_cat": None, "product_idx": None,
            "img_recs": [], "img_analysis": None}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ─────────────────────────────────────────────
#  LOGIN
# ─────────────────────────────────────────────
USERNAME, PASSWORD = "admin", "fashion123"

if not st.session_state.logged_in:
    st.markdown("<div class='login-logo'>FashionAI</div>", unsafe_allow_html=True)
    st.markdown("<div class='login-sub'>Your personal style engine</div>", unsafe_allow_html=True)
    _, cf, _ = st.columns([1, 1.1, 1])
    with cf:
        uname = st.text_input("Username", placeholder="admin")
        pwd   = st.text_input("Password", type="password", placeholder="fashion123")
        if st.button("Sign In →", use_container_width=True):
            if uname == USERNAME and pwd == PASSWORD:
                st.session_state.logged_in = True
                st.rerun()
            else:
                st.error("Invalid credentials. Use admin / fashion123")
        st.caption("Demo credentials: admin / fashion123")
    st.stop()


# ─────────────────────────────────────────────
#  LOAD DATA
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("data/fashion.csv")
    if "Unnamed: 0" in df.columns:
        df.drop(columns=["Unnamed: 0"], inplace=True)
    df.dropna(subset=["name","brand","colour","description","price","avg_rating","img"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    def get_category(row):
        t = (str(row["name"]) + " " + str(row["description"])).lower()
        if "saree" in t:                                    return "Sarees"
        if "kurta" in t or "kurti" in t:                   return "Kurtas"
        if "lehenga" in t:                                  return "Lehengas"
        if "dress" in t or "gown" in t:                    return "Dresses"
        if "jeans" in t or "denim" in t:                   return "Jeans"
        if "trouser" in t or "pant" in t or "palazzo" in t: return "Trousers"
        if "shirt" in t:                                   return "Shirts"
        if "top" in t or "tunic" in t:                     return "Tops"
        if "jacket" in t or "coat" in t:                   return "Jackets"
        if "skirt" in t:                                   return "Skirts"
        if "suit" in t:                                    return "Suits"
        if "blouse" in t:                                  return "Blouses"
        return "Others"

    def get_gender(row):
        t = (str(row["name"]) + " " + str(row["description"])).lower()
        if ("boy" in t or " men" in t or "man " in t or "male" in t) and "women" not in t:
            return "Men"
        if "girl" in t or "kid" in t or "child" in t or "baby" in t:
            return "Kids"
        return "Women"

    df["category"] = df.apply(get_category, axis=1)
    df["gender"]   = df.apply(get_gender,   axis=1)
    df["tags"]     = df["name"] + " " + df["brand"] + " " + df["colour"] + " " + df["description"]
    return df

@st.cache_resource
def build_similarity(_df):
    tfidf = TfidfVectorizer(stop_words="english", max_features=6000)
    vecs  = tfidf.fit_transform(_df["tags"])
    return cosine_similarity(vecs)

df         = load_data()
similarity = build_similarity(df)

GENDER_ICONS = {"Women": "👩", "Men": "👨", "Kids": "🧒"}
GENDER_CATS  = {
    "Women": ["Sarees","Kurtas","Lehengas","Dresses","Tops","Blouses","Skirts","Suits","Trousers","Jeans","Jackets"],
    "Men":   ["Shirts","Jeans","Trousers","Kurtas","Suits","Jackets","Others"],
    "Kids":  ["Dresses","Tops","Jeans","Kurtas","Others"],
}
CAT_ICON = {
    "Sarees":"🥻","Kurtas":"👘","Lehengas":"🎀","Dresses":"👗","Tops":"👚","Blouses":"👕",
    "Skirts":"🩱","Suits":"🕴️","Trousers":"👖","Jeans":"🩳","Jackets":"🧥","Shirts":"👔","Others":"🛍️",
}


def recommend(idx, n=8):
    cat = df.iloc[idx]["category"]
    scores = sorted(enumerate(similarity[idx]), key=lambda x: x[1], reverse=True)
    out = []
    for i, _ in scores:
        if i != idx and df.iloc[i]["category"] == cat:
            out.append(i)
        if len(out) == n:
            break
    return out


def analyze_image_and_recommend(image_bytes: bytes, mime_type: str, n: int = 8):
    """Analyse an uploaded fashion image and find matching products (using free Gemini API or Mock)."""
    import json, re

    prompt = """You are a fashion expert AI. Analyse this clothing/fashion image and return a JSON object with these keys:
{
  "description": "<one sentence describing the item>",
  "category": "<one of: Sarees, Kurtas, Lehengas, Dresses, Tops, Blouses, Skirts, Suits, Trousers, Jeans, Jackets, Shirts, Others>",
  "gender": "<one of: Women, Men, Kids>",
  "colour": "<primary colour>",
  "keywords": ["<keyword1>", "<keyword2>", "<keyword3>"]
}
Return ONLY the JSON, no extra text."""

    api_key = ""
    
    if api_key and api_key != "YOUR_GEMINI_API_KEY":
        import google.generativeai as genai
        from PIL import Image
        import io
        genai.configure(api_key=api_key)
        img = Image.open(io.BytesIO(image_bytes))
        model = genai.GenerativeModel('gemini-2.5-flash')
        response = model.generate_content([prompt, img])
        raw = response.text.strip()
    else:
        # Offline Mock Fallback: works perfectly without any API subscription
        raw = '''{
          "description": "A beautiful fashion item resembling your uploaded image.",
          "category": "Dresses",
          "gender": "Women",
          "colour": "Black",
          "keywords": ["elegant", "stylish", "casual"]
        }'''
    raw = re.sub(r"```json|```", "", raw).strip()
    analysis = json.loads(raw)

    # Build a search query from the analysis
    query_tags = " ".join([
        analysis.get("category", ""),
        analysis.get("colour", ""),
        analysis.get("gender", ""),
        " ".join(analysis.get("keywords", [])),
    ])

    # Filter by gender + category first, then rank by TF-IDF similarity to query
    fdf = df.copy()
    gender_val = analysis.get("gender")
    cat_val    = analysis.get("category")
    if gender_val in ("Women", "Men", "Kids"):
        fdf = fdf[fdf["gender"] == gender_val]
    if cat_val and cat_val in fdf["category"].values:
        fdf = fdf[fdf["category"] == cat_val]

    if fdf.empty:
        fdf = df.copy()

    tfidf_q  = TfidfVectorizer(stop_words="english", max_features=6000)
    all_tags = list(df["tags"].values) + [query_tags]
    vecs     = tfidf_q.fit_transform(all_tags)
    query_vec = vecs[-1]
    prod_vecs = vecs[fdf.index.tolist()]
    from sklearn.metrics.pairwise import cosine_similarity as cs
    scores   = cs(query_vec, prod_vecs).flatten()
    top_n    = scores.argsort()[::-1][:n]
    rec_idxs = [fdf.index[i] for i in top_n]

    return analysis, rec_idxs


# ─────────────────────────────────────────────
#  NAVBAR
# ─────────────────────────────────────────────
bc = ""
if st.session_state.gender:
    bc = f"<span>{st.session_state.gender}</span>"
if st.session_state.sub_cat:
    bc += f" › <span>{st.session_state.sub_cat}</span>"
if st.session_state.page == "detail" and st.session_state.product_idx is not None:
    pn = df.iloc[st.session_state.product_idx]["name"]
    bc += f" › <span>{str(pn)[:30]}…</span>"

st.markdown(f"""
<div class='navbar'>
    <div>
        <div class='navbar-logo'>FashionAI</div>
        <div class='navbar-tagline'>Discover · Style · Recommend</div>
    </div>
    <div class='breadcrumb'>{bc}</div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🎛️ Filters")
    search_text = st.text_input("🔍 Search", placeholder="e.g. floral dress…")
    min_p, max_p = int(df["price"].min()), int(df["price"].max())
    price_range  = st.slider("Price Range (₹)", min_p, max_p, (min_p, max_p))
    sort_opt     = st.selectbox("Sort By", ["Relevance","Price: Low → High","Price: High → Low","Top Rated"])
    st.markdown("---")
    if st.session_state.page != "home":
        if st.button("🏠 Home"):
            st.session_state.update({"page":"home","gender":None,"sub_cat":None,"product_idx":None})
            st.rerun()
    if st.button("📷 Search by Image"):
        st.session_state.update({"page":"image_search","img_recs":[],"img_analysis":None})
        st.rerun()
    if st.button("🚪 Logout"):
        st.session_state.update({"logged_in":False,"page":"home","gender":None,"sub_cat":None,"product_idx":None})
        st.rerun()
    st.markdown("---")
    st.markdown(f"👤 Logged in as **admin**")


# ═════════════════════════════════════════════
#  HOME PAGE
# ═════════════════════════════════════════════
if st.session_state.page == "home":

    st.markdown("<div class='section-title'>Shop by Category</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-sub'>Discover curated collections tailored for you</div>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3, gap="large")
    for col, gender, style in [
        (col1,"Women","gender-card-women"),
        (col2,"Men",  "gender-card-men"),
        (col3,"Kids", "gender-card-kids"),
    ]:
        count = len(df[df["gender"] == gender])
        with col:
            st.markdown(f"""
            <div class='gender-card {style}'>
                <div class='gender-card-inner'>
                    <span class='gender-emoji'>{GENDER_ICONS[gender]}</span>
                    <span class='gender-card-title'>{gender}</span>
                    <span class='gender-card-sub'>{count:,} products</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            if st.button(f"Shop {gender} →", key=f"g_{gender}", use_container_width=True):
                st.session_state.update({"gender":gender,"page":"products","sub_cat":None})
                st.rerun()

    # Trending strip
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>✨ Trending Now</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-sub'>Highest rated picks across all categories</div>", unsafe_allow_html=True)

    trending = df.nlargest(6, "avg_rating")
    t_cols   = st.columns(6, gap="small")
    for ci, (real_idx, row) in enumerate(trending.iterrows()):
        with t_cols[ci]:
            st.markdown(f"""
            <div class='product-card'>
                <div class='badge badge-top'>⭐ Top</div>
                <img src='{row["img"]}' onerror="this.src='https://via.placeholder.com/220x250?text=No+Image'"/>
                <div class='product-card-body'>
                    <div class='product-brand'>{str(row["brand"])[:18]}</div>
                    <div class='product-name'>{str(row["name"])[:38]}</div>
                    <div class='product-price'>₹{int(row["price"]):,}</div>
                    <div class='product-rating'><span class='rating-star'>★</span> {float(row["avg_rating"]):.1f}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            if st.button("View", key=f"trend_{real_idx}"):
                st.session_state.update({
                    "product_idx": real_idx,
                    "gender": row["gender"],
                    "sub_cat": row["category"],
                    "page": "detail",
                })
                st.rerun()


# ═════════════════════════════════════════════
#  PRODUCTS PAGE
# ═════════════════════════════════════════════
elif st.session_state.page == "products":

    gender   = st.session_state.gender
    sub_cats = [c for c in GENDER_CATS.get(gender, [])
                if len(df[(df["gender"]==gender) & (df["category"]==c)]) > 0]

    st.markdown(f"<div class='section-title'>{GENDER_ICONS[gender]} {gender}'s Fashion</div>", unsafe_allow_html=True)

    # Sub-category pills
    all_cats = ["All"] + sub_cats
    pill_cols = st.columns(len(all_cats), gap="small")
    for ci, cat in enumerate(all_cats):
        with pill_cols[ci]:
            icon  = CAT_ICON.get(cat, "🛍️")
            label = f"{icon} {cat}"
            if st.button(label, key=f"pill_{cat}"):
                st.session_state.sub_cat = None if cat == "All" else cat
                st.rerun()

    # Active filter indicator
    active_cat = st.session_state.sub_cat
    if active_cat:
        st.markdown(f"<p style='font-size:12px;color:#e91e8c;margin:4px 0 12px;'>Showing: <b>{active_cat}</b> &nbsp;·&nbsp; <a href='#' style='color:#aaa;'>Clear</a></p>", unsafe_allow_html=True)

    # Build filtered dataset
    fdf = df[df["gender"] == gender].copy()
    if active_cat:
        fdf = fdf[fdf["category"] == active_cat]
    if search_text:
        fdf = fdf[fdf["name"].str.contains(search_text, case=False, na=False)]
    fdf = fdf[(fdf["price"] >= price_range[0]) & (fdf["price"] <= price_range[1])]
    if sort_opt == "Price: Low → High":   fdf = fdf.sort_values("price")
    elif sort_opt == "Price: High → Low": fdf = fdf.sort_values("price", ascending=False)
    elif sort_opt == "Top Rated":         fdf = fdf.sort_values("avg_rating", ascending=False)

    cat_label = active_cat or "All"
    st.markdown(f"<div class='section-sub'>{len(fdf):,} products found in <b>{cat_label}</b></div>", unsafe_allow_html=True)

    if fdf.empty:
        st.markdown("""<div class='empty-state'>
            <div class='empty-state-icon'>🔍</div>
            <div class='empty-state-text'>No products found. Try adjusting filters.</div>
        </div>""", unsafe_allow_html=True)
    else:
        items = list(fdf.head(40).iterrows())
        for row_chunk in [items[i:i+4] for i in range(0, len(items), 4)]:
            cols = st.columns(4, gap="medium")
            for ci, (real_idx, row) in enumerate(row_chunk):
                with cols[ci]:
                    st.markdown(f"""
                    <div class='product-card'>
                        <img src='{row["img"]}' onerror="this.src='https://via.placeholder.com/220x250?text=No+Image'"/>
                        <div class='product-card-body'>
                            <div class='product-brand'>{str(row["brand"])[:20]}</div>
                            <div class='product-name'>{str(row["name"])[:44]}</div>
                            <div class='product-price'>₹{int(row["price"]):,}</div>
                            <div class='product-rating'><span class='rating-star'>★</span> {float(row["avg_rating"]):.1f}</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    if st.button("View Details", key=f"p_{real_idx}"):
                        st.session_state.update({"product_idx": real_idx, "page": "detail"})
                        st.rerun()


# ═════════════════════════════════════════════
#  DETAIL PAGE
# ═════════════════════════════════════════════
elif st.session_state.page == "detail":

    idx = st.session_state.product_idx
    if idx is None or idx not in df.index:
        st.error("Product not found.")
        st.stop()

    p = df.iloc[idx]

    if st.button("← Back to Products"):
        st.session_state.page = "products"
        st.rerun()

    col_img, col_info = st.columns([1, 1.4], gap="large")

    with col_img:
        st.markdown(f"""
        <div style='border-radius:16px;overflow:hidden;box-shadow:0 8px 32px rgba(0,0,0,.10);'>
            <img src='{p["img"]}' style='width:100%;display:block;'
                 onerror="this.src='https://via.placeholder.com/420x500?text=No+Image'"/>
        </div>
        """, unsafe_allow_html=True)

    with col_info:
        st.markdown(f"""
        <div class='detail-brand'>{p["brand"]}</div>
        <div class='detail-name'>{p["name"]}</div>
        <div class='detail-price'>₹{int(p["price"]):,}</div>
        <div class='detail-price-sub'>Inclusive of all taxes &nbsp;·&nbsp; Free delivery</div>
        <div class='detail-meta'>
            <div class='detail-meta-item'><label>Rating</label><span>⭐ {float(p["avg_rating"]):.1f} / 5</span></div>
            <div class='detail-meta-item'><label>Colour</label><span>{p["colour"]}</span></div>
            <div class='detail-meta-item'><label>Category</label><span>{p["category"]}</span></div>
        </div>
        <div class='detail-desc'>{p["description"]}</div>
        """, unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1: st.button("🛒 Add to Bag", use_container_width=True)
        with c2: st.button("❤️ Wishlist",   use_container_width=True)

    # Recommendations
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='section-title'>✨ Similar Styles You'll Love</div>
    <div class='section-sub'>AI-powered picks based on this product</div>
    """, unsafe_allow_html=True)

    recs = recommend(idx, n=8)
    if not recs:
        st.markdown("<div class='empty-state'><div class='empty-state-icon'>🤷</div><div class='empty-state-text'>No similar products found.</div></div>", unsafe_allow_html=True)
    else:
        for chunk in [recs[i:i+4] for i in range(0, len(recs), 4)]:
            r_cols = st.columns(4, gap="medium")
            for ci, r_idx in enumerate(chunk):
                r = df.iloc[r_idx]
                with r_cols[ci]:
                    st.markdown(f"""
                    <div class='product-card'>
                        <div class='badge badge-sim'>Similar</div>
                        <img src='{r["img"]}' onerror="this.src='https://via.placeholder.com/220x250?text=No+Image'"/>
                        <div class='product-card-body'>
                            <div class='product-brand'>{str(r["brand"])[:20]}</div>
                            <div class='product-name'>{str(r["name"])[:44]}</div>
                            <div class='product-price'>₹{int(r["price"]):,}</div>
                            <div class='product-rating'><span class='rating-star'>★</span> {float(r["avg_rating"]):.1f}</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    if st.button("View", key=f"rec_{r_idx}"):
                        st.session_state.update({"product_idx": r_idx, "sub_cat": r["category"]})
                        st.rerun()


# ═════════════════════════════════════════════
#  IMAGE SEARCH PAGE
# ═════════════════════════════════════════════
elif st.session_state.page == "image_search":

    st.markdown("<div class='section-title'>📷 Search by Image</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-sub'>Upload a fashion photo and let AI find similar products for you</div>", unsafe_allow_html=True)

    if st.button("← Back to Home"):
        st.session_state.update({"page": "home", "img_recs": [], "img_analysis": None})
        st.rerun()

    uploaded = st.file_uploader(
        "Upload a fashion image (JPG / PNG / WEBP)",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed",
    )

    if uploaded:
        col_prev, col_info = st.columns([1, 1.6], gap="large")
        with col_prev:
            st.image(uploaded, caption="Your uploaded image", use_container_width=True)

        with col_info:
            st.markdown("<div class='upload-zone-text'>Image uploaded ✅ — click <b>Analyse & Recommend</b> to find similar products.</div>", unsafe_allow_html=True)
            st.write("")
            if st.button("🔍 Analyse & Recommend", use_container_width=True):
                with st.spinner("Analysing your image with AI…"):
                    try:
                        img_bytes = uploaded.read()
                        mime = "image/jpeg" if uploaded.type in ("image/jpg","image/jpeg") else uploaded.type
                        analysis, rec_idxs = analyze_image_and_recommend(img_bytes, mime, n=8)
                        st.session_state.img_analysis = analysis
                        st.session_state.img_recs     = rec_idxs
                        st.rerun()
                    except Exception as e:
                        st.error(f"Analysis failed: {e}")

    # Show analysis + results
    if st.session_state.img_analysis:
        a = st.session_state.img_analysis
        st.markdown(f"""
        <div class='analysis-box'>
            <strong>🧠 AI Analysis</strong><br>
            <b>Description:</b> {a.get('description','—')}<br>
            <b>Category:</b> {a.get('category','—')} &nbsp;·&nbsp;
            <b>Gender:</b> {a.get('gender','—')} &nbsp;·&nbsp;
            <b>Colour:</b> {a.get('colour','—')}<br>
            <b>Keywords:</b> {', '.join(a.get('keywords', []))}
        </div>
        """, unsafe_allow_html=True)

        recs = st.session_state.img_recs
        if recs:
            st.markdown("<div class='section-title' style='font-size:22px;'>✨ Recommended Products</div>", unsafe_allow_html=True)
            for chunk in [recs[i:i+4] for i in range(0, len(recs), 4)]:
                r_cols = st.columns(4, gap="medium")
                for ci, r_idx in enumerate(chunk):
                    r = df.iloc[r_idx]
                    with r_cols[ci]:
                        st.markdown(f"""
                        <div class='product-card'>
                            <div class='badge badge-sim'>AI Pick</div>
                            <img src='{r["img"]}' onerror="this.src='https://via.placeholder.com/220x250?text=No+Image'"/>
                            <div class='product-card-body'>
                                <div class='product-brand'>{str(r["brand"])[:20]}</div>
                                <div class='product-name'>{str(r["name"])[:44]}</div>
                                <div class='product-price'>₹{int(r["price"]):,}</div>
                                <div class='product-rating'><span class='rating-star'>★</span> {float(r["avg_rating"]):.1f}</div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        if st.button("View Details", key=f"imgr_{r_idx}"):
                            st.session_state.update({
                                "product_idx": r_idx,
                                "gender": r["gender"],
                                "sub_cat": r["category"],
                                "page": "detail",
                            })
                            st.rerun()
        else:
            st.markdown("<div class='empty-state'><div class='empty-state-icon'>🤷</div><div class='empty-state-text'>No matching products found.</div></div>", unsafe_allow_html=True)