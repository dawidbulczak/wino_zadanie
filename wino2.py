# wino2.py
# Streamlit app – analiza jakości wina + dopasowanie do restauracji
# Zawiera SHAP (Explainable AI)

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import shap

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# --------------------------------------------------
# KONFIGURACJA STRONY
# --------------------------------------------------
st.set_page_config(
    page_title="Wine Analytics for Restaurants",
    layout="wide"
)

st.title("🍷 Wine Analytics for Restaurants")
st.markdown(
    """
    Aplikacja wspierająca **sprzedaż win do restauracji**:
    - eksploracja jakości win,
    - dopasowanie win do kuchni i dań,
    - wsparcie decyzji: **gdzie najlepiej sprzedać dane wino z magazynu**.
    """
)

# --------------------------------------------------
# WCZYTYWANIE DANYCH
# --------------------------------------------------
@st.cache_data
def load_data():
    wine = pd.read_csv("winequality-red.csv")
    pairings = pd.read_csv("wine_food_pairings.csv")
    return wine, pairings

wine_df, pairing_df = load_data()

# --------------------------------------------------
# FUNKCJA EDA
# --------------------------------------------------
def basic_eda(df: pd.DataFrame):
    st.write("**Podgląd danych (head):**")
    st.dataframe(df.head())

    col1, col2, col3 = st.columns(3)
    col1.metric("Liczba wierszy", df.shape[0])
    col2.metric("Liczba kolumn", df.shape[1])
    col3.metric("Duplikaty", df.duplicated().sum())

    st.write("**Typy danych:**")
    st.dataframe(df.dtypes.astype(str))

    st.write("**Brakujące wartości:**")
    na = df.isna().sum()
    st.dataframe(na[na > 0] if na.sum() > 0 else pd.DataFrame({"Braki": [0]}))

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
module = st.sidebar.radio(
    "Wybierz sekcję:",
    [
        "Wine Quality – eksploracja",
        "Wine Quality – rozkłady, porównania i SHAP",
        "Food Pairings – restauracje i sprzedaż"
    ]
)

# ==================================================
# 1️⃣ WINE QUALITY – EKSPLORACJA + FILTRY
# ==================================================
if module == "Wine Quality – eksploracja":

    st.header("📊 Wine Quality – podstawowa eksploracja danych")
    basic_eda(wine_df)

    st.subheader("🔎 Filtrowanie i szybkie wnioski")

    q_min, q_max = int(wine_df.quality.min()), int(wine_df.quality.max())
    q_range = st.slider("Zakres quality", q_min, q_max, (q_min, q_max))

    feature = st.selectbox(
        "Dodatkowa cecha",
        [c for c in wine_df.columns if c != "quality"]
    )

    f_min, f_max = float(wine_df[feature].min()), float(wine_df[feature].max())
    f_range = st.slider(f"Zakres {feature}", f_min, f_max, (f_min, f_max))

    filt = wine_df[
        wine_df.quality.between(*q_range) &
        wine_df[feature].between(*f_range)
    ]

    st.write(f"Pozostało rekordów: **{len(filt)}**")
    st.dataframe(filt)

    st.write("**Statystyki:**")
    st.write(filt[[feature, "quality"]].agg(["mean", "median", "min", "max"]))

# ==================================================
# 2️⃣ ROZKŁADY + PORÓWNANIA + 3D + SHAP
# ==================================================
elif module == "Wine Quality – rozkłady, porównania i SHAP":

    st.header("📈 Rozkłady i porównania jakości wina")

    feature = st.selectbox(
        "Wybierz cechę",
        [c for c in wine_df.columns if c != "quality"]
    )

    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots()
        sns.histplot(wine_df[feature], bins=30, ax=ax)
        ax.set_title(f"Histogram – {feature}")
        st.pyplot(fig)

    with col2:
        fig, ax = plt.subplots()
        sns.boxplot(x=wine_df[feature], ax=ax)
        ax.set_title(f"Boxplot – {feature}")
        st.pyplot(fig)

    st.subheader("⚖️ Porównanie grup jakości")

    mode = st.radio(
        "Tryb porównania",
        ["quality ≤ X vs > X", "quality = A vs B"]
    )

    if mode == "quality ≤ X vs > X":
        x = st.slider("X", int(wine_df.quality.min()), int(wine_df.quality.max()), 6)
        g1 = wine_df[wine_df.quality <= x][feature]
        g2 = wine_df[wine_df.quality > x][feature]
        labels = [f"≤ {x}", f"> {x}"]
    else:
        q_vals = sorted(wine_df.quality.unique())
        a, b = st.selectbox("A", q_vals), st.selectbox("B", q_vals, index=1)
        g1 = wine_df[wine_df.quality == a][feature]
        g2 = wine_df[wine_df.quality == b][feature]
        labels = [str(a), str(b)]

    fig, ax = plt.subplots()
    ax.boxplot([g1, g2], labels=labels)
    ax.set_title(f"{feature} vs quality")
    st.pyplot(fig)

    st.subheader("🌐 Wykres 3D – profil chemiczny wina")

    fig3d = px.scatter_3d(
        wine_df,
        x="alcohol",
        y="volatile acidity",
        z="sulphates",
        color="quality",
        title="Profil chemiczny wina vs jakość"
    )
    st.plotly_chart(fig3d, use_container_width=True)

    # ---------------- SHAP ----------------
    st.subheader("🤖 Wyjaśnienie modelu jakości (SHAP)")

    X = wine_df.drop("quality", axis=1)
    y = wine_df["quality"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )
    model.fit(X_train, y_train)

    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_test)

    st.markdown(
        """
        **Interpretacja:**
        SHAP pokazuje, które cechy chemiczne wina
        najbardziej wpływają na ocenę jakości.
        """
    )

    fig, ax = plt.subplots()
    shap.plots.bar(shap_values, show=False)
    st.pyplot(fig)

# ==================================================
# 3️⃣ FOOD PAIRINGS – RESTAURACJE + SPRZEDAŻ
# ==================================================
else:

    st.header("🍽️ Food Pairings – sprzedaż win do restauracji")
    basic_eda(pairing_df)

    st.subheader("🔎 Filtrowanie pod restaurację")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        wine_type = st.multiselect("wine_type", sorted(pairing_df.wine_type.unique()))
    with c2:
        food_cat = st.multiselect("food_category", sorted(pairing_df.food_category.unique()))
    with c3:
        cuisine = st.multiselect("cuisine", sorted(pairing_df.cuisine.unique()))
    with c4:
        pq_min = int(pairing_df.pairing_quality.min())
        pq_max = int(pairing_df.pairing_quality.max())
        min_pq = st.slider("Minimalna pairing_quality", pq_min, pq_max, pq_min)

    filt = pairing_df.copy()
    if wine_type:
        filt = filt[filt.wine_type.isin(wine_type)]
    if food_cat:
        filt = filt[filt.food_category.isin(food_cat)]
    if cuisine:
        filt = filt[filt.cuisine.isin(cuisine)]
    filt = filt[filt.pairing_quality >= min_pq]

    st.write(f"Pozostało dopasowań: **{len(filt)}**")
    st.dataframe(filt)

    st.write("**Statystyki dopasowań:**")
    st.write(filt[["pairing_quality"]].agg(["mean", "median", "min", "max"]))

    # ---------------- SPRZEDAŻ MAGAZYNOWA ----------------
    st.subheader("📦 Gdzie najlepiej sprzedać dane wino?")

    selected_wine = st.selectbox(
        "Wino zalegające w magazynie",
        sorted(pairing_df.wine_type.unique())
    )

    wine_focus = pairing_df[pairing_df.wine_type == selected_wine]

    sales = (
        wine_focus
        .groupby("cuisine")
        .agg(
            avg_quality=("pairing_quality", "mean"),
            count=("pairing_quality", "count")
        )
        .reset_index()
    )

    sales["sales_score"] = sales["avg_quality"] * sales["count"]

    fig = px.bar(
        sales.sort_values("sales_score", ascending=False),
        x="cuisine",
        y="sales_score",
        title=f"Potencjał sprzedaży wina: {selected_wine}",
        labels={
            "cuisine": "Profil restauracji",
            "sales_score": "Potencjał sprzedaży"
        }
    )

    st.plotly_chart(fig, use_container_width=True)

    st.success(
        "🎯 **Wniosek biznesowy:** "
        "Wino warto kierować do restauracji, gdzie ma najwyższy potencjał sprzedaży."
    )
