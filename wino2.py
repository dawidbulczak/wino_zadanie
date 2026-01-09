# wino2.py
# Streamlit app do analizy jakości win i dopasowania do restauracji

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# --------------------------------------------------
# Konfiguracja strony
# --------------------------------------------------
st.set_page_config(
    page_title="Wine Analytics for Restaurants",
    layout="wide"
)

st.title("🍷 Wine Analytics for Restaurants")
st.markdown(
    "Aplikacja wspierająca **sprzedaż i dobór win** do restauracji na podstawie "
    "jakości wina oraz dopasowania do typu kuchni i dań."
)

# --------------------------------------------------
# Wczytywanie danych
# --------------------------------------------------
@st.cache_data
def load_data():
    wine = pd.read_csv("winequality-red.csv")
    pairings = pd.read_csv("wine_food_pairings.csv")
    return wine, pairings

wine_df, pairing_df = load_data()

# --------------------------------------------------
# Funkcja eksploracji danych (wspólna)
# --------------------------------------------------
def basic_eda(df: pd.DataFrame):
    st.write("**Podgląd danych (head):**")
    st.dataframe(df.head())

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Liczba wierszy", df.shape[0])
    with col2:
        st.metric("Liczba kolumn", df.shape[1])
    with col3:
        st.metric("Duplikaty", df.duplicated().sum())

    st.write("**Typy danych:**")
    st.dataframe(df.dtypes.astype(str), use_container_width=True)

    st.write("**Brakujące wartości:**")
    na = df.isna().sum()
    st.dataframe(na[na > 0] if na.sum() > 0 else pd.DataFrame({"Braki": [0]}))

# --------------------------------------------------
# Sidebar
# --------------------------------------------------
module = st.sidebar.radio(
    "Wybierz sekcję",
    [
        "Wine Quality – eksploracja",
        "Wine Quality – rozkłady i porównania",
        "Food Pairings – eksploracja i filtracja"
    ]
)

# ==================================================
# 1. WINE QUALITY – EDA + FILTROWANIE
# ==================================================
if module == "Wine Quality – eksploracja":
    st.header("📊 Wine Quality – podstawowa eksploracja")
    basic_eda(wine_df)

    st.subheader("🔎 Filtrowanie i szybkie wnioski")

    q_min, q_max = int(wine_df.quality.min()), int(wine_df.quality.max())
    q_range = st.slider("Zakres quality", q_min, q_max, (q_min, q_max))

    feature = st.selectbox(
        "Dodatkowa cecha do filtrowania",
        [c for c in wine_df.columns if c != "quality"]
    )

    f_min, f_max = float(wine_df[feature].min()), float(wine_df[feature].max())
    f_range = st.slider(
        f"Zakres {feature}",
        f_min, f_max, (f_min, f_max)
    )

    filt = wine_df[
        (wine_df.quality.between(*q_range)) &
        (wine_df[feature].between(*f_range))
    ]

    st.write(f"Pozostało rekordów: **{len(filt)}**")
    st.dataframe(filt)

    st.write("**Statystyki:**")
    st.write(filt[[feature, "quality"]].agg(["mean", "median", "min", "max"]))

# ==================================================
# 2. ROZKŁADY, PORÓWNANIA + 3D
# ==================================================
elif module == "Wine Quality – rozkłady i porównania":
    st.header("📈 Rozkłady i porównania cech jakości")

    feature = st.selectbox(
        "Wybierz cechę",
        [c for c in wine_df.columns if c != "quality"]
    )

    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots()
        sns.histplot(wine_df[feature], bins=30, ax=ax)
        ax.set_title(f"Histogram: {feature}")
        st.pyplot(fig)

    with col2:
        fig, ax = plt.subplots()
        sns.boxplot(x=wine_df[feature], ax=ax)
        ax.set_title(f"Boxplot: {feature}")
        st.pyplot(fig)

    st.subheader("⚖️ Porównanie grup jakości")

    mode = st.radio(
        "Tryb porównania",
        ["quality ≤ X vs > X", "quality = A vs B"]
    )

    if mode == "quality ≤ X vs > X":
        x = st.slider("X", int(wine_df.quality.min()), int(wine_df.quality.max()), 5)
        g1 = wine_df[wine_df.quality <= x]
        g2 = wine_df[wine_df.quality > x]
        labels = [f"≤ {x}", f"> {x}"]
    else:
        q_vals = sorted(wine_df.quality.unique())
        a, b = st.selectbox("A", q_vals), st.selectbox("B", q_vals, index=1)
        g1 = wine_df[wine_df.quality == a]
        g2 = wine_df[wine_df.quality == b]
        labels = [str(a), str(b)]

    fig, ax = plt.subplots()
    ax.boxplot([g1[feature], g2[feature]], labels=labels)
    ax.set_title(f"{feature} vs quality")
    st.pyplot(fig)

    st.subheader("🧊 Wykres 3D (sprzedażowy insight)")
    fig3d = px.scatter_3d(
        wine_df,
        x="alcohol",
        y="volatile acidity",
        z="sulphates",
        color="quality",
        title="Profil chemiczny wina vs jakość"
    )
    st.plotly_chart(fig3d, use_container_width=True)

# ==================================================
# 3. FOOD PAIRINGS – RESTAURACJE
# ==================================================
elif module == "Food Pairings – eksploracja i filtracja":
    st.header("🍽️ Food Pairings – dobór win do restauracji")
    basic_eda(pairing_df)

    st.subheader("🔎 Filtrowanie pod klienta (restaurację)")

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

    st.write("**Statystyki jakości dopasowań:**")
    st.write(
        filt[["pairing_quality"]]
        .agg(["mean", "median", "min", "max"])
    )

    st.success(
        "🎯 **Zastosowanie biznesowe:**\n"
        "Na podstawie filtrów możesz przygotować **spersonalizowaną ofertę win** "
        "dla konkretnej restauracji, kuchni lub typu dań."
    )
