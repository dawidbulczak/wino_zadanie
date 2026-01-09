import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

# -------------------------------------------------
# KONFIGURACJA
# -------------------------------------------------
st.set_page_config(
    page_title="Wine Analytics & Food Pairings",
    layout="wide"
)

st.title("🍷 Wine Analytics – wsparcie sprzedaży wina do restauracji")
st.markdown(
    """
    Aplikacja wspiera **sprzedawcę wina**, który chce:
    - analizować jakość win,
    - dopasowywać wina do **konkretnych restauracji i typów dań**,
    - podejmować szybkie decyzje sprzedażowe.
    """
)

# -------------------------------------------------
# WCZYTANIE DANYCH
# -------------------------------------------------
@st.cache_data
def load_data():
    wine_quality = pd.read_csv("winequality-red.csv")
    pairings = pd.read_csv("wine_food_pairings.csv")
    return wine_quality, pairings


wine_df, pairings_df = load_data()

# -------------------------------------------------
# FUNKCJA EDA
# -------------------------------------------------
def basic_eda(df):
    col1, col2 = st.columns(2)

    with col1:
        st.write("**Podgląd danych:**")
        st.dataframe(df.head())
        st.write("**Liczba wierszy / kolumn:**", df.shape)
        st.write("**Duplikaty:**", df.duplicated().sum())

    with col2:
        st.write("**Typy danych:**")
        st.write(df.dtypes)
        st.write("**Brakujące wartości:**")
        na = df.isna().sum()
        st.write(na[na > 0] if na.sum() > 0 else "Brak braków danych ✅")

# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------
module = st.sidebar.radio(
    "Wybierz moduł:",
    [
        "Analiza jakości wina",
        "Parowanie wina z jedzeniem"
    ]
)

# =================================================
# MODUŁ 1 – ANALIZA JAKOŚCI WINA
# =================================================
if module == "Analiza jakości wina":

    st.header("📊 Analiza jakości czerwonych win")

    with st.expander("📌 Podstawowa eksploracja danych"):
        basic_eda(wine_df)

    # ---------------------------------------------
    # FILTROWANIE
    # ---------------------------------------------
    st.subheader("🔎 Filtrowanie")

    min_q, max_q = int(wine_df.quality.min()), int(wine_df.quality.max())
    q_range = st.slider(
        "Zakres quality:",
        min_q, max_q, (min_q, max_q)
    )

    feature = st.selectbox(
        "Dodatkowa cecha:",
        [c for c in wine_df.columns if c != "quality"]
    )

    f_min, f_max = float(wine_df[feature].min()), float(wine_df[feature].max())
    f_range = st.slider(
        f"Zakres {feature}:",
        f_min, f_max, (f_min, f_max)
    )

    filtered = wine_df[
        (wine_df.quality.between(q_range[0], q_range[1])) &
        (wine_df[feature].between(f_range[0], f_range[1]))
    ]

    st.write(f"📌 Rekordów po filtrach: **{filtered.shape[0]}**")
    st.dataframe(filtered)

    # ---------------------------------------------
    # STATYSTYKI
    # ---------------------------------------------
    st.subheader("📈 Szybkie statystyki")

    col1, col2, col3 = st.columns(3)
    col1.metric("Średnia", round(filtered[feature].mean(), 3))
    col2.metric("Mediana", round(filtered[feature].median(), 3))
    col3.metric("Min / Max", f"{filtered[feature].min():.2f} / {filtered[feature].max():.2f}")

    # ---------------------------------------------
    # ROZKŁADY
    # ---------------------------------------------
    st.subheader("📊 Rozkład cechy")

    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots()
        ax.hist(wine_df[feature], bins=30)
        ax.set_title(f"Histogram – {feature}")
        st.pyplot(fig)

    with col2:
        fig, ax = plt.subplots()
        ax.boxplot(wine_df[feature], vert=False)
        ax.set_title(f"Boxplot – {feature}")
        st.pyplot(fig)

    # ---------------------------------------------
    # PORÓWNANIE JAKOŚCI
    # ---------------------------------------------
    st.subheader("⚖️ Porównanie grup jakości")

    mode = st.radio(
        "Tryb porównania:",
        ["quality ≤ X vs > X", "quality = A vs B"]
    )

    if mode == "quality ≤ X vs > X":
        x = st.slider("X:", min_q, max_q, 6)
        g1 = wine_df[wine_df.quality <= x][feature]
        g2 = wine_df[wine_df.quality > x][feature]
        labels = [f"≤ {x}", f"> {x}"]
    else:
        a, b = st.multiselect(
            "Wybierz dwie jakości:",
            sorted(wine_df.quality.unique()),
            default=[5, 6]
        )
        if len([a, b]) == 2:
            g1 = wine_df[wine_df.quality == a][feature]
            g2 = wine_df[wine_df.quality == b][feature]
            labels = [str(a), str(b)]

    fig, ax = plt.subplots()
    ax.boxplot([g1, g2], labels=labels)
    ax.set_title("Porównanie rozkładów")
    st.pyplot(fig)

    # ---------------------------------------------
    # WYKRES 3D
    # ---------------------------------------------
    st.subheader("🌐 Wykres 3D")

    x3 = st.selectbox("Oś X", wine_df.columns, index=0)
    y3 = st.selectbox("Oś Y", wine_df.columns, index=1)
    z3 = st.selectbox("Oś Z", wine_df.columns, index=wine_df.columns.get_loc("alcohol"))

    fig3d = px.scatter_3d(
        wine_df,
        x=x3,
        y=y3,
        z=z3,
        color="quality",
        opacity=0.7
    )

    st.plotly_chart(fig3d, use_container_width=True)

# =================================================
# MODUŁ 2 – PAROWANIE WINA Z JEDZENIEM
# =================================================
else:

    st.header("🍽️ Parowanie wina z jedzeniem (sprzedaż do restauracji)")

    with st.expander("📌 Podstawowa eksploracja danych"):
        basic_eda(pairings_df)

    # ---------------------------------------------
    # FILTROWANIE
    # ---------------------------------------------
    st.subheader("🔎 Filtry")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        wine_type = st.multiselect(
            "Typ wina:",
            sorted(pairings_df.wine_type.unique())
        )
    with col2:
        food_cat = st.multiselect(
            "Kategoria dania:",
            sorted(pairings_df.food_category.unique())
        )
    with col3:
        cuisine = st.multiselect(
            "Kuchnia:",
            sorted(pairings_df.cuisine.unique())
        )
    with col4:
        min_q = int(pairings_df.pairing_quality.min())
        max_q = int(pairings_df.pairing_quality.max())
        pq = st.slider("Minimalna jakość:", min_q, max_q, min_q)

    filt = pairings_df.copy()

    if wine_type:
        filt = filt[filt.wine_type.isin(wine_type)]
    if food_cat:
        filt = filt[filt.food_category.isin(food_cat)]
    if cuisine:
        filt = filt[filt.cuisine.isin(cuisine)]

    filt = filt[filt.pairing_quality >= pq]

    st.write(f"📌 Dopasowań: **{filt.shape[0]}**")
    st.dataframe(
        filt.sort_values("pairing_quality", ascending=False)
    )

    # ---------------------------------------------
    # STATYSTYKI
    # ---------------------------------------------
    st.subheader("📊 Szybkie wnioski")

    col1, col2, col3 = st.columns(3)
    col1.metric("Średnia", round(filt.pairing_quality.mean(), 2))
    col2.metric("Mediana", filt.pairing_quality.median())
    col3.metric("Max", filt.pairing_quality.max())
