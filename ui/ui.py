import streamlit as st
from wordcloud import WordCloud
import numpy as np
import plotly.express as px
import nltk
from nltk import tokenize
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD



def build_word_cloud(all_words):
    word_cloud = WordCloud(
        max_words=100,
        background_color="white",
        colormap="Reds",
        width=1200,
        height=700,
        random_state=42
    ).generate(all_words)

    wordcloud_array = np.array(word_cloud)

    fig = px.imshow(wordcloud_array)
    fig.update_layout(
        font=dict(
            family="Roboto",
            size=14,
            color="#2c3e50"
        ),
        title=dict(
            text="<b>Word Cloud — Overview</b>\n",
            x=0.5,
            xanchor='center',
            font=dict(size=20)
        ),
        coloraxis_showscale=False,
        margin=dict(l=0, r=0, t=40, b=0),
        plot_bgcolor="#f8f9fa",
        paper_bgcolor="#f8f9fa"

    )
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)

    return fig


def build_top_words(all_words):
    token_space = tokenize.WhitespaceTokenizer()
    token_statment = token_space.tokenize(all_words)
    frequency_words = nltk.FreqDist(token_statment)
    df_frequency = pd.DataFrame({'Word': list(frequency_words.keys()),
                                 'Frequency': list(frequency_words.values())})
    df_top10_words = df_frequency.nlargest(columns='Frequency', n=10)

    fig = px.bar(
        df_top10_words.sort_values("Frequency"),
        x="Frequency",
        y="Word",
        orientation="h",
        text="Frequency",
        color_discrete_sequence=["#C8102E"]
    )

    fig.update_layout(
        title=dict(
            text="<b>Top 10 Most Frequent Terms</b>",
            x=0.5,
            xanchor="center",
            font=dict(size=22)
        ),
        font=dict(
            family="Roboto",
            size=14,
            color="#2c3e50"
        ),
        xaxis_title="Frequency",
        yaxis_title=None,  # remove redundância
        template="plotly_white",
        margin=dict(l=40, r=20, t=70, b=40)
    )

    fig.update_traces(
        textposition="outside",
        textfont=dict(size=12),
        cliponaxis=False  # evita cortar os números
    )

    fig.update_xaxes(
        showgrid=True,
        gridcolor="#e5e7eb",
        zeroline=False
    )

    fig.update_yaxes(
        showgrid=False
    )

    # fig.show()
    return fig


def build_cluster(dataframe, X):
    # pca = PCA(n_components=2)
    # X_reduced = pca.fit_transform(X.toarray())

    svd = TruncatedSVD(n_components=2, random_state=48)
    X_reduced = svd.fit_transform(X)

    # sample_size = min(1000, X.shape[0])
    # indices = np.random.choice(X.shape[0], sample_size, replace=False)

    # X_sample = X[indices].toarray()
    # df_sample = dataframe.iloc[indices]

    # pca = PCA(n_components=2)
    # X_reduced = pca.fit_transform(X_sample)

    fig = px.scatter(
        x=X_reduced[:, 0],
        y=X_reduced[:, 1],
        color=dataframe["cluster"].astype(str),
        title="<b>Cluster Visualization (PCA Projection)</b>",
        labels={
            "x": "Principal Component 1",
            "y": "Principal Component 2",
            "color": "Cluster"
        },
        opacity=0.75
    )

    fig.update_layout(
        font=dict(
            family="Roboto",
            size=14,
            color="#2c3e50"
        ),
        title=dict(
            x=0.5,
            xanchor="center",
            font=dict(size=22)
        ),
        template="plotly_white",
        margin=dict(l=40, r=40, t=70, b=40),
        legend_title_text="Cluster"
    )

    fig.update_traces(
        marker=dict(
            size=6,
            line=dict(width=0.5, color="white")
        )
    )

    fig.update_xaxes(
        showgrid=True,
        gridcolor="#e5e7eb",
        zeroline=False
    )

    fig.update_yaxes(
        showgrid=True,
        gridcolor="#e5e7eb",
        zeroline=False
    )

    return fig


# --------------------------------------------------------------------------------------------------------------------
def build_figures(dataframe, X):
    all_words = ' '.join([text for text in dataframe['content']])
    fig_cluster = build_cluster(dataframe, X)
    fig_top_words = build_top_words(all_words)
    fig_word_cloud = build_word_cloud(all_words)
    return fig_word_cloud, fig_top_words, fig_cluster
    # return fig_word_cloud, fig_top_words, []
# --------------------------------------------------------------------------------------------------------------------


def render_header():
    st.title("Sygnały Miner")
    st.caption("Baseline NLP clustering with CountVectorizer + KMeans")


def render_dataset_info(df):
    st.subheader("General Dataset Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Documents", len(df))

    with col2:
        st.metric("Clusters", df["cluster"].nunique()
                  if "cluster" in df.columns else "N/A")

    with col3:
        avg_length = int(df["content"].str.len().mean()
                         ) if "content" in df.columns else 0
        st.metric("Avg. Text Length", avg_length)

    st.markdown(
        """
        **Baseline setup**
        - Text representation: CountVectorizer
        - Clustering algorithm: KMeans
        - Number of clusters: 8 (empirical choice)
        - Purpose: establish a simple and interpretable clustering baseline
        """
    )


def render_metrics(inertia, silhouette, silhouette_std):
    st.subheader("Baseline Metrics")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Inertia", f"{inertia:,.2f}")

    with col2:
        st.metric("Silhouette Score", f"{silhouette:.4f}")

    st.warning(
        "This is a baseline evaluation. Inertia reflects cluster compactness, "
        "while silhouette score indicates cluster separation."
    )


def render_charts(fig_cluster_dist, fig_pca, fig_top_words):
    st.subheader("Visualizations")

    st.plotly_chart(fig_cluster_dist, use_container_width=True)
    st.divider()
    st.plotly_chart(fig_top_words, use_container_width=True)
    st.divider()
    st.plotly_chart(fig_pca, use_container_width=True)
    st.divider()


@st.cache_resource
def build_ui(df, inertia, silhouette_mean, silhouette_std, fig_cluster_dist, fig_pca, fig_top_words):
    render_header()
    render_dataset_info(df)
    st.divider()
    render_metrics(inertia, silhouette_mean, silhouette_std)
    st.divider()
    render_charts(fig_cluster_dist, fig_pca, fig_top_words)
