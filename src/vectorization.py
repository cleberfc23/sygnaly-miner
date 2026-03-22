from sklearn.feature_extraction.text import CountVectorizer


def build_bow(dataframe):
    vectorizer = CountVectorizer()
    X_vectorized = vectorizer.fit_transform(dataframe)
    return X_vectorized, vectorizer
