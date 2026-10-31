"""Fold-local text vectorizers. Fit only on training rows."""
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack


def build_features(train_text, other_text=None, *, ngram_char=(3, 5), ngram_word=(1, 2), max_features=None):
    """Fit char and word TF-IDF on train_text, transform optional other_text."""
    chars = TfidfVectorizer(analyzer="char", ngram_range=ngram_char, min_df=2, max_features=max_features, sublinear_tf=True)
    words = TfidfVectorizer(analyzer="word", ngram_range=ngram_word, min_df=2, max_features=max_features, strip_accents="unicode", sublinear_tf=True)
    train_text = [str(t) for t in train_text]
    x_train = hstack((chars.fit_transform(train_text), words.fit_transform(train_text)), format="csr")
    if other_text is None:
        return x_train, (chars, words)
    other_text = [str(t) for t in other_text]
    x_other = hstack((chars.transform(other_text), words.transform(other_text)), format="csr")
    return x_train, x_other, (chars, words)
