from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import nltk

def download_nltk_deps():
    """
    Downloads NLTK English language datasets. This function needs to be called only once in a lifetime.
    """
    nltk.download("punkt")
    nltk.download("stopwords")


def tokenize(text):
    """
    Given an input string `text`, returns a list of lowercase words containing
    representing the tokenized version of `text`, excluding punctuation
    like comma, question mark, whitespace, etc.
    """
    assert type(text) == str
    tokenized_words = word_tokenize(text.lower())

    # Remove tokens like comma, question mark and other punctuation.
    filtered_words = list(filter(lambda word: len(word) > 1, tokenized_words))
    return filtered_words

def remove_stopwords(token_list):
    """
    Given a list of english words, removes stopwords from it ( such as he her was etc.).
    """
    stop_words = set(stopwords.words("english"))
    filtered_tokens = [w for w in token_list if not w in stop_words]
    return filtered_tokens


def stem(token_list):
    """
    Given an list of tokenized words, returns their stemmed forms, using NLTK
    For example, `["playing"]` becoms `["play"]`.
    """

    assert type(token_list) == list
    ps = PorterStemmer()
    stemmed_words = list(map(lambda word: ps.stem(word), token_list))
    result = " ".join(map(str, stemmed_words))
    return result

def process_string(query):
    """
    Given a query string, performs stemming, normalization, tokenization and removal of stopwords and returns
    a list.
    """
    assert type(query) == str
    disposable = query.strip()
    disposable = tokenize(disposable)
    disposable = remove_stopwords(disposable)
    disposable = stem(disposable)
    return disposable.split(" ")
    
