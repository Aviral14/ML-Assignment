DELIMITER = [" ", "\n", "\t", ",", "!", "."]
STOPWORDS = [
    "i",
    "me",
    "my",
    "myself",
    "we",
    "our",
    "ours",
    "ourselves",
    "you",
    "your",
    "yours",
    "yourself",
    "yourselves",
    "he",
    "him",
    "his",
    "himself",
    "she",
    "her",
    "hers",
    "herself",
    "it",
    "its",
    "itself",
    "they",
    "them",
    "their",
    "theirs",
    "themselves",
    "what",
    "which",
    "who",
    "whom",
    "this",
    "that",
    "these",
    "those",
    "am",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "have",
    "has",
    "had",
    "having",
    "do",
    "does",
    "did",
    "doing",
    "a",
    "an",
    "the",
    "and",
    "but",
    "if",
    "or",
    "because",
    "as",
    "until",
    "while",
    "of",
    "at",
    "by",
    "for",
    "with",
    "about",
    "against",
    "between",
    "into",
    "through",
    "during",
    "before",
    "after",
    "above",
    "below",
    "to",
    "from",
    "up",
    "down",
    "in",
    "out",
    "on",
    "off",
    "over",
    "under",
    "again",
    "further",
    "then",
    "once",
    "here",
    "there",
    "when",
    "where",
    "why",
    "how",
    "all",
    "any",
    "both",
    "each",
    "few",
    "more",
    "most",
    "other",
    "some",
    "such",
    "no",
    "nor",
    "not",
    "only",
    "own",
    "same",
    "so",
    "than",
    "too",
    "very",
    "s",
    "t",
    "can",
    "will",
    "just",
    "don",
    "should",
    "now",
]


def tokenize(text):
    """
    Given an input string `text`, returns a list of lowercase words containing
    representing the tokenized version of `text`, excluding punctuation
    like comma, question mark, whitespace, etc.
    """
    assert type(text) == str
    token = str()
    tokenized_words = list()
    for char in text:
        if not char in DELIMITER:
            token += char
        else:
            if token:
                tokenized_words.append(token)
            token = str()
    if token:
        tokenized_words.append(token)
    return tokenized_words


def remove_stopwords(token_list):
    """
    Given a list of english words, removes stopwords from it ( such as he her was etc.).
    """
    filtered_tokens = [w for w in token_list if not w in STOPWORDS]
    return filtered_tokens


def process_string(query):
    """
    Given a query string, performs stemming, normalization, tokenization and removal of stopwords and returns
    a list.
    """
    assert type(query) == str
    disposable = query.strip()
    disposable = tokenize(disposable)
    disposable = remove_stopwords(disposable)
    return disposable
