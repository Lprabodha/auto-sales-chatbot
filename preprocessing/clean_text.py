import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet as wn
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

lemmatizer = WordNetLemmatizer()

stop_words = set(stopwords.words('english'))

ignore_words = set([
    "the", "a", "is", "are", "am", "can", "please", "would", "could", "should",
    "show", "tell", "me", "i", "you", "about", "of", "and", "how", "to", "for", 
    "in", "find", "help", "want", "need", "get", "let", "give", "know"
])

def get_synonyms(word):
    synonyms = set()
    for syn in wn.synsets(word):
        for lemma in syn.lemmas():
            name = lemma.name().replace("_", " ").lower()
            if name != word:
                synonyms.add(name)
    return synonyms

def preprocess(text, expand_synonyms=True):
    tokens = word_tokenize(text.lower()) 
    clean_tokens = [
        lemmatizer.lemmatize(w)
        for w in tokens
        if w.isalpha() and w not in stop_words and w not in ignore_words
    ]

    if expand_synonyms:
        expanded = set(clean_tokens)
        for token in clean_tokens:
            expanded.update(get_synonyms(token))
        return ' '.join(expanded)

    return ' '.join(clean_tokens)
