import re
import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

def preprocess(sentence):
    tokens = word_tokenize(sentence.lower())
    return [stemmer.stem(word) for word in tokens if word.isalnum()]

def lemmatize_words(words):
    return [lemmatizer.lemmatize(word.lower()) for word in words if word.isalnum()]

def bag_of_words(tokenized_sentence, all_words):
    sentence_words = [stemmer.stem(word) for word in tokenized_sentence]
    bag = [1 if word in sentence_words else 0 for word in all_words]
    return bag

def safe_int(value):
    try:
        if isinstance(value, str):
            value = re.sub(r"[^0-9.]", "", value)
        return int(float(value))
    except:
        return 0

def extract_price_range(text):
    text = text.lower().replace(",", "")

    def to_lkr(amount_str):
        if "lakh" in amount_str or "lakhs" in amount_str:
            return int(float(re.sub(r"[^\d.]", "", amount_str)) * 100000)
        elif "m" in amount_str or "million" in amount_str:
            return int(float(re.sub(r"[^\d.]", "", amount_str)) * 1000000)
        elif "k" in amount_str:
            return int(float(re.sub(r"[^\d.]", "", amount_str)) * 1000)
        else:
            return int(float(re.sub(r"[^\d.]", "", amount_str)))

    # Match patterns like 'between 3M and 5M', 'from 4 million to 7 million'
    match = re.search(r'(?:between|from)?\s*(\d+[\.\d]*\s*(?:m|million|lakhs|lakh|k)?)\s*(?:to|and|-)\s*(\d+[\.\d]*\s*(?:m|million|lakhs|lakh|k)?)', text)
    if match:
        return to_lkr(match.group(1)), to_lkr(match.group(2))

    # Match 'under 3 million', 'below 5m', 'less than 2M'
    match = re.search(r'(?:under|below|less than)\s*(\d+[\.\d]*\s*(?:m|million|lakhs|lakh|k)?)', text)
    if match:
        return 0, to_lkr(match.group(1))

    # Match 'over 2 million', 'above 5m', 'more than 1.5M'
    match = re.search(r'(?:over|above|more than)\s*(\d+[\.\d]*\s*(?:m|million|lakhs|lakh|k)?)', text)
    if match:
        return to_lkr(match.group(1)), 999999999

    # Match simple value: 'cars for 5M', 'budget 3 million'
    match = re.search(r'(\d+[\.\d]*\s*(?:m|million|lakhs|lakh|k)?)', text)
    if match:
        amt = to_lkr(match.group(1))
        return 0, amt

    return None
