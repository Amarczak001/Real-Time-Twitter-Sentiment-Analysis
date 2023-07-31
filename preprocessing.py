from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re

def clean_text(df):
    df['text'] = df['text'].apply(lambda x: re.sub(r'https?://\S+|www\.\S+', '', x))  # Remove URLs
    df['text'] = df['text'].apply(lambda x: re.sub(r'<.*?>', '', x))  # Remove HTML tags
    df['text'] = df['text'].apply(lambda x: re.sub(r'[^a-zA-Z\s]', '', x))  # Remove non-alphabetic characters
    df['text'] = df['text'].apply(lambda x: x.lower())  # Convert to lowercase
    df['text'] = df['text'].apply(lambda x: re.sub(r'\s+', ' ', x).strip())  # Remove extra spaces and strip

    return df

def tokenize_text(texts, num_words=10000):
    tokenizer = Tokenizer(num_words=num_words, oov_token='<OOV>')
    tokenizer.fit_on_texts(texts)
    return tokenizer

def text_to_sequences(tokenizer, texts, max_sequence_length=100):
    sequences = tokenizer.texts_to_sequences(texts)
    padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length, truncating='post', padding='post')
    return padded_sequences
