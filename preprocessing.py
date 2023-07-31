from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from data_loading import load_data
from model import create_model
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

if __name__ == "__main__":
    file_path = './training.1600000.processed.noemoticon.csv'
    df = load_data(file_path, nrows_per_class=5000)  # Load a subset of the data (5,000 rows of each class)

    # Clean the text
    df = clean_text(df)
    print(df.head())

    # Tokenize the text and convert to sequences
    tokenizer = tokenize_text(df['text'])
    sequences = text_to_sequences(tokenizer, df['text'])
    print(sequences[0])

    # Convert labels to binary format
    labels = df['target'].map({'negative': 0, 'positive': 1})
    labels = to_categorical(labels)

    # Split the data into training and testing sets
    sequences_train, sequences_test, labels_train, labels_test = train_test_split(sequences, labels, test_size=0.2)

    # Create the model
    vocab_size = len(tokenizer.word_index) + 1
    model = create_model(input_dim=vocab_size, output_dim=100, input_length=100)
    model.summary()

    # Train the model
    model.fit(sequences_train, labels_train, epochs=3, validation_data=(sequences_test, labels_test))
