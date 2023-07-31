from data_loading import load_data
from preprocessing import clean_text, tokenize_text, text_to_sequences
from model import create_model

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

    # Create the model
    vocab_size = len(tokenizer.word_index) + 1
    model = create_model(input_dim=vocab_size, output_dim=100, input_length=100)
    model.summary()
