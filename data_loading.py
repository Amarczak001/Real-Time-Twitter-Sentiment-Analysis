import pandas as pd

def load_data(file_path, nrows=None):
    # Define column names
    col_names = ['target', 'id', 'date', 'flag', 'user', 'text']
    
    # Load data from CSV
    df = pd.read_csv(file_path, encoding='ISO-8859-1', names=col_names, nrows=nrows)

    # Map target labels to 'negative' and 'positive'
    df['target'] = df['target'].map({0: 'negative', 4: 'positive'})

    return df

if __name__ == "__main__":
    file_path = './training.1600000.processed.noemoticon.csv'
    df = load_data(file_path, nrows=10000)  # Load a subset of the data (10,000 rows)
    print(df.head())
