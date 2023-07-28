import pandas as pd

def load_data(file_path, nrows_per_class=None):
    # Define column names
    col_names = ['target', 'id', 'date', 'flag', 'user', 'text']
    
    # Load negative tweets
    df_negative = pd.read_csv(file_path, encoding='ISO-8859-1', names=col_names, nrows=nrows_per_class)

    # Load positive tweets
    df_positive = pd.read_csv(file_path, encoding='ISO-8859-1', names=col_names, skiprows=800000, nrows=nrows_per_class)

    # Concatenate the two dataframes
    df = pd.concat([df_negative, df_positive])

    # Map target labels to 'negative' and 'positive'
    df['target'] = df['target'].map({0: 'negative', 4: 'positive'})

    return df

if __name__ == "__main__":
    file_path = './training.1600000.processed.noemoticon.csv'
    df = load_data(file_path, nrows_per_class=5000)  # Load 5,000 rows of each class
    print(df.head())
