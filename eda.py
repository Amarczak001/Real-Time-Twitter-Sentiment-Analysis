import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from data_loading import load_data

def plot_class_distribution(df):
    print(df['target'].value_counts())
    sns.countplot(x='target', data=df)
    plt.show()

def plot_tweet_length(df):
    df['length'] = df['text'].apply(len)
    df['length'].plot(kind='hist', bins=50)
    plt.show()

if __name__ == "__main__":
    file_path = './training.1600000.processed.noemoticon.csv'
    df = load_data(file_path, nrows_per_class=5000)  # Load 5,000 rows of each class

    plot_class_distribution(df)
    plot_tweet_length(df)
