import tweepy

def get_tweets(keyword, num_tweets):
    # Replace with your own keys
    consumer_key = '9fwjFwkCh6TDD7Ohjix3g3Upy'
    consumer_secret = 'ZwelIvhbbdfdkPIZgMBNhPvha5c7bUkKCzv3sOE2AjBDZ220wH'
    access_token = '1600280133751001089-QxXbSkCMty7L7I9bdX3c4HafAt0L42'
    access_token_secret = 'Jxr4XWw6iuHK8yplnF3xzRtFOHBeAVg2vDeABjqWoFba5'

    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)

    api = tweepy.API(auth, wait_on_rate_limit=True)

    # Cursor handles pagination and lets us retrieve as many tweets as we want
    tweets = tweepy.Cursor(api.search_tweets, q=keyword, lang='en', tweet_mode='extended').items(num_tweets)
    
    # Collect the tweets in a list and return
    tweet_list = [tweet.full_text for tweet in tweets]
    
    return tweet_list

if __name__ == "__main__":
    tweets = get_tweets('machine learning', 100)  # Fetch 100 tweets about machine learning
    print(tweets[:5])  # Print the first 5 tweets
