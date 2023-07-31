from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from preprocessing import clean_text, tokenize_text, text_to_sequences, predict_sentiment
import pickle

app = Flask(__name__)

# Load the tokenizer and the model
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

model = load_model('sentiment_analysis_model.h5')

@app.route('/', methods=['GET', 'POST'])
def index():
    sentiment = ''
    if request.method == 'POST':
        tweet = request.form.get('tweet')
        sentiment = predict_sentiment(model, tokenizer, tweet)
    return render_template('index.html', sentiment=sentiment)

if __name__ == '__main__':
    app.run(debug=True)
