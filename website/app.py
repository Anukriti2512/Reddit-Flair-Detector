#prediction function
import sklearn
import pickle
import praw
import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer

from flask import Flask, render_template, request, url_for

app = Flask(__name__)

@app.route("/")
@app.route("/home")
def home():
    return render_template("home.html")
    

@app.route('/predict',methods=['POST'])
def predict():
    loaded_model = pickle.load(open('model.sav', 'rb'))
    if request.method == 'POST':
        url = request.form['url']

        prediction = detect_flair(url, loaded_model)
        return render_template("home.html",prediction=prediction)


def getStemmedText(text):
    
    text = text.lower()
    post = text
    #post = text.split()
    #post = ""
    #for word in new_text:
    #   if d.check(word)==True:
    #        post = post + " " + word
    tokenizer = RegexpTokenizer(r'\w+')
    stopwords_eng = set(stopwords.words('english'))
    stemmer = PorterStemmer()
    
    tokens = tokenizer.tokenize(post)
    new_tokens = []
    stemmed_tokens = []
    
    for token in tokens:
        if token not in stopwords_eng:
            new_tokens.append(token)
    
    for token in new_tokens:
        stemmed_tokens.append(stemmer.stem(token))
        
    cleaned_text = ' '.join(stemmed_tokens)
    return cleaned_text
    
def detect_flair(url,loaded_model):

    reddit = praw.Reddit(client_id='8kRfdrPdpbJe9Q', client_secret='qeSfFi6kizAdurAFL790ZhQL_uM', user_agent='Anukriti Jain', username = 'ameowkriti', password = 'Jain@123')
    ans = "Invalid URL!"
    if(len(url)==0):
        return ans

    else:
        submission = reddit.submission(url=url)

        data = {}

        data['title'] = submission.title
        data['url'] = submission.url
        data['body'] = submission.selftext

        submission.comments.replace_more(limit=None)
        comment = ''
        #count = 1
        for top_level_comment in submission.comments:
            #if(count<=20):
            comment = comment + ' ' + top_level_comment.body
            #count+=1
        data["comment"] = comment
      
        data['title'] = getStemmedText(data['title'])
        data['comment'] = getStemmedText(data['comment'])
        data['body'] = getStemmedText(data['body'])
        data['url'] = getStemmedText(data['url'])
        
        data['combine'] = data['title'] + data['comment'] + data['url'] + data['body']
        ans = loaded_model.predict([data['combine']])

        return ans.item()


if __name__ == "__main__":
    loaded_model = pickle.load(open('model.sav', 'rb'))
    app.run(debug=True)

