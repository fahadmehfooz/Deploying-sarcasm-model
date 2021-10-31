from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle

# load the model from disk
logit = pickle.load(open('logit.pkl', 'rb'))
vectorizer=pickle.load(open('vectorizer.pkl','rb'))
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():

    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        my_prediction = logit.predict(TfidfVectorizer(vocabulary=vectorizer.get_feature_names()).fit_transform(data).toarray())

    return render_template('result.html',prediction = my_prediction)



if __name__ == '__main__':
    app.run(debug=False)