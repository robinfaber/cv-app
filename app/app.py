from flask import Flask, request, render_template

import os
import docx
import pickle

import tensorflow as tf
# from tensorflow import keras
import numpy as np

# from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

app = Flask(__name__, template_folder='templates')

# set paths to upload folder
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
app.config['TEXT_UPLOADS'] = os.path.join(APP_ROOT, 'static')

# function to read .docx files
def getText(filename):
    doc = docx.Document(filename)
    fullText = []
    for para in doc.paragraphs:
        fullText.append(para.text.replace('\t', ' '))
    return ' '.join(fullText), doc

# initialize model and tokenizer
new_model = tf.keras.models.load_model('assets/model1.h5')
with open('assets/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# labels
labels = ['data scientist', 'sales manager', 'front-office manager', 'front-end developer']

# parameters
STOPWORDS = set(stopwords.words('english'))
max_length = 1000
trunc_type = 'post'
padding_type = 'post'




@app.route("/display",methods=["GET","POST"])
def display():

    if request.method == "POST":

        # read and process file
        cv_file = request.files['input_file']
        filename = cv_file.filename
        file_path = os.path.join(app.config["TEXT_UPLOADS"], filename)

        cv_text, doc = getText(cv_file)
        doc.save(file_path)

        job_list = []

        for word in STOPWORDS:
            token = ' ' + word + ' '
            job = cv_text.replace(token, ' ')
            job = job.replace(' ', ' ')

        job_list.append(job)

        # classify cv
        # max_length = 1000
        # trunc_type = 'post'
        # padding_type = 'post'

        seq = tokenizer.texts_to_sequences(job_list)
        padded = pad_sequences(seq, maxlen=max_length, padding=padding_type, truncating=trunc_type)
        pred = new_model.predict(padded)
        
        prediction = labels[np.argmax(pred)]

        #display prediction
        return render_template("upload.html", text_path = filename, prediction = 'Prediction: '+prediction)
    return render_template("upload.html")

if __name__ == '__main__':

    app.run(host='127.0.0.1', debug=False, threaded=False, port=8000)