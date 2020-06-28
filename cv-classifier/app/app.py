"""
1-page web-app that predicts job class based on input CV
"""

import os
import pickle
from flask import Flask, request, render_template
import docx
import tensorflow as tf
import numpy as np
from keras.preprocessing.sequence import pad_sequences

import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

APP = Flask(__name__, template_folder='templates')
PORT = int(os.environ.get("PORT", 5000))

# set paths to upload folder
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
APP.config['TEXT_UPLOADS'] = os.path.join(APP_ROOT, 'static')

# function to read .docx files
def get_text(filename):

    """function to get text from input docx

        param filename: filename of input document
        returns ' '.join(full_text): list item with text
        returns doc: docx object
    """
    doc = docx.Document(filename)
    full_text = []
    for para in doc.paragraphs:
        full_text.append(para.text.replace('\t', ' '))
    return ' '.join(full_text), doc

# initialize model and tokenizer
NEW_MODEL = tf.keras.models.load_model('assets/model1.h5')
with open('assets/tokenizer.pickle', 'rb') as handle:
    TOKENIZER = pickle.load(handle)

# labels
LABELS = ['data analyst', 'event manager', 'marketing manager', 'sales manager']

# parameters
STOPWORDS = set(stopwords.words('english'))
MAX_LENGTH = 5000
TRUNC_TYPE = 'post'
PADDING_TYPE = 'post'

@APP.route("/display", methods=["GET", "POST"])
def display():

    """
    Function to classify input doc, return prediction en load one-page app
    """

    if request.method == "POST":

        # read and process file
        cv_file = request.files['input_file']
        filename = cv_file.filename
        file_path = os.path.join(APP.config["TEXT_UPLOADS"], filename)

        cv_text, doc = get_text(cv_file)
        doc.save(file_path)

        job_list = []

        for word in STOPWORDS:
            token = ' ' + word + ' '
            job = cv_text.replace(token, ' ')
            job = job.replace(' ', ' ')

        job_list.append(job)

        seq = TOKENIZER.texts_to_sequences(job_list)
        padded = pad_sequences(seq, maxlen=MAX_LENGTH, padding=PADDING_TYPE, truncating=TRUNC_TYPE)
        pred = NEW_MODEL.predict(padded)

        prediction = LABELS[np.argmax(pred)]

        #display prediction
        return render_template("upload.html",
                               text_path=filename,
                               prediction="You're a: "+prediction)
    return render_template("upload.html")

if __name__ == '__main__':

    APP.run(host='0.0.0.0', debug=True, port=PORT)
