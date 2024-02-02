import flask
from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np

import tensorflow as tf
from tensorflow.keras.layers import Layer

import tensorflow_hub as hub
 


# Define the KerasLayer custom class
# class KerasLayer(tf.keras.layers.Layer):
#     def __init__(self, hub_url, **kwargs):
#         super(KerasLayer, self).__init__(**kwargs)
#         self.hub_url = hub_url

#     def build(self, input_shape):
#         self.bert_layer = hub.KerasLayer(self.hub_url, trainable=True)
#         super(KerasLayer, self).build(input_shape)

#     def call(self, inputs, **kwargs):
#         return self.bert_layer(inputs, **kwargs)

#     def get_config(self):
#         config = super(KerasLayer, self).get_config()
#         config['hub_url'] = self.hub_url
#         return config

#     @classmethod
#     def from_config(cls, config):
#         return cls(hub_url=config['hub_url'])


# Load the pre-trained model and any necessary preprocessing functions
# Replace these with the actual functions to load your model and perform preprocessing
# Example
model = tf.keras.models.load_model(r'C:\Users\akash\Python_DL\PROJECT_DL\email_classification_model.h5')

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify():
    # Get the input sentence from the form
    sentence = request.form['sentence']

    # Perform any necessary preprocessing on the sentence
    # Example:
    #  preprocessed_sentence = preprocess(sentence)

    # Assuming your model accepts a batch of sentences, create a batch with just one sentence
    # Example:
    # sentence_batch = [preprocessed_sentence]

    # Make predictions using your model
    # Example:
    predictions = model.predict(sentence)

    # If your model outputs probabilities, you can use a threshold to decide spam/ham
    # Example:
    threshold = 0.5
    result = 'Spam' if predictions[0][0] >= threshold else 'Ham'

    # For demonstration purposes, let's assume we have a random result
    result = np.random.choice(['Spam', 'Ham'])

    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
