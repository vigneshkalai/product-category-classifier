#!/usr/bin/env python3

import sys
sys.path.insert(0, './deep-learning-models')

from keras.models import model_from_json
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing import image as image_utils
from imagenet_utils import preprocess_input
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image

import tensorflow as tf
import keras.backend.tensorflow_backend as ktf

import csv
import hashlib
import nltk
import numpy as np
import os
import pickle
import re
import requests
import shutil
import string
import unicodedata
import urllib

from urllib.parse import urlparse

from nltk import word_tokenize
from nltk.corpus import stopwords

import socket
socket.setdefaulttimeout(30)

##
## What you would like to predict?
##
PRODUCT = { 
  "url": "https://www.macys.com/shop/product/michael-michael-kors-studio-mercer-medium-bonded-leather-messenger?ID=3042290&CategoryID=46011",
  "title":"Mercer Medium Bonded-Leather Messenger",
  "main_image":"https://px.twotap.com/unsafe/https%3A//slimages.macysassets.com/is/image/MCY/products/3/optimized/3980693_fpx.tif%3Fop_sharpen%3D1%26hei%3D434%26%24filterlrg%24",
  "description":"A structured silhouette and elegantly rolled handles really elevate the look of an on-the-go crossbody crafted in bonded pebble leather. A gleaming padlock charm adds signature shine that can only be MICHAEL Michael Kors. <ul> <li>Leather; lining: polyester</li> <li>Imported</li> <li>Medium sized bag; 8 1/2&#x94;W X 7 1/2&#x94;H X 4&#x94;D</li> <li>Interior features center zip divider pocket with 1 zip pocket and 1 slip pocket inside</li> <li>4 1/4&#x94; Handle Drop ; Adjustable Strap: 18-20 1/2&quot; </li> <li>Open top</li> <li>Exterior features gold-tone hardware</li> <li>Silhouette is based off 5&apos;9&quot; model</li> <li>WEB ID: 3042290</li> </ul>"
}
USE_IMAGE = True


##
## Prediction part.
##

MAX_SEQUENCE_LENGTH = 150

# Load the model.
full_model_json_file = open('./models/text_and_image.json', 'r')
full_model_json = full_model_json_file.read()
full_model_json_file.close()

full_model = model_from_json(full_model_json)

# Load the weights.
full_model.load_weights('./models/text_and_image_weights.h5')

# Load the tokenizer.
tokenizer = pickle.load(open('./models/tokenizer.p', 'rb'), encoding='latin1')

# Load the label_index.
labels_index = pickle.load(open('./models/label_index.p', 'rb'), encoding='latin1')
reverse_labels_index  = {v: k for k, v in labels_index.items()}

print('Model loaded!')

# NTLK
nltk.download('punkt')
nltk.download('stopwords')
stop = stopwords.words('english')

punctuation_tbl = dict.fromkeys(i for i in range(sys.maxunicode) if unicodedata.category(chr(i)).startswith('P'))


def md5(string):
  m = hashlib.md5()
  m.update(string.encode('utf-8'))
  return m.hexdigest()


def ensure_image_exists(product_url, product_image_url):
  parsed_product_url = urlparse(product_url)
  product_host = parsed_product_url.hostname
  product_host_md5 = md5(product_host)
  product_md5 = md5(product_image_url)

  product_host_dir = "./ml-data/images/%s" % product_host_md5
  product_image_path = "./ml-data/images/%s/img_%s.png" % (product_host_md5, product_md5)

  if os.path.exists(product_image_path):
    return product_image_path

  if not os.path.isdir(product_host_dir):
     os.makedirs(product_host_dir)

  try:
    headers = {
      'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'
    }

    response = requests.get(product_image_url, stream=True, headers=headers)

    with open(product_image_path, 'wb') as out_file:
      shutil.copyfileobj(response.raw, out_file)

    del response
  except Exception as e:
    print(e)
    return None

  return product_image_path

def clean_string(s):
  s = s.translate(punctuation_tbl)

  # Remove common things.
  s = re.sub(r'\bshop\b', '', s)
  s = re.sub(r'\bhome\b', '', s)
  s = re.sub(r'[&,+]', ' ', s)

  # Remove stop words.
  s = ' '.join([i for i in word_tokenize(s.lower()) if i not in stop])

  return s


def predict(product, use_image):
  product_url = product.get('url')
  product_image_url = product.get('main_image')

  # Prepare the image.
  parsed_image = np.zeros((299, 299, 3))

  if use_image and product_url and product_image_url:
    product_image_path = ensure_image_exists(product_url, product_image_url)

    if product_image_path:
      try:
        img = image_utils.load_img(product_image_path, target_size=(299, 299))
        x = image_utils.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
      
        parsed_image = x[0]
      except Exception as e:
        print(e)

  # Prepare the text.
  product_categories = product.get('categories')
  if product_categories:
      product_categories = ' '.join(product_categories)


  product_title = product.get('title')

  if product_categories:
    parsed_text = clean_string(product_title + ' ' + product_categories)
  else:
    parsed_text = clean_string(product_title)

  parsed_text = [ parsed_text ]
  parsed_text = tokenizer.texts_to_sequences(parsed_text)
  parsed_text = pad_sequences(parsed_text, maxlen=MAX_SEQUENCE_LENGTH)

  parsed_image = np.expand_dims(parsed_image, axis=0)

  prediction = full_model.predict([ parsed_text, parsed_image ])[0]

  prediction_indices = np.argpartition(prediction, -4)[-4:]
  prediction_indices = prediction_indices[np.argsort(prediction[prediction_indices])]

  results = []

  for idx in prediction_indices:
    results.append({ 'category': reverse_labels_index[idx], 'value': float(prediction[idx]) })

  return results

print(predict(PRODUCT, USE_IMAGE))
