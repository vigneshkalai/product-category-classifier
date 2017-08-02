#!/usr/bin/env python3

import sys
sys.path.insert(0, './deep-learning-models')

from sklearn.cross_validation import train_test_split
from sklearn.utils import class_weight
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image as image_utils
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D, GlobalAveragePooling2D, merge, Input, Dropout
from keras.layers.core import Dense, Activation, Flatten
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import TimeDistributed
from keras.models import Sequential, Model
from keras.utils import np_utils
from keras.utils import to_categorical

from nltk import word_tokenize
from nltk.corpus import stopwords

import csv
import hashlib
import json
import nltk
import numpy as np
import pandas as pd
import pickle
import os
import re
import requests
import string
import sys
import unicodedata
import urllib

from urllib.parse import urlparse

from imagenet_utils import decode_predictions
from imagenet_utils import preprocess_input

MAX_NB_WORDS = 40000
MAX_SEQUENCE_LENGTH = 150
EMBEDDING_DIM = 100


if len(sys.argv) == 1 or (sys.argv[1] != 'dev' and sys.argv[1] != 'prod'):
  print("Run script as: python train.py (dev or prod)")
  sys.exit(0)

if sys.argv[1] == 'dev':
  print("Running in dev mode.")
  GENERATOR_BATCH_SIZE = 25
  MAX_TRAINING = 100
else:
  print("Running in prod mode.")
  GENERATOR_BATCH_SIZE = 100
  MAX_TRAINING = None


# Make sure we have the nltk thing.
nltk.download('punkt')
nltk.download('stopwords')

punctuation_tbl = dict.fromkeys(i for i in range(sys.maxunicode) if unicodedata.category(chr(i)).startswith('P'))

ENGLISH_STOPWORDS = stopwords.words('english')

# Load Glove (Word2Vec).
embeddings_index = {}
glove_data = './ml-data/glove.6B.100d.txt'
f = open(glove_data)
for line in f:
    values = line.split()
    word = values[0]
    value = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = value
f.close()

print(f"Loaded {len(embeddings_index)} word vectors.")

##
## Helper funcs.
##
def clean_string(s):
  s = s.translate(punctuation_tbl)

  # Remove common things.
  s = re.sub(r'\bshop\b', '', s)
  s = re.sub(r'\bhome\b', '', s)
  s = re.sub(r'[&,+]', ' ', s)

  # Remove stop words.
  s = ' '.join([i for i in word_tokenize(s.lower()) if i not in ENGLISH_STOPWORDS])

  return s

def md5(string):
  m = hashlib.md5()
  m.update(string.encode('utf-8'))
  return m.hexdigest()

##
## Fetch CSV data
##

# How many lines in the csv?
df = pd.read_csv('./ml-data/training_data.csv')
lines_count = df.shape[0]

print(f"Found {lines_count} lines in the csv file.")


# How many categories do we actually have?
tt_categories_in_dataset = []

for index, row in df.iterrows():
    if len(row) != 6:
        continue

    tt_categories = row[5].split('~~')
    for slice_idx in range(len(tt_categories)):
        category_part = '~~'.join(tt_categories[0:(slice_idx+1)])

        if category_part not in tt_categories_in_dataset:
            tt_categories_in_dataset.append(category_part)

print(f"Found {len(tt_categories_in_dataset)} labels (categories).")

##
## Fetch labels
##
url = 'https://api.twotap.com/v1.0/product/taxonomy'

resp = requests.get(url=url, params=None)
label_taxonomy = json.loads(resp.text)['taxonomy']
label_index = {}
for label in label_taxonomy:
    if label in tt_categories_in_dataset:
        label_index[label] = len(label_index)
    
label_length = len(label_index)

print(f"Label length is {label_length}.")

# Allocate arrays.
data = np.zeros((lines_count, 2)).astype('U1000')
labels = np.zeros((lines_count, (label_length)))


# For class_weights.
flat_labels = []

print("Starting to read CSV file.")
count = 0
for chunk in pd.read_csv('./ml-data/training_data.csv', chunksize=1000, encoding='utf-8'):
    if MAX_TRAINING and count > MAX_TRAINING:
        break

    for row in chunk.itertuples():
        if len(row) != 7:
            continue

        if label_index.get(row[6]) == None:
            continue

        tt_categories = row[6]

        if len(tt_categories.split('~~')) == 1:
          continue

        # Prepare the image.
        try:
          parsed_product_url = urlparse(row[1])
          product_host = parsed_product_url.hostname
          product_host_md5 = md5(product_host)
          product_host_dir = "./ml-data/images/%s" % product_host_md5
        except Exception as e:
          print(e)
          continue

        image_path = 'fake_img.png'
        images = row[4]
        if images and isinstance(images, str):
          image_url = images.split('###')[0]
          product_md5 = md5(image_url)
          image_path = "./ml-data/images/%s/img_%s.png" % (product_host_md5, product_md5)

        # Prepare the text.
        name = row[2]

        if not isinstance(name, str):
          print(f"{name} is not a string.")
          continue

        text_entry = clean_string(name)

        data[count][0] = text_entry
        data[count][1] = image_path

        # Labels
        label_map = np.zeros(label_length)

        # We set only one category.
        idx = label_index[tt_categories]
        label_map[idx] = 1
        flat_labels.append(idx)

        labels[count] = label_map

        count += 1

    print(f"Read {count} lines...")

data = data[0:count]
labels = labels[0:count, :]

print(f"Added {count} lines.")

data_train, data_test, labels_train, labels_test = train_test_split(data, labels, test_size=0.2, random_state=42)


##
## Tokenizer
##

print("Tokenizing data.")
all_text_data = data.transpose()[0]

tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(all_text_data)

word_index = tokenizer.word_index

##
## Class weights. Our data is skewed.
##
class_weights = class_weight.compute_class_weight('balanced', np.unique(flat_labels), flat_labels)

print(f"Class weights: {class_weights}")


##
## Generator
##
def product_generator(input_data, input_labels):
  iteration = 0

  while True:
    batch_data = []
    batch_labels = []
    
    if iteration >= (len(input_data) / GENERATOR_BATCH_SIZE):
      iteration = 0

    start = iteration * GENERATOR_BATCH_SIZE
    end = (iteration + 1) * GENERATOR_BATCH_SIZE - 1

    batch_data = input_data[start:end,].transpose()

    # Batch text. Tokenize.
    raw_batch_text_data = batch_data[0]
    batch_text_data = tokenizer.texts_to_sequences(raw_batch_text_data)
    batch_text_data = pad_sequences(batch_text_data, maxlen=MAX_SEQUENCE_LENGTH)

    # Batch images. Load them.
    raw_batch_image_data = batch_data[1]
    batch_image_data = []
    for image_path in raw_batch_image_data:
      try:
        img = image_utils.load_img(image_path, target_size=(299, 299))
        x = image_utils.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
      
        batch_image_data.append(x[0])
      except Exception as e:
        print(e)
        batch_image_data.append(np.zeros((299, 299, 3)))

    batch_image_data = np.asarray(batch_image_data)

    batch_labels = input_labels[start:end,]

    yield [ batch_text_data, batch_image_data], batch_labels
    iteration += 1


##
## Text Model
##
def build_text_model(word_index):
  text_input = Input(shape=(MAX_SEQUENCE_LENGTH,))

  embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))

  for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)

    if embedding_vector is not None:
      # words not found in embedding index will be all-zeros.
      embedding_matrix[i] = embedding_vector[:EMBEDDING_DIM]

  embedding_layer = Embedding(embedding_matrix.shape[0],
                              embedding_matrix.shape[1],
                              weights=[embedding_matrix],
                              input_length=MAX_SEQUENCE_LENGTH)



  x = embedding_layer(text_input)
  x.trainable = False
  x = Conv1D(128, 5, activation='relu')(x)
  x = MaxPooling1D(5)(x)
  x = Conv1D(128, 5, activation='relu')(x)
  x = MaxPooling1D(5)(x)
  x = Flatten()(x)
  x = Dense(1024, activation='relu')(x)
  
  return x, text_input

##
## Image model
##
def build_image_model():
  base_model = InceptionV3(weights='imagenet', include_top=False)

   # Freeze Inception's weights - we don't want to train these
  for layer in base_model.layers:
     layer.trainable = False

  # add a fully connected layer after Inception - we do want to train these
  x = base_model.output
  x = GlobalAveragePooling2D()(x)
  x = Dense(2048, activation='relu')(x)

  return x, base_model.input


# Build the two models.
text_x, text_input = build_text_model(word_index)
image_x, image_input = build_image_model()

joint_x = merge([ text_x, image_x ], mode='concat')
joint_x = Dense(512, activation='relu')(joint_x)
# todo add batch normalization before merge.
predictions = Dense(label_length, activation='sigmoid')(joint_x)

full_model = Model(inputs=[ text_input, image_input ], outputs=[ predictions ])

full_model.compile(loss='binary_crossentropy',
                   optimizer='rmsprop',
                   metrics=[ 'binary_accuracy' ])


print("Training...")
full_model.fit_generator(product_generator(data_train, labels_train),
                         validation_data=product_generator(data_test, labels_test),
                         validation_steps=1,
                         steps_per_epoch=len(data)/GENERATOR_BATCH_SIZE,
                         class_weight=class_weights,
                         epochs=2)



# Save the model
full_model_json = full_model.to_json()

with open('./models/text_and_image.json', 'w') as json_file:
  json_file.write(full_model_json)

# Save the weights.
full_model.save_weights('./models/text_and_image_weights.h5')

# Save the tokenizer.
pickle.dump(tokenizer, open('./models/tokenizer.p', 'wb'))

# Save the label index.
pickle.dump(label_index, open('./models/label_index.p', 'wb'))

print("Model saved!")