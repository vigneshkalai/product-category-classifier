#!/usr/bin/env python3

# This deletes text/html from the images/ directory.
# Sometimes we ask for images and we get 404s for instance.

import socket
socket.setdefaulttimeout(30)

import csv
import hashlib
import os
import magic
import urllib

from threading import Thread
from urllib.parse import urlparse

concurrent = 20

def md5(string):
  m = hashlib.md5()
  m.update(string.encode('utf-8'))
  return m.hexdigest()


count = 0
with open('./ml-data/training_data.csv', 'r') as csvfile:
  csvreader = csv.reader(csvfile)

  for row in csvreader:
    try:
      parsed_product_url = urlparse(row[0])
      product_host = parsed_product_url.hostname
      product_host_md5 = md5(product_host)
      product_host_dir = "./ml-data/images/%s" % product_host_md5
    except Exception as e:
      print(e)
      continue

    if not os.path.isdir(product_host_dir):
       os.makedirs(product_host_dir)


    all_images = row[3]
    all_images = all_images.split('###')

    for image_url in all_images:

      product_md5 = md5(image_url)

      image_path = "ml-data/images/%s/img_%s.png" % (product_host_md5, product_md5)

      if os.path.exists(image_path):
        file_type = magic.from_file(image_path, mime=True)

        if file_type == 'text/html':
          os.remove(image_path) 

          print("Deleting:")
          print(image_path)
          print(file_type)