#!/usr/bin/env python3

import socket
socket.setdefaulttimeout(30)

import csv
import hashlib
import queue
import os
import requests
import shutil
import urllib

from threading import Thread

from urllib.parse import urlparse

concurrent = 20

def md5(string):
  m = hashlib.md5()
  m.update(string.encode('utf-8'))
  return m.hexdigest()

# Thread/Queue based download.
def queueDownload():
  while True:
    image_meta = queue.get()
    image_url = image_meta['url']
    image_path = image_meta['path']

    print(f"Downloading {image_url} - {image_path}")

    try:
      headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'
      }

      response = requests.get(image_url, stream=True, headers=headers)

      with open(image_path, 'wb') as out_file:
        shutil.copyfileobj(response.raw, out_file)

      del response
    except Exception as e:
      print(e)

    queue.task_done()


queue = queue.Queue(concurrent * 2)
for i in range(concurrent):
  t = Thread(target = queueDownload)
  t.daemon = True
  t.start()


count = 0
with open('./ml-data/training_data.csv', 'r') as csvfile:
  csvreader = csv.reader(csvfile)

  for row in csvreader:
    try:
      parsed_product_url = urlparse(row[0])
      product_host = parsed_product_url.hostname
      product_host_md5 = md5(product_host)
      product_host_dir = f"./ml-data/images/{product_host_md5}" 
    except Exception as e:
      print(e)
      continue

    if not os.path.isdir(product_host_dir):
       os.makedirs(product_host_dir)


    all_images = row[3]
    all_images = all_images.split('###')

    for image_url in all_images:
      product_md5 = md5(image_url)

      image_path = f"ml-data/images/{product_host_md5}/img_{product_md5}.png"

      if os.path.exists(image_path):
        print(f"Exists {image_path}")
        continue

      queue.put({ 'url': image_url, 'path': image_path })

    count += 1

queue.join()