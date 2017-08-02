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



count = 0
simple_count = 0

with open('./ml-data/training_data.csv', 'r') as csvfile:
  csvreader = csv.reader(csvfile)

  for row in csvreader:
    if len(row) != 6:
        continue

    tt_categories = row[5]

    count += 1

    if len(tt_categories.split('~~')) == 2 and tt_categories == 'Apparel & Accessories~~Clothing':
      simple_count += 1


print(count)
print(simple_count)