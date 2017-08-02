#!/usr/bin/env python3

# Requires a S3 dump. You can use tools/download-s3-data from squashed-feeds.
from bs4 import BeautifulSoup
import csv
import glob
import msgpack
import requests
import json
import inflection
import re

##
## Fetch labels
##
url = 'https://api.twotap.com/v1.0/product/taxonomy'

resp = requests.get(url=url, params=None)
taxonomy = json.loads(resp.text)['taxonomy']

##
## Find the deepest categories.
##
deep_categories = {}
for category in taxonomy:
  split_categories = category.split('~~')
  last_category = split_categories.pop()
  deep_categories[last_category] = category

  for subcategory in split_categories:
    try:
      del deep_categories[subcategory]
    except KeyError:
      pass

##
## Break up things like Shorts & Rompers
##
temp_categories = {}
for category in deep_categories:
  if '&' in category:
    pieces = category.split('&')
    pieces = [piece.strip() for piece in pieces]
    last_piece = ' '.join(pieces[-1].split(' ')[1:])
    pieces[-1] = pieces[-1].split(' ')[0]

    for piece in pieces:
      piece = piece.strip() + ' ' + last_piece
      temp_categories[piece.strip()] = deep_categories[category]

    if temp_categories.get(category):
      del temp_categories[category]
  else:
    temp_categories[category] = deep_categories[category]


deep_categories = temp_categories


##
## Singularize
##
temp_categories = {}
for category in deep_categories:
  single_category = inflection.singularize(category)
  temp_categories[single_category] = deep_categories[category]


deep_categories = temp_categories

##
## Ban certain words
##
for word in [ 'Baby', 'Set', 'Collection', 'Jersey' ]:
  del deep_categories[word]


with open('./ml-data/training_data.csv', 'w') as csvfile:
  csvwriter = csv.writer(csvfile)

  for filename in glob.iglob('./ml-data/products/*/*'):
    file = open(filename, 'rb') 
    raw_product = file.read() 
    
    product = msgpack.unpackb(raw_product, encoding='utf-8')

    if product.get('categories') == None or product.get('site_categories') == None:
      continue

    title = product.get('title')
    image = product.get('image')
    category = product['categories'][0]

    if category == 'Everything Else':
      continue

    for deep_category in deep_categories:
      category_path = deep_categories[deep_category]

      if category.split('~~')[0] == category_path.split('~~')[0]:
        pattern = re.compile(r"\b" + deep_category + r"s?\b", re.IGNORECASE)
        #print(pattern)
        if pattern.match(title):
          print(f"{title}-{deep_category}-{category_path}")
          print(image)


    site_categories = product.get('site_categories')
    if site_categories:
      site_categories = '###'.join(site_categories)

    all_images = []

    if image:
      all_images.append(image)

    alt_images = product.get('alt_images')
    if alt_images:
      all_images = all_images + alt_images

    all_images = list(set(all_images))
    all_images = '###'.join(all_images)

    description = product.get('description')
    if description:
      description = BeautifulSoup(description, 'html') 
      description = description.get_text().encode('ascii','ignore').strip()

    csvwriter.writerow([ product.get('url'), title, description, all_images, site_categories, category ])