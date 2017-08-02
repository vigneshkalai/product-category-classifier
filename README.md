# Product Category Classification Engine

This is a snapshot of the product category classification engine used by Two Tap. It uses glove word vectors and images to categorize a product to a unified [taxonomy](https://api.twotap.com/v1.0/product/taxonomy).

## How to use the it with the pre-trained weights.

* Clone the repo.
* Download the pretrained weights from [here](https://twotapml.blob.core.windows.net/tt-product-classification/08012017/categories/models.tar.gz), unpack them, and place them in the models/ directory.
* Run: ```python predict.py```.
* You can edit things there.

The models/ directory should look like below:

![models-dir](http://i.imgur.com/n0i2Pst.png)


## How to train the model.

* Download training.csv from [here](https://twotapml.blob.core.windows.net/tt-product-classification/08012017/categories/training_data.csv.gz) and place it in the ml-data/ directory.
* Download the glove pre-trained vectors from [here](http://nlp.stanford.edu/data/glove.6B.zip) and place them in the ml-data/ directory.
* Run training/download_images.py to fetch all the images.
* Run training/train.py. This might take a long time.

Check the models/ directory for the model json and weights afterward.

The ml-data/ directory should look like below, but with more images:

![ml-data-dir](http://i.imgur.com/rptggkE.png)

Want access to the product images so you don't have to download them from the retailers? Email support@twotap.com.
