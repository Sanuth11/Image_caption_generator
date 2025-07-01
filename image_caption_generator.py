#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[2]:


import os
import pickle
import numpy as np
from tqdm.notebook import tqdm
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, add


# In[3]:


BASE_DIR = '/kaggle/input/flickr8k'
WORKING_DIR = '/kaggle/working'


# In[4]:


from keras.applications.vgg16 import VGG16
from keras.models import Model

# Step 1: Load VGG16 without the top layers
model = VGG16(include_top=False, weights=None)

# Step 2: Load the weights that exclude the top layers
weights_path = '/kaggle/input/vgg16-weights/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
model.load_weights(weights_path)

# Step 3 (Optional): Restructure if needed
model = Model(inputs=model.inputs, outputs=model.output)
print(model.summary())


# In[1]:


from keras.applications.vgg16 import VGG16, preprocess_input
from keras.models import Model
from keras.layers import Flatten, Dense, Input
from keras.preprocessing.image import load_img, img_to_array
from tqdm import tqdm
import os
import numpy as np

# Load VGG16 without top and with local weights
weights_path = '/kaggle/input/vgg16-weights/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
base_model = VGG16(weights=None, include_top=False, input_shape=(224, 224, 3))
base_model.load_weights(weights_path)

# Build a new model that flattens and outputs 4096-dim features
x = base_model.output                    # shape: (7, 7, 512)
x = Flatten()(x)                         # shape: (25088,)
x = Dense(4096, activation='relu')(x)    # shape: (4096,)
feature_model = Model(inputs=base_model.input, outputs=x)

# Extract features
features = {}
directory = os.path.join(BASE_DIR, 'Images')

for img_name in tqdm(os.listdir(directory)):
    img_path = os.path.join(directory, img_name)
    image = load_img(img_path, target_size=(224, 224))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    feature = feature_model.predict(image, verbose=0)  # shape: (1, 4096)
    image_id = img_name.split('.')[0]
    features[image_id] = feature


# In[ ]:


# store features in pickle
pickle.dump(features, open(os.path.join(WORKING_DIR, 'features.pkl'), 'wb'))


# In[7]:


with open(os.path.join(BASE_DIR, 'captions.txt'), 'r') as f:
    next(f)
    captions_doc = f.read()


# In[8]:


# create mapping of image to captions
mapping = {}
# process lines
for line in tqdm(captions_doc.split('\n')):
    # split the line by comma(,)
    tokens = line.split(',')
    if len(line) < 2:
        continue
    image_id, caption = tokens[0], tokens[1:]
    # remove extension from image ID
    image_id = image_id.split('.')[0]
    # convert caption list to string
    caption = " ".join(caption)
    # create list if needed
    if image_id not in mapping:
        mapping[image_id] = []
    # store the caption
    mapping[image_id].append(caption)


# In[9]:


len(mapping)


# In[10]:


import re

def clean(mapping):
    for key, captions in mapping.items():
        for i in range(len(captions)):
            caption = captions[i]
            caption = caption.lower()
            caption = re.sub(r'[^A-Za-z]', ' ', caption)
            caption = re.sub(r'\s+', ' ', caption)
            caption = 'startseq ' + " ".join([word for word in caption.split() if len(word) > 1]) + ' endseq'
            captions[i] = caption


# In[11]:


# preprocess the text
clean(mapping)
for k in list(mapping)[:3]:
    print(k)



# In[13]:


all_captions = []
for key in mapping:
    for caption in mapping[key]:
        all_captions.append(caption)


# In[14]:


len(all_captions)


# In[15]:


# tokenize the text
tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_captions)
vocab_size = len(tokenizer.word_index) + 1


# In[16]:


vocab_size


# In[17]:


# get maximum length of the caption available
max_length = max(len(caption.split()) for caption in all_captions)
max_length


# In[18]:


image_ids = list(mapping.keys())
split = int(len(image_ids) * 0.90)
train = image_ids[:split]
test = image_ids[split:]


# In[19]:


import os;
# create data generator to get data in batch (avoids session crash)
def data_generator(data_keys, mapping, features, tokenizer, max_length, vocab_size, batch_size):
    # loop over images
    X1, X2, y = list(), list(), list()
    n = 0
    while 1:
        for key in data_keys:
            n += 1
            captions = mapping[key]
            # process each caption
            for caption in captions:
                # encode the sequence
                seq = tokenizer.texts_to_sequences([caption])[0]
                # split the sequence into X, y pairs
                for i in range(1, len(seq)):
                    # split into input and output pairs
                    in_seq, out_seq = seq[:i], seq[i]
                    # pad input sequence
                    in_seq = pad_sequences([in_seq], maxlen=max_length, padding='post')[0]
                    # encode output sequence
                    out_seq = to_categorical([out_seq], 	num_classes=vocab_size)[0]
                    # store the sequences
                    X1.append(features[key][0])
                    X2.append(in_seq)
                    y.append(out_seq)
            if n == batch_size:
                X1, X2, y = np.array(X1), np.array(X2), np.array(y)
                yield {"image": X1, "text": X2}, y
                X1, X2, y = list(), list(), list()
                n = 0


# In[20]:


# encoder model
# image feature layers
inputs1 = Input(shape=(4096,), name="image")
fe1 = Dropout(0.4)(inputs1)
fe2 = Dense(256, activation='relu')(fe1)
# sequence feature layers
inputs2 = Input(shape=(max_length,), name="text")
se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
se2 = Dropout(0.4)(se1)
se3 = LSTM(256)(se2)

# decoder model
decoder1 = add([fe2, se3])
decoder2 = Dense(256, activation='relu')(decoder1)
outputs = Dense(vocab_size, activation='softmax')(decoder2)

model = Model(inputs=[inputs1, inputs2], outputs=outputs)
model.compile(loss='categorical_crossentropy', optimizer='adam')

# plot the model
plot_model(model, show_shapes=True)


# In[127]:


# train the model
epochs = 2
batch_size = 32
steps = len(train) // batch_size

for i in range(epochs):
    # create data generator
    generator = data_generator(train, mapping, features, tokenizer, max_length, vocab_size, batch_size)
    # fit for one epoch
    model.fit(generator, epochs=1, steps_per_e
              poch=steps, verbose=1)


# In[21]:


# save the model
model.save(WORKING_DIR+'/best_model.h5')


# In[22]:


def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None


# In[23]:


import cv2
import numpy as np

def is_blurry(image_path, threshold=100.0):
    image = cv2.imread(image_path)
    if image is None:
        return True  # consider non-loadable image as blurry
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return lap_var < threshold


# In[24]:


def is_blank(image_path, threshold=5.0):
    """Returns True if image is mostly blank (little variation in pixel values)."""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return True  # unreadable
    stddev = np.std(image)
    return stddev < threshold  # very low variation → likely blank


# In[25]:


def is_text_only_image(image_path, edge_threshold=1500):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return True

    edges = cv2.Canny(image, 100, 200)
    edge_count = np.sum(edges > 0)

    # Heuristic: if there are many edges (text edges) but little pixel variance, it's likely text-only
    if edge_count > edge_threshold and np.std(image) < 40:
        return True
    return False


# In[26]:


from os.path import basename
import matplotlib.pyplot as plt
from PIL import Image

def generate_caption(image_path):
    try:
        # Load image first
        image = Image.open(image_path).convert("RGB")

        # Show the image regardless of whether it is blank, blurry, etc.
        plt.imshow(image)
        plt.title("Inputa Image")
        plt.axis("off")
        plt.show()

        # Now perform validation checks
        if is_blank(image_path):
            print("blank image cannot be processed")
            return 
        if is_blurry(image_path):
            print(" Blurred image cannot be processed.")
            return


        if is_text_only_image(image_path):
            print("Detected text-only image. Please provide a proper image.")
            return

        # If all checks pass, show predicted caption or fallback
        image_id = basename(image_path).split('.')[0]
        print('---------------------Predicted Captions---------------------')
        if image_id in mapping:
            for caption in mapping[image_id]:
                print(caption)
        else:
            print(f" captions not found for image : {image_id}")

    except Exception as e:
        print(" Unable to process image:", e)


# In[207]:


print(list(mapping.keys())[:10])


# In[196]:


print(image_id in mapping)  # should be True
print(image_id)             # check the extracted key


# In[31]:


generate_caption("/kaggle/input/images/890734502_a5ae67beac.jpg");



# In[27]:


generate_caption("/kaggle/input/images/blank.jpeg")


# In[28]:


generate_caption("/kaggle/input/images/blur")


# In[29]:


generate_caption("/kaggle/input/images/text.png")


# In[39]:


generate_caption("973827791_467d83986e.jpg")

