from flask import Flask, render_template, request, send_file
import os
import pickle
import numpy as np
from werkzeug.utils import secure_filename
# from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing.image import load_img, img_to_array
# from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.utils import to_categorical, plot_model
# from keras.layers import Input, Dense, LSTM, Embedding, Dropout, add
from keras.models import load_model


app = Flask(__name__)
model = VGG16()
vgg_model = Model(inputs=model.inputs, outputs=model.layers[-2].output)

# Get the absolute path of the 'images' directory in the current file's directory
images_dir = os.path.join(os.path.dirname(__file__), 'img')

# Create the 'images' directory if it doesn't exist
os.makedirs(images_dir, exist_ok=True)

md = load_model(r'D:\near_by_share\mlai\ImageC\modeltrain1.h5')

BASE_DIR = r'D:\near_by_share\mlai\ImageC'
WORKING_DIR = r'D:\near_by_share\mlai\ImageC'

# load features from pickle
with open(os.path.join(WORKING_DIR, 'features (1).pkl'), 'rb') as f:
    features = pickle.load(f)

with open(os.path.join(BASE_DIR, 'captions.txt'), 'r') as f:
    next(f)
    captions_doc = f.read()


def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None



# generate caption for an image
def predict_caption(model, image, tokenizer, max_length):
    # add start tag for generation process
    in_text = 'startseq'
    # iterate over the max length of sequence
    for i in range(max_length):
        # encode input sequence
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        # pad the sequence
        sequence = pad_sequences([sequence], max_length)
        # predict next word
        yhat = model.predict([image, sequence], verbose=0)
        # get index with high probability
        yhat = np.argmax(yhat)
        # convert index to word
        word = idx_to_word(yhat, tokenizer)
        # stop if word not found
        if word is None:
            break
        # append word as input for generating next word
        in_text += " " + word
        # stop if we reach end tag
        if word == 'endseq':
            break

    return in_text



image_path =None

@app.route('/', methods=['GET'])
def hello_word():
    return render_template('index.html')


@app.route('/', methods=['POST'])
def predict():
    images_dir = 'D:/near_by_share/mlai/ImageC/fl/img'
    
    # Check if the 'imagefile' key is in the request.files
    if 'imagefile' not in request.files:
        return 'No file part'

    imagefile = request.files['imagefile']

    # Check if the file has a valid name
    if imagefile.filename == '':
        return 'No selected file'

    # Use secure_filename to ensure a safe filename
    filename = secure_filename(imagefile.filename)

    global image_path
    # Build the absolute path for the image file
    image_path = os.path.join(images_dir, filename)

    # Save the uploaded image to the specified directory
    imagefile.save(image_path)

    # load image
    image = load_img(image_path, target_size=(224, 224))
    # convert image pixels to numpy array
    image = img_to_array(image)
    # reshape data for model
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    # preprocess image for vgg
    image = preprocess_input(image)
    # extract features
    vision_features = vgg_model.predict(image, verbose=0)    

    # Load the tokenizer and max length used during training
    with open(os.path.join(WORKING_DIR, 'tokenizer.pkl'), 'rb') as token_file:
        tokenizer = pickle.load(token_file)
 
    with open(os.path.join(WORKING_DIR, 'max_length.pkl'), 'rb') as maxlen_file:
        max_length_prediction = pickle.load(maxlen_file)

    predicted_caption=predict_caption(md, vision_features, tokenizer, max_length_prediction)

    # return render_template('index.html', prediction=predicted_caption)
    return render_template('index.html',predicted=predicted_caption)

@app.route('/get_img',methods=['POST', 'GET'])
def get_img():
    return send_file(image_path, as_attachment=False)

if __name__ == '__main__':
    app.run(port=3000, debug=True)