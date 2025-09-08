# from flask import Flask, render_template, request, send_file
# import os
# import pickle
# import numpy as np
# from werkzeug.utils import secure_filename
# # from keras.applications.vgg16 import decode_predictions
# from keras.applications.vgg16 import VGG16, preprocess_input
# from keras.preprocessing.image import load_img, img_to_array
# # from keras.preprocessing.text import Tokenizer
# from keras.preprocessing.sequence import pad_sequences
# from keras.models import Model
# from keras.utils import to_categorical, plot_model
# # from keras.layers import Input, Dense, LSTM, Embedding, Dropout, add
# from keras.models import load_model


# app = Flask(__name__)
# model = VGG16()
# vgg_model = Model(inputs=model.inputs, outputs=model.layers[-2].output)

# # Get the absolute path of the 'images' directory in the current file's directory
# images_dir = os.path.join(os.path.dirname(__file__), 'img')

# # Create the 'images' directory if it doesn't exist
# os.makedirs(images_dir, exist_ok=True)

# md = load_model(r'D:\near_by_share\mlai\ImageC\fl\modeltrain1.h5')

# BASE_DIR = r'D:\near_by_share\mlai\ImageC\fl'
# WORKING_DIR = r'D:\near_by_share\mlai\ImageC\fl'

# # load features from pickle
# with open(os.path.join(WORKING_DIR, 'features1.pkl'), 'rb') as f:
#     features = pickle.load(f)

# with open(os.path.join(BASE_DIR, 'captions.txt'), 'r') as f:
#     next(f)
#     captions_doc = f.read()


# def idx_to_word(integer, tokenizer):
#     for word, index in tokenizer.word_index.items():
#         if index == integer:
#             return word
#     return None



# # generate caption for an image
# def predict_caption(model, image, tokenizer, max_length):
#     # add start tag for generation process
#     in_text = 'startseq:'
#     # iterate over the max length of sequence
#     for i in range(max_length):
#         # encode input sequence
#         sequence = tokenizer.texts_to_sequences([in_text])[0]
#         # pad the sequence
#         sequence = pad_sequences([sequence], max_length)
#         # predict next word
#         yhat = model.predict([image, sequence], verbose=0)
#         # get index with high probability
#         yhat = np.argmax(yhat)
#         # convert index to word
#         word = idx_to_word(yhat, tokenizer)
#         # stop if word not found
#         if word is None:
#             break
#         # append word as input for generating next word
#         in_text += " " + word
#         # stop if we reach end tag
#         if word == 'endseq':
#             break

#     return in_text



# image_path =None

# @app.route('/', methods=['GET'])
# def hello_word():
#     return render_template('index.html')


# @app.route('/', methods=['POST'])
# def predict():
#     images_dir = 'D:/near_by_share/mlai/ImageC/fl/img'
    
#     # Check if the 'imagefile' key is in the request.files
#     if 'imagefile' not in request.files:
#         return 'No file part'

#     imagefile = request.files['imagefile']

#     # Check if the file has a valid name
#     if imagefile.filename == '':
#         return 'No selected file'

#     # Use secure_filename to ensure a safe filename
#     filename = secure_filename(imagefile.filename)

#     global image_path
#     # Build the absolute path for the image file
#     image_path = os.path.join(images_dir, filename)

#     # Save the uploaded image to the specified directory
#     imagefile.save(image_path)

#     # load image
#     image = load_img(image_path, target_size=(224, 224))
#     # convert image pixels to numpy array
#     image = img_to_array(image)
#     # reshape data for model
#     image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
#     # preprocess image for vgg
#     image = preprocess_input(image)
#     # extract features
#     vision_features = vgg_model.predict(image, verbose=0)    

#     # Load the tokenizer and max length used during training
#     with open(os.path.join(WORKING_DIR, 'tokenizer.pkl'), 'rb') as token_file:
#         tokenizer = pickle.load(token_file)
 
#     with open(os.path.join(WORKING_DIR, 'max_length.pkl'), 'rb') as maxlen_file:
#         max_length_prediction = pickle.load(maxlen_file)

#     predicted_caption=predict_caption(md, vision_features, tokenizer, max_length_prediction)

#     # return render_template('index.html', prediction=predicted_caption)
#     return render_template('index.html',predicted=predicted_caption)

# @app.route('/get_img',methods=['POST', 'GET'])
# def get_img():
#     return send_file(image_path, as_attachment=False)

# if __name__ == '__main__':
#     # app.run(port=3000, debug=True)
#     port = int(os.environ.get("PORT", 5000))  # Default to 5000 locally
#     app.run(host="0.0.0.0", port=port, debug=True)




from flask import Flask, render_template, request, send_file
import os
import pickle
import numpy as np
from werkzeug.utils import secure_filename
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing.image import load_img, img_to_array
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model, load_model
import traceback

# -------------------------------
# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# -------------------------------

app = Flask(__name__)

# -------------------------------
# Global variables
# -------------------------------
BASE_DIR = os.path.dirname(__file__)
images_dir = os.path.join(BASE_DIR, 'img')
os.makedirs(images_dir, exist_ok=True)

# Load features/captions once
# with open(os.path.join(BASE_DIR, 'features1.pkl'), 'rb') as f:
#     features = pickle.load(f)

# with open(os.path.join(BASE_DIR, 'captions.txt'), 'r') as f:
#     next(f)
#     captions_doc = f.read()

# Lazy-loaded models
vgg_model = None
caption_model = None
tokenizer = None
max_length_prediction = None
image_path = None


# -------------------------------
# Utility functions
# -------------------------------
def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None


def predict_caption(model, image, tokenizer, max_length):
    in_text = 'startseq'   # <-- removed the colon
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], max_length)
        yhat = model.predict([image, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = idx_to_word(yhat, tokenizer)
        if word is None:
            break
        in_text += " " + word
        if word == 'endseq':
            break
    return in_text


def load_all_models():
    """Load models & tokenizer once at startup"""
    global vgg_model, caption_model, tokenizer, max_length_prediction

    if vgg_model is None:
        print("Loading VGG16 model...")
        vgg_base = VGG16()
        vgg_model = Model(inputs=vgg_base.inputs, outputs=vgg_base.layers[-2].output)

    if caption_model is None:
        print("Loading captioning model...")
        caption_model = load_model(os.path.join(BASE_DIR, 'modeltrain1.h5'))

    if tokenizer is None:
        print("Loading tokenizer...")
        with open(os.path.join(BASE_DIR, 'tokenizer.pkl'), 'rb') as token_file:
            tokenizer = pickle.load(token_file)

    if max_length_prediction is None:
        print("Loading max length...")
        with open(os.path.join(BASE_DIR, 'max_length.pkl'), 'rb') as maxlen_file:
            max_length_prediction = pickle.load(maxlen_file)


# -------------------------------
# Routes
# -------------------------------
@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')


@app.route('/', methods=['POST'])
def predict():
    global image_path
    try:
        if 'imagefile' not in request.files:
            return 'No file part'
        imagefile = request.files['imagefile']
        if imagefile.filename == '':
            return 'No selected file'

        filename = secure_filename(imagefile.filename)
        image_path = os.path.join(images_dir, filename)
        imagefile.save(image_path)

        # Ensure models are loaded
        load_all_models()

        # Preprocess image
        img = load_img(image_path, target_size=(224, 224))
        img = img_to_array(img).reshape((1, 224, 224, 3))
        img = preprocess_input(img)
        vision_features = vgg_model.predict(img, verbose=0)

        # Generate caption
        predicted_caption = predict_caption(
            caption_model, vision_features, tokenizer, max_length_prediction
        )

        return render_template('index.html', predicted=predicted_caption)

    except Exception as e:
        print("❌ Error in prediction:", str(e))
        traceback.print_exc()
        return f"Error: {str(e)}", 500


@app.route('/get_img', methods=['POST', 'GET'])
def get_img():
    if image_path and os.path.exists(image_path):
        return send_file(image_path, as_attachment=False)
    return "No image uploaded", 404


# -------------------------------
# Entrypoint
# -------------------------------
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    load_all_models()  # <-- preload on startup
    app.run(host="0.0.0.0", port=port, debug=True)
