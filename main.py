import numpy as np
from keras.applications.mobilenet_v3 import MobileNetV3Large, preprocess_input
import matplotlib.pyplot as plt
from PIL import Image
from pickle import load
from keras.utils import load_img,img_to_array
from keras_preprocessing.sequence import pad_sequences
from keras.models import Model, load_model

tokenizer = load(open('tokenizer.pkl','rb'))
def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

def predict_caption(model, image, tokenizer, max_length):
    in_text = 'startseq'
    for i in range(max_length):
        seq = tokenizer.texts_to_sequences([in_text])[0]
        seq = pad_sequences([seq], maxlen=max_length)
        yhat = model.predict([image, seq], verbose=0)
        pred_ids = np.argmax(yhat)
        word = idx_to_word(pred_ids, tokenizer)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'endseq':
            break
    processed_text = in_text.split(" ")[1:-1]
    processed_text = " ".join(processed_text)
    return processed_text

def extract_feature(img_path):
    model = MobileNetV3Large(weights='imagenet')
    model = Model(model.input, model.layers[-2].output)
    image = load_img(img_path, target_size=(224, 224),keep_aspect_ratio=True)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    
    feature = model.predict(image, verbose=0)
    return feature

model = load_model('./best_model_1000.h5')
actual, predicted = list(), list()

def generate_caption(image_path):
    image = extract_feature(image_path)
    y_pred = predict_caption(model, image, tokenizer, 41)
    return y_pred
def main():
    image_path = "test3.jpg"
    caption = generate_caption(image_path)
    fig, ax = plt.subplots()
    image = Image.open(image_path)
    ax.imshow(image)
    plt.text(0, image.height, caption, color='white', backgroundcolor='black', fontsize=12)
    ax.axis('off')
    plt.show()
if __name__ == "__main__":
    main()