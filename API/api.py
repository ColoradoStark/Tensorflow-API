
from flask import Flask, request, jsonify
import numpy as np
from skimage import transform, io
from tensorflow import keras
import tensorflow as tf


app = Flask(__name__)


@app.route('/api/recognize_image', methods=['POST'])
def recognize_image():

    img_url = request.get_json()['img_url']

    # prepare image for prediction
    img_array = io.imread(img_url, as_gray=True)
    small_grey = transform.resize(
        img_array, (28, 28, 1), mode='symmetric', preserve_range=True)
    img_to_predict = np.expand_dims(small_grey / 255.0, 0)

    # predict
    prediction_array = model.predict(img_to_predict)

    # prepare api response
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress',
                   'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    result = {
        "prediction": class_names[np.argmax(prediction_array)],
        "confidence": '{:2.0f}%'.format(100*np.max(prediction_array))
    }

    return jsonify(isError=False, message="Success", statusCode=200, data=result), 200


if __name__ == '__main__':
    model = keras.models.load_model('/tf/API/Fashion/1')
    app.run(debug=True, host='0.0.0.0')
