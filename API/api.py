
from flask import Flask, request, jsonify
import numpy as np
from skimage import transform, io
from tensorflow import keras
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)


@app.route('/api/recognize_image', methods=['POST'])
def recognize_image():

    img_url = request.get_json()['img_url']

    # Get the Image and convert it into an array
    # Also flip the colors so that it matches MNIST set
    img_array = io.imread(img_url, as_gray=True)
    small_grey = transform.resize(
        img_array, (28, 28), mode='symmetric', preserve_range=True)
    small_grey = (small_grey * -1)
    small_grey = small_grey / 255.0

    # Uncomment the next line you want to preview image in a Notebook
    # plt.imshow(small_grey)

    # Scale it MinMax between 0 and 1
    scaler = MinMaxScaler()
    scaled_image = small_grey
    scaler.fit_transform(scaled_image)
    scaled_image.reshape(28, 28, 1)

    # Now Make an N-Dimensional Numpy Array and Insert it
    # So that it is compatible with the model
    data_array = np.ndarray((1, 28, 28, 1), dtype=float)
    np.append(data_array, scaled_image)

    # predict
    prediction_array = model.predict(data_array)

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
