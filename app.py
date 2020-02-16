import io
from keras.applications import ResNet50
from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
from PIL import Image
import numpy as np
import flask
from flask import request

app = flask.Flask(__name__)
model = None


def load_model():
    """Load ResNet50 Keras model."""
    global model
    model = ResNet50(weights="imagenet")


load_model()


def prepare_image(image, target):
    """Pre-process image to be ran with the model.

    Parameters
    ----------
    image : BytesIO
        Bytes object from user uploaded from interface
    target : tuple
        Image size to resize (244,244)

    Returns
    -------
    image
        Ready to ingest into keras model
    """
    # if the image mode is not RGB, convert it
    if image.mode != "RGB":
        image = image.convert("RGB")

    # resize the input image and preprocess it
    image = image.resize(target)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = imagenet_utils.preprocess_input(image)

    # return the processed image
    return image


@app.route("/predict", methods=["POST"])
def predict():
    # initialize the data dictionary that will be returned from the
    # view
    data = {"success": False}

    # ensure an image was properly uploaded to our endpoint
    if flask.request.method == "POST":
        if flask.request.files.get("image"):
            # read the image in PIL format
            image = flask.request.files["image"].read()
            image = Image.open(io.BytesIO(image))

            # preprocess the image and prepare it for classification
            image = prepare_image(image, target=(224, 224))

            # classify the input image and then initialize the list
            # of predictions to return to the client
            preds = model.predict(image)
            results = imagenet_utils.decode_predictions(preds)
            data["predictions"] = []

            # loop over the results and add them to the list of
            # returned predictions
            for (imagenetID, label, prob) in results[0]:
                r = {"label": label, "probability": float(prob)}
                data["predictions"].append(r)

            # indicate that the request was a success
            data["success"] = True

    # return the data dictionary as a JSON response
    return flask.jsonify(data)


@app.route("/", methods=["POST", "GET"])
def home():
    # initialize the data dictionary that will be returned from the
    # view
    data = {"success": False}

    if request.method == "GET":
        return """
        <!doctype html>
        <title>Image Classification Using Resnet </title>
        <h1>This is Nitin Gupta Image Classification model </h1>
        <h3>Pls Upload Image </h3>
        <strong>Note:</strong> Due To Free Server output Will be in json fromate
        <form method=post enctype=multipart/form-data>
        <p><input type=file name=file>
            <input type=submit value=Select Prediction Image>
        </form>
        """

    # ensure an image was properly uploaded to our endpoint
    if flask.request.method == "POST":
        # read the image in PIL format
        image = request.files["file"].read()
        image = Image.open(io.BytesIO(image))

        # preprocess the image and prepare it for classification
        image = prepare_image(image, target=(224, 224))
        # classify the input image and then initialize the list
        # of predictions to return to the client
        preds = model.predict(image)
        results = imagenet_utils.decode_predictions(preds)
        data["predictions"] = []

        # loop over the results and add them to the list of
        # returned predictions
        for (imagenetID, label, prob) in results[0]:
            r = {"label": label, "probability": float(prob)}
            data["predictions"].append(r)

        # indicate that the request was a success
        data["success"] = True

    # return the data dictionary as a JSON response
    return flask.jsonify(data)


if __name__ == "__main__":
    print(
        (
            "******This is nitin Gupta Server Pls wait loading Model"
        )
    )
    load_model()
    app.run()
