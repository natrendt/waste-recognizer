import base64
import io
import os
from base64 import encodebytes

import PIL.Image
import requests
from flask import Flask, flash, request, redirect, url_for, render_template, json
from werkzeug.utils import secure_filename

from model import predictions

UPLOAD_FOLDER = 'static/photos'

if not os.path.exists(UPLOAD_FOLDER):
   os.makedirs(UPLOAD_FOLDER)

ALLOWED_EXTENSIONS = {'jpg', 'png'}
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_image(image_path):
    pil_img = PIL.Image.open(image_path, mode='r')  # reads the PIL image
    byte_arr = io.BytesIO()
    pil_img.save(byte_arr, format='PNG')  # convert the PIL image to byte array
    encoded_img = encodebytes(byte_arr.getvalue()).decode('ascii')  # encode as base64
    return encoded_img


def decode_image(coded_image: str):
    decoded_image = PIL.Image.open(io.BytesIO(base64.b64decode(str(coded_image))))
    image_rgb = decoded_image.convert("RGB")
    return image_rgb


@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(os.path.abspath(app.config['UPLOAD_FOLDER']), filename))
            res = requests.post(url_for('process_file', _external=True), json={
                "image": get_image(os.path.join(app.config['UPLOAD_FOLDER'], filename))})
            if res.ok:
                return render_template('results.html', results=res.json(),
                                       file=os.path.join(app.config['UPLOAD_FOLDER'], filename))

    return render_template('index.html')


@app.route('/prediction', methods=['POST'])
def process_file():
    image = decode_image(request.json["image"])
    modelresult = predictions(image)

    return app.response_class(
        response=json.dumps(modelresult).encode("utf8"), content_type="application/json"
    )


if __name__ == '__main__':
    app.run(host="localhost", port=5000)
