from pathlib import Path

from flask import Flask, redirect, request, render_template, url_for
from werkzeug.utils import secure_filename

from dog_breed_classifier import paths
from dog_breed_classifier.detection.breed_detection import detect_breed
from dog_breed_classifier.detection.dog_detection import detect_dog
from dog_breed_classifier.detection.face_detection import detect_human_face


ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = str(paths.UPLOAD_FOLDER)


@app.route('/', methods=('GET', ))
def index():
    """
    Enables the user to submit .png, .jpg or .jpeg files.
    """
    return render_template('upload.html')


@app.route('/upload', methods=('POST', ))
def upload():
    """
    Stores the file submitted via the '/' endpoint.
    :return:
    """
    if 'file' not in request.files:
        return redirect(url_for('index'))

    file = request.files['file']
    if not file.filename:
        return redirect(url_for('index'))

    if not allowed_file(file.filename):
        return redirect(url_for('index'))

    file_name = secure_filename(file.filename)
    file.save(Path(app.config['UPLOAD_FOLDER']) / file_name)

    return redirect(url_for('result', file_name=file_name))


@app.route('/result/<file_name>', methods=('GET', ))
def result(file_name):
    """
    Classifies the file given by file_name.
    :param file_name: name of the file to classify.
    :return: html page displaying the file and the classification result.
    """

    img_path: Path = Path(app.config['UPLOAD_FOLDER']) / file_name

    if not detect_dog(img_path) and not detect_human_face(img_path):
        message = 'Neither a dog or a human face could be detected in the following picture: '
        breed = ''

    else:
        message = 'The following breed was detected for the uploaded image:'
        breed = detect_breed(img_path)
    return render_template('result.html',
                           message=message,
                           breed=breed,
                           file_name=file_name)
