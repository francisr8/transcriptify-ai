import base64
import json
import os

from flask import (Flask, abort, flash, jsonify, redirect, render_template,
                   request)

from pyannote_handler import auto_diarization, custom_diarization, test

app = Flask(__name__)

save_path = 'data/temp.wav'


@app.route('/')
def index():
    return "Pyannote"


@app.route("/diarization", methods=["POST"])
def uploadfile():
    data = request.form['file']
    saveFile(data)
    segments = auto_diarization(save_path)
    return jsonify(segments), 200


@app.route("/custom_diarization/<speakers>", methods=["POST"])
def uploadfile_custom(speakers):
    data = request.form['file']
    saveFile(data)
    segments = custom_diarization(save_path, speakers)
    return jsonify(segments), 200


@app.route('/test_custom/<dataset>', methods=["GET"])
def test_custom(dataset):
    if os.path.exists('results.json'):
        with open('results.json') as json_file:
            result = json.load(json_file)
            key = 'customSpectralClustering' if dataset == 'default' else dataset + \
                'customSpectralClustering'
            return jsonify(result[key]), 200
    else:
        return '', 400


@app.route('/test_auto/<dataset>', methods=["GET"])
def test_auto(dataset):
    if os.path.exists('results.json'):
        with open('results.json') as json_file:
            result = json.load(json_file)
            key = 'auto' if dataset == 'default' else dataset + \
                'auto'
            return jsonify(result[key]), 200
    else:
        return '', 400


@app.route('/test_data', methods=['GET'])
def test_data():
    if os.path.exists('results.json'):
        with open('results.json') as json_file:
            result = json.load(json_file)
            return jsonify(result), 200
    else:
        return '', 400


@app.route('/generate_test/<service>', methods=['GET'])
def generate_test(service):
    custom = 'PyannoteManual' == service
    test(custom)
    return '', 200


@app.route('/test_status/<service>', methods=["GET"])
def test_status(service):
    path = 'custom_results.json' if service == 'custom' else 'auto_results.json'
    if os.path.exists(path):
        return jsonify('ready'), 200
    else:
        return jsonify('not ready'), 200


def saveFile(data):
    data = base64.b64decode(data)
    if not os.path.exists('data'):
        os.makedirs('data')
    with open(save_path, "wb") as f:
        f.write(data)


if __name__ == '__main__':
    app.run()
