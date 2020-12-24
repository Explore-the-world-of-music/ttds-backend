import time
from flask import Flask, jsonify
from flask_cors import CORS, cross_origin

app = Flask(__name__)
CORS(app)

@app.route('/api/time')
def get_current_time():
    return jsonify({'time': time.time()})
