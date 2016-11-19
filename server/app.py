from flask import Flask, render_template, jsonify


app = Flask(__name__)


@app.route('/')
def home():
    return 'home'

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
