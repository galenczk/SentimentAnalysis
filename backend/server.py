from flask import Flask, request
from flask_cors import CORS
from sentiment import Sentiment

app = Flask('server')
CORS(app)

# Debug mode
app.debug = True

sentiment = Sentiment()


# ----------ROUTES----------
@app.route('/', methods=['GET'])
def server():
    return (
        '<p> Server is running!.... </p>'
    )


@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    text = data.get('text_input')

    response = sentiment.analyze(text)

    return response


if __name__ == '__main__':
    app.run()
