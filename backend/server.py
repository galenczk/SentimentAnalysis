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
    text = []
    text.append(data.get('text_input'))

    analysis = sentiment.analyze(text)

    answer = {
        "senti": "",
        "prob": "",
    }

    # response[0] = probability of text being NEGATIVE
    # response[1] = probablity of text being POSITIVE
    if analysis[0] > analysis[1]:
        answer["senti"] = "NEGATIVE"
        answer["prob"] = str(analysis[0])
    else:
        answer["senti"] = "POSITIVE"
        answer["prob"] = str(analysis[1])

    return answer


if __name__ == '__main__':
    app.run()
