from flask import Flask, request
from flask_cors import CORS
from huggingFacePipeline import Sentiment
from myModelPipeline import myModelSentiment
import torch

app = Flask('server')
CORS(app)

# Debug mode
app.debug = True

# Init two models.
sentiment = Sentiment()
mySentiment = myModelSentiment()


# ----------ROUTES----------
@app.route('/', methods=['GET'])
def server():
    return (
        '<p> Server is running!.... </p>'
    )

# Route using HuggingFace pipeline for sentiment analysis.
@app.route('/huggingface', methods=['POST'])
def huggingface():
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


@app.route('/mymodel', methods=['POST'])
def my_model():
    data = request.json
    temp = []
    temp.append(data.get('text_input'))
    text = temp[0]

    analysis = mySentiment.analyze(text)

    prob = analysis.item()

    tone = torch.round(torch.sigmoid(analysis))

    print(tone.item())
    print(prob)

    answer = {
        "senti": "",
        "prob": "",
    }

    if tone.item() > 0:
        answer["senti"] = "POSITIVE"
        answer["prob"] = str(analysis)
    else:
        answer["senti"] = "NEGATIVE"
        answer["prob"] = str(analysis)

    return answer


if __name__ == '__main__':
    app.run()
