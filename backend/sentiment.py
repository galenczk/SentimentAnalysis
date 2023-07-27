from transformers import pipeline


class Sentiment:
    def __init__(self) -> None:
        self.pipe = pipeline(
            "text-classification",
            model="nlptown/bert-base-multilingual-uncased-sentiment",
            device=0,
        )

        self.tones = [
            "Very Negative",
            "Negative",
            "Neutral",
            "Positive",
            "Very Positive",
        ]

    def analyze(self, text):
        response = self.pipe(text)[0]

        star_number = int(response["label"][0])

        sentiment_desc = self.tones[star_number - 1]

        response["description"] = sentiment_desc

        return response
