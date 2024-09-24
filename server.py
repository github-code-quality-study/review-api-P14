import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import json
import pandas as pd
from datetime import datetime
import uuid
import os
from typing import Callable, Any
from wsgiref.simple_server import make_server
from urllib.parse import parse_qs

nltk.download('vader_lexicon', quiet=True)

sia = SentimentIntensityAnalyzer()

VALID_LOCATIONS = [
    "Albuquerque, New Mexico", "Carlsbad, California", "Chula Vista, California",
    "Colorado Springs, Colorado", "Denver, Colorado", "El Cajon, California",
    "El Paso, Texas", "Escondido, California", "Fresno, California",
    "La Mesa, California", "Las Vegas, Nevada", "Los Angeles, California",
    "Oceanside, California", "Phoenix, Arizona", "Sacramento, California",
    "Salt Lake City, Utah", "San Diego, California", "Tucson, Arizona"
]

class ReviewAnalyzerServer:
    def __init__(self) -> None:
        self.reviews = pd.read_csv('data/reviews.csv').to_dict('records')
        for review in self.reviews:
            review['sentiment'] = self.analyze_sentiment(review['ReviewBody'])

    def analyze_sentiment(self, review_body):
        return sia.polarity_scores(review_body)

    def __call__(self, environ: dict[str, Any], start_response: Callable[..., Any]) -> bytes:
        if environ["REQUEST_METHOD"] == "GET":
            return self.handle_get(environ, start_response)
        elif environ["REQUEST_METHOD"] == "POST":
            return self.handle_post(environ, start_response)
        else:
            start_response("405 Method Not Allowed", [("Content-Type", "text/plain")])
            return [b"Method not allowed"]

    def handle_get(self, environ, start_response):
        query = parse_qs(environ['QUERY_STRING'])
        location = query.get('location', [None])[0]
        start_date = query.get('start_date', [None])[0]
        end_date = query.get('end_date', [None])[0]

        filtered_reviews = self.reviews

        if location and location in VALID_LOCATIONS:
            filtered_reviews = [r for r in filtered_reviews if r['Location'] == location]

        if start_date:
            start_date = datetime.strptime(start_date, '%Y-%m-%d')
            filtered_reviews = [r for r in filtered_reviews if datetime.strptime(r['Timestamp'], '%Y-%m-%d %H:%M:%S') >= start_date]

        if end_date:
            end_date = datetime.strptime(end_date, '%Y-%m-%d')
            filtered_reviews = [r for r in filtered_reviews if datetime.strptime(r['Timestamp'], '%Y-%m-%d %H:%M:%S') <= end_date]

        # Ensure all reviews have sentiment analysis
        for review in filtered_reviews:
            if 'sentiment' not in review:
                review['sentiment'] = self.analyze_sentiment(review['ReviewBody'])

        sorted_reviews = sorted(filtered_reviews, key=lambda x: x['sentiment']['compound'], reverse=True)

        response_body = json.dumps(sorted_reviews, indent=2).encode("utf-8")
        start_response("200 OK", [
            ("Content-Type", "application/json"),
            ("Content-Length", str(len(response_body)))
        ])
        return [response_body]

    

    def handle_post(self, environ, start_response):
        try:
            content_length = int(environ.get('CONTENT_LENGTH', 0))
            post_data = environ['wsgi.input'].read(content_length).decode('utf-8')
            data = parse_qs(post_data)

            location = data.get('Location', [''])[0]
            review_body = data.get('ReviewBody', [''])[0]

            if not location or not review_body:
                start_response("400 Bad Request", [("Content-Type", "text/plain")])
                return [b"Missing Location or ReviewBody"]

            if location not in VALID_LOCATIONS:
                start_response("400 Bad Request", [("Content-Type", "text/plain")])
                return [b"Invalid Location"]

            new_review = {
                "ReviewId": str(uuid.uuid4()),
                "ReviewBody": review_body,
                "Location": location,
                "Timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }

            # Add the new review without sentiment
            self.reviews.append(new_review)

            response_body = json.dumps(new_review, indent=2).encode("utf-8")
            start_response("201 Created", [
                ("Content-Type", "application/json"),
                ("Content-Length", str(len(response_body)))
            ])
            return [response_body]

        except Exception as e:
            start_response("500 Internal Server Error", [("Content-Type", "text/plain")])
            return [str(e).encode('utf-8')]


if __name__ == "__main__":
    app = ReviewAnalyzerServer()
    port = 8000  # Changed to 8080
    with make_server("", port, app) as httpd:
        print(f"Listening on port {port}...")
        httpd.serve_forever()

