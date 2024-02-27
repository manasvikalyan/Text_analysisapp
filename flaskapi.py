from flask import Flask, request, jsonify
from flask_restful import Resource, Api, reqparse
import werkzeug
import os
import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import bert_score
from rouge_score import rouge_scorer

class msg(Resource):

    def calculate_similarity(self, text1, text2):
        vectorizer = CountVectorizer().fit_transform([text1, text2])
        vectors = vectorizer.toarray()
        return cosine_similarity(vectors)[0][1]
    
    def bert_similarity(self,text1, text2):
        P, R, F1 = bert_score.score([text1], [text2], lang="en", verbose=True)
        return F1.item()

    def rouge_similarity(self, text1, text2):
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        scores = scorer.score(text1, text2)
        return scores['rougeL'].fmeasure
    
    def post(self):
        parser = reqparse.RequestParser()
        parser.add_argument('text1', type=str)
        parser.add_argument('text2', type=str)
        args = parser.parse_args()
        sim = self.calculate_similarity(args['text1'], args['text2'])
        # sim2 = self.bert_similarity(args['text1'], args['text2'])
        sim3 = self.rouge_similarity(args['text1'], args['text2'])
        return jsonify({'similarity': sim, 'rouge_similarity': sim3})


app = Flask(__name__)
api = Api(app)
api.add_resource(msg, '/msg')

if __name__ == '__main__':
    app.run(debug=True)
