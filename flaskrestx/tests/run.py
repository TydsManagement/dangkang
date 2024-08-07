from flask import Flask
from flask_restx import Api, Resource

api = Api()

app = Flask(__name__)
api.init_app(app)

@api.route("/api/hello")
class HelloWorld(Resource):
    def get(self):
        return {"hello": "world"}
