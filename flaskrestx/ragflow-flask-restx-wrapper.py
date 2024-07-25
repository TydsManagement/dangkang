import os
import requests
from flask import Flask, request, send_file
from flask_restx import Api, Resource, fields
from werkzeug.datastructures import FileStorage

app = Flask(__name__)
api = Api(app, version='1.0', title='RAGFlow API',
          description='A Flask-RESTX wrapper for RAGFlow API')

BASE_URL = "https://demo.ragflow.io/v1/"
API_KEY = os.environ.get("RAGFLOW_API_KEY")

HEADERS = {
    "Authorization": f"Bearer {API_KEY}"
}

# Define namespaces
ns_conversation = api.namespace('conversation', description='Conversation operations')
ns_document = api.namespace('document', description='Document operations')
ns_kb = api.namespace('kb', description='Knowledge Base operations')

# Define models
new_conversation_model = api.model('NewConversation', {
    'user_id': fields.String(required=True, description='User ID')
})

completion_model = api.model('Completion', {
    'conversation_id': fields.String(required=True, description='Conversation ID'),
    'messages': fields.Raw(required=True, description='Messages'),
    'quote': fields.Boolean(required=False, description='Quote'),
    'stream': fields.Boolean(required=False, description='Stream'),
    'doc_ids': fields.String(required=False, description='Document IDs')
})

upload_parser = api.parser()
upload_parser.add_argument('file', location='files', type=FileStorage, required=True)
upload_parser.add_argument('kb_name', type=str, required=True)
upload_parser.add_argument('parser_id', type=str, required=False)
upload_parser.add_argument('run', type=str, required=False)

list_chunks_model = api.model('ListChunks', {
    'doc_name': fields.String(required=False, description='Document name'),
    'doc_id': fields.String(required=False, description='Document ID')
})

list_kb_docs_model = api.model('ListKbDocs', {
    'kb_name': fields.String(required=True, description='Knowledge Base name'),
    'page': fields.Integer(required=False, description='Page number'),
    'page_size': fields.Integer(required=False, description='Page size'),
    'orderby': fields.String(required=False, description='Order by'),
    'desc': fields.Boolean(required=False, description='Descending order'),
    'keywords': fields.String(required=False, description='Keywords')
})

delete_documents_model = api.model('DeleteDocuments', {
    'doc_names': fields.List(fields.String, required=False, description='Document names'),
    'doc_ids': fields.List(fields.String, required=False, description='Document IDs')
})

@ns_conversation.route('/new')
class NewConversation(Resource):
    @api.expect(new_conversation_model)
    def get(self):
        user_id = request.args.get('user_id')
        response = requests.get(f"{BASE_URL}api/new_conversation", headers=HEADERS, params={"user_id": user_id})
        return response.json()

@ns_conversation.route('/<string:id>')
class Conversation(Resource):
    def get(self, id):
        response = requests.get(f"{BASE_URL}api/conversation/{id}", headers=HEADERS)
        return response.json()

@ns_conversation.route('/completion')
class Completion(Resource):
    @api.expect(completion_model)
    def post(self):
        data = request.json
        response = requests.post(f"{BASE_URL}api/completion", headers=HEADERS, json=data)
        return response.json()

@ns_document.route('/<string:id>')
class Document(Resource):
    def get(self, id):
        response = requests.get(f"{BASE_URL}document/get/{id}", headers=HEADERS)
        return send_file(response.content, mimetype=response.headers['Content-Type'])

@ns_document.route('/upload')
class Upload(Resource):
    @api.expect(upload_parser)
    def post(self):
        args = upload_parser.parse_args()
        file = args['file']
        kb_name = args['kb_name']
        parser_id = args['parser_id']
        run = args['run']

        files = {'file': (file.filename, file)}
        data = {'kb_name': kb_name}
        if parser_id:
            data['parser_id'] = parser_id
        if run:
            data['run'] = run

        response = requests.post(f"{BASE_URL}api/document/upload", headers=HEADERS, files=files, data=data)
        return response.json()

@ns_document.route('/list_chunks')
class ListChunks(Resource):
    @api.expect(list_chunks_model)
    def get(self):
        doc_name = request.args.get('doc_name')
        doc_id = request.args.get('doc_id')

        if not doc_name and not doc_id:
            api.abort(400, "Either doc_name or doc_id is required")

        params = {}
        if doc_name:
            params['doc_name'] = doc_name
        if doc_id:
            params['doc_id'] = doc_id

        response = requests.get(f"{BASE_URL}api/list_chunks", headers=HEADERS, params=params)
        return response.json()

@ns_kb.route('/list_docs')
class ListKbDocs(Resource):
    @api.expect(list_kb_docs_model)
    def post(self):
        data = request.json
        response = requests.post(f"{BASE_URL}api/list_kb_docs", headers=HEADERS, json=data)
        return response.json()

@ns_document.route('/delete')
class DeleteDocuments(Resource):
    @api.expect(delete_documents_model)
    def delete(self):
        data = request.json
        response = requests.delete(f"{BASE_URL}api/document", headers=HEADERS, json=data)
        return response.json()

if __name__ == '__main__':
    app.run(debug=True)
