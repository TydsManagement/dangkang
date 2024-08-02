from flask import Flask
from flask_restx import Api, Resource, fields
from werkzeug.datastructures import FileStorage
import requests

app = Flask(__name__)
api = Api(app, version='1.0', title='RAGFlow API', description='A wrapper for RAGFlow API')

# Base URL for RAGFlow API
BASE_URL = 'https://demo.ragflow.io/v1'

# Helper function to make authenticated requests
def make_request(method, endpoint, **kwargs):
    headers = {'Authorization': f'Bearer {API_KEY}'}
    url = f'{BASE_URL}{endpoint}'
    response = requests.request(method, url, headers=headers, **kwargs)
    return response.json()

# Namespaces
ns_conversation = api.namespace('conversation', description='Conversation operations')
ns_document = api.namespace('document', description='Document operations')

# Models
conversation_model = api.model('Conversation', {
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

# Endpoints
@ns_conversation.route('/new')
class NewConversation(Resource):
    @api.expect(conversation_model)
    def get(self):
        """Create a new conversation"""
        return make_request('GET', '/api/new_conversation', params=api.payload)

@ns_conversation.route('/<string:id>')
class ConversationHistory(Resource):
    def get(self, id):
        """Get conversation history"""
        return make_request('GET', f'/api/conversation/{id}')

@ns_conversation.route('/completion')
class Completion(Resource):
    @api.expect(completion_model)
    def post(self):
        """Get answer"""
        return make_request('POST', '/api/completion', json=api.payload)

@ns_document.route('/get/<string:id>')
class GetDocument(Resource):
    def get(self, id):
        """Get document content"""
        return make_request('GET', f'/document/get/{id}')

@ns_document.route('/upload')
class UploadDocument(Resource):
    @api.expect(upload_parser)
    def post(self):
        """Upload a file"""
        args = upload_parser.parse_args()
        files = {'file': (args['file'].filename, args['file'].read())}
        data = {k: v for k, v in args.items() if k != 'file' and v is not None}
        return make_request('POST', '/api/document/upload', files=files, data=data)

@ns_document.route('/chunks')
class ListChunks(Resource):
    @api.param('doc_name', 'Document name')
    @api.param('doc_id', 'Document ID')
    def get(self):
        """Get document chunks"""
        return make_request('GET', '/api/list_chunks', params=api.payload)

@ns_document.route('/list')
class ListDocuments(Resource):
    @api.param('kb_name', 'Knowledge base name', required=True)
    @api.param('page', 'Page number')
    @api.param('page_size', 'Page size')
    @api.param('orderby', 'Order by')
    @api.param('desc', 'Descending order')
    @api.param('keywords', 'Keywords')
    def post(self):
        """Get document list"""
        return make_request('POST', '/api/list_kb_docs', json=api.payload)

@ns_document.route('/delete')
class DeleteDocuments(Resource):
    @api.param('doc_names', 'Document names', type='list')
    @api.param('doc_ids', 'Document IDs', type='list')
    def delete(self):
        """Delete documents"""
        return make_request('DELETE', '/api/document', json=api.payload)

if __name__ == '__main__':
    app.run(debug=True)

# Test cases
import unittest
import json

class TestRAGFlowAPI(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()

    def test_new_conversation(self):
        response = self.app.get('/conversation/new', json={'user_id': 'test_user'})
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('data', data)
        self.assertIn('id', data['data'])

    def test_conversation_history(self):
        # First, create a new conversation
        response = self.app.get('/conversation/new', json={'user_id': 'test_user'})
        conversation_id = json.loads(response.data)['data']['id']

        # Then, get the conversation history
        response = self.app.get(f'/conversation/{conversation_id}')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('data', data)
        self.assertIn('message', data['data'])

    def test_completion(self):
        # First, create a new conversation
        response = self.app.get('/conversation/new', json={'user_id': 'test_user'})
        conversation_id = json.loads(response.data)['data']['id']

        # Then, get a completion
        response = self.app.post('/conversation/completion', json={
            'conversation_id': conversation_id,
            'messages': [{'role': 'user', 'content': 'Hello, how are you?'}]
        })
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('data', data)
        self.assertIn('answer', data['data'])

    def test_upload_document(self):
        with open('test_document.txt', 'w') as f:
            f.write('This is a test document.')

        with open('test_document.txt', 'rb') as f:
            response = self.app.post('/document/upload', data={
                'file': (f, 'test_document.txt'),
                'kb_name': 'test_kb'
            }, content_type='multipart/form-data')

        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('data', data)
        self.assertIn('id', data['data'])

    def test_list_documents(self):
        response = self.app.post('/document/list', json={'kb_name': 'test_kb'})
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('data', data)
        self.assertIn('docs', data['data'])

    def test_delete_documents(self):
        # First, list documents to get an ID
        response = self.app.post('/document/list', json={'kb_name': 'test_kb'})
        docs = json.loads(response.data)['data']['docs']
        if docs:
            doc_id = docs[0]['doc_id']
            response = self.app.delete('/document/delete', json={'doc_ids': [doc_id]})
            self.assertEqual(response.status_code, 200)
            data = json.loads(response.data)
            self.assertIn('data', data)
            self.assertTrue(data['data'])

if __name__ == '__main__':
    unittest.main()
