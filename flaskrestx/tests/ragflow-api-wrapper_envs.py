from flask import Flask
from flask_restx import Api, Resource, fields
from werkzeug.datastructures import FileStorage
import requests
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)
api = Api(app, version='1.0', title='RAGFlow API', description='A wrapper for RAGFlow API')

# Base URL for RAGFlow API
BASE_URL = os.getenv('RAGFLOW_BASE_URL', 'https://demo.ragflow.io/v1')
API_KEY = os.getenv('RAGFLOW_API_KEY')

# Helper function to make authenticated requests
def make_request(method, endpoint, **kwargs):
    headers = {'Authorization': f'Bearer {API_KEY}'}
    url = f'{BASE_URL}{endpoint}'
    response = requests.request(method, url, headers=headers, **kwargs)
    return response.json()

# ... [rest of the API wrapper code remains the same] ...

# Test cases
import unittest
import json
from unittest.mock import patch

class TestRAGFlowAPI(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()

    def test_env_variables(self):
        self.assertIsNotNone(os.getenv('RAGFLOW_BASE_URL'))
        self.assertIsNotNone(os.getenv('RAGFLOW_API_KEY'))
        self.assertEqual(os.getenv('RAGFLOW_BASE_URL'), BASE_URL)
        self.assertEqual(os.getenv('RAGFLOW_API_KEY'), API_KEY)

    @patch('requests.request')
    def test_new_conversation(self, mock_request):
        mock_response = {
            'data': {
                'id': 'test_conversation_id',
                'create_date': '2024-07-26 10:00:00',
                'user_id': 'test_user'
            },
            'retcode': 0,
            'retmsg': 'success'
        }
        mock_request.return_value.json.return_value = mock_response

        response = self.app.get('/conversation/new', json={'user_id': 'test_user'})
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('data', data)
        self.assertIn('id', data['data'])
        self.assertEqual(data['data']['id'], 'test_conversation_id')

        # Check if the API was called with correct parameters
        mock_request.assert_called_once_with(
            'GET',
            f'{BASE_URL}/api/new_conversation',
            headers={'Authorization': f'Bearer {API_KEY}'},
            params={'user_id': 'test_user'}
        )

    @patch('requests.request')
    def test_conversation_history(self, mock_request):
        mock_response = {
            'data': {
                'id': 'test_conversation_id',
                'message': [
                    {'role': 'assistant', 'content': 'Hello! How can I help you today?'},
                    {'role': 'user', 'content': 'What is RAGFlow?'}
                ]
            },
            'retcode': 0,
            'retmsg': 'success'
        }
        mock_request.return_value.json.return_value = mock_response

        response = self.app.get('/conversation/test_conversation_id')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('data', data)
        self.assertIn('message', data['data'])
        self.assertEqual(len(data['data']['message']), 2)

        # Check if the API was called with correct parameters
        mock_request.assert_called_once_with(
            'GET',
            f'{BASE_URL}/api/conversation/test_conversation_id',
            headers={'Authorization': f'Bearer {API_KEY}'}
        )

    @patch('requests.request')
    def test_completion(self, mock_request):
        mock_response = {
            'data': {
                'answer': 'RAGFlow is a powerful tool for managing and querying knowledge bases.',
                'reference': {
                    'chunks': [
                        {'content': 'RAGFlow is a tool for knowledge management.', 'doc_name': 'about_ragflow.pdf'}
                    ]
                }
            },
            'retcode': 0,
            'retmsg': 'success'
        }
        mock_request.return_value.json.return_value = mock_response

        response = self.app.post('/conversation/completion', json={
            'conversation_id': 'test_conversation_id',
            'messages': [{'role': 'user', 'content': 'What is RAGFlow?'}]
        })
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('data', data)
        self.assertIn('answer', data['data'])
        self.assertIn('RAGFlow', data['data']['answer'])

        # Check if the API was called with correct parameters
        mock_request.assert_called_once_with(
            'POST',
            f'{BASE_URL}/api/completion',
            headers={'Authorization': f'Bearer {API_KEY}'},
            json={
                'conversation_id': 'test_conversation_id',
                'messages': [{'role': 'user', 'content': 'What is RAGFlow?'}]
            }
        )

    # ... [other test cases remain the same] ...

if __name__ == '__main__':
    unittest.main()
