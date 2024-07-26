# test_config.py
import os
from dotenv import load_dotenv
import unittest

load_dotenv()

class TestConfig(unittest.TestCase):
    def test_env_variables(self):
        self.assertIsNotNone(os.getenv('RAGFLOW_BASE_URL'))
        self.assertIsNotNone(os.getenv('RAGFLOW_API_KEY'))
        self.assertTrue(os.getenv('RAGFLOW_BASE_URL').startswith('https://'))
        self.assertTrue(len(os.getenv('RAGFLOW_API_KEY')) > 0)

# test_new_conversation.py
import unittest
from unittest.mock import patch
import json
from app import app  # Import your Flask app

class TestNewConversation(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()

    @patch('app.make_request')
    def test_new_conversation(self, mock_make_request):
        mock_response = {
            'data': {
                'id': 'test_conversation_id',
                'create_date': '2024-07-26 10:00:00',
                'user_id': 'test_user'
            },
            'retcode': 0,
            'retmsg': 'success'
        }
        mock_make_request.return_value = mock_response

        response = self.app.get('/conversation/new', json={'user_id': 'test_user'})
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('data', data)
        self.assertIn('id', data['data'])
        self.assertEqual(data['data']['id'], 'test_conversation_id')

        mock_make_request.assert_called_once_with('GET', '/api/new_conversation', params={'user_id': 'test_user'})

# test_conversation_history.py
import unittest
from unittest.mock import patch
import json
from app import app

class TestConversationHistory(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()

    @patch('app.make_request')
    def test_conversation_history(self, mock_make_request):
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
        mock_make_request.return_value = mock_response

        response = self.app.get('/conversation/test_conversation_id')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('data', data)
        self.assertIn('message', data['data'])
        self.assertEqual(len(data['data']['message']), 2)

        mock_make_request.assert_called_once_with('GET', '/api/conversation/test_conversation_id')

# test_completion.py
import unittest
from unittest.mock import patch
import json
from app import app

class TestCompletion(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()

    @patch('app.make_request')
    def test_completion(self, mock_make_request):
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
        mock_make_request.return_value = mock_response

        response = self.app.post('/conversation/completion', json={
            'conversation_id': 'test_conversation_id',
            'messages': [{'role': 'user', 'content': 'What is RAGFlow?'}]
        })
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('data', data)
        self.assertIn('answer', data['data'])
        self.assertIn('RAGFlow', data['data']['answer'])

        mock_make_request.assert_called_once_with('POST', '/api/completion', json={
            'conversation_id': 'test_conversation_id',
            'messages': [{'role': 'user', 'content': 'What is RAGFlow?'}]
        })

# test_upload_document.py
import unittest
from unittest.mock import patch, mock_open
import json
from app import app

class TestUploadDocument(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()

    @patch('app.make_request')
    def test_upload_document(self, mock_make_request):
        mock_response = {
            'data': {
                'id': 'test_document_id',
                'name': 'test_document.txt',
                'kb_id': 'test_kb_id'
            },
            'retcode': 0,
            'retmsg': 'success'
        }
        mock_make_request.return_value = mock_response

        with patch('builtins.open', mock_open(read_data='This is a test document.')) as mock_file:
            response = self.app.post('/document/upload', data={
                'file': (mock_file, 'test_document.txt'),
                'kb_name': 'test_kb'
            }, content_type='multipart/form-data')

        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('data', data)
        self.assertIn('id', data['data'])
        self.assertEqual(data['data']['name'], 'test_document.txt')

        mock_make_request.assert_called_once()
        call_args = mock_make_request.call_args
        self.assertEqual(call_args[0][0], 'POST')
        self.assertEqual(call_args[0][1], '/api/document/upload')
        self.assertIn('files', call_args[1])
        self.assertIn('data', call_args[1])

# test_list_documents.py
import unittest
from unittest.mock import patch
import json
from app import app

class TestListDocuments(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()

    @patch('app.make_request')
    def test_list_documents(self, mock_make_request):
        mock_response = {
            'data': {
                'docs': [
                    {'doc_id': 'doc1', 'doc_name': 'document1.pdf'},
                    {'doc_id': 'doc2', 'doc_name': 'document2.txt'}
                ],
                'total': 2
            },
            'retcode': 0,
            'retmsg': 'success'
        }
        mock_make_request.return_value = mock_response

        response = self.app.post('/document/list', json={'kb_name': 'test_kb'})
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('data', data)
        self.assertIn('docs', data['data'])
        self.assertEqual(len(data['data']['docs']), 2)

        mock_make_request.assert_called_once_with('POST', '/api/list_kb_docs', json={'kb_name': 'test_kb'})

# test_delete_documents.py
import unittest
from unittest.mock import patch
import json
from app import app

class TestDeleteDocuments(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()

    @patch('app.make_request')
    def test_delete_documents(self, mock_make_request):
        mock_response = {
            'data': True,
            'retcode': 0,
            'retmsg': 'success'
        }
        mock_make_request.return_value = mock_response

        response = self.app.delete('/document/delete', json={'doc_ids': ['doc1', 'doc2']})
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('data', data)
        self.assertTrue(data['data'])

        mock_make_request.assert_called_once_with('DELETE', '/api/document', json={'doc_ids': ['doc1', 'doc2']})

if __name__ == '__main__':
    unittest.main()
