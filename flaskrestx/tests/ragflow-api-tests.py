import pytest
from flask import json
from ragflow-flask-restx-wrapper import app, api
from unittest.mock import patch, MagicMock

@pytest.fixture
def client():
    with app.test_client() as client:
        yield client

def test_new_conversation(client):
    with patch('requests.get') as mock_get:
        mock_get.return_value.json.return_value = {'conversation_id': '123456'}
        response = client.get('/conversation/new?user_id=test_user')
        assert response.status_code == 200
        assert json.loads(response.data) == {'conversation_id': '123456'}
        mock_get.assert_called_once_with(
            "https://demo.ragflow.io/v1/api/new_conversation",
            headers={"Authorization": "Bearer None"},
            params={"user_id": "test_user"}
        )

def test_get_conversation(client):
    with patch('requests.get') as mock_get:
        mock_get.return_value.json.return_value = {'conversation': 'data'}
        response = client.get('/conversation/123456')
        assert response.status_code == 200
        assert json.loads(response.data) == {'conversation': 'data'}
        mock_get.assert_called_once_with(
            "https://demo.ragflow.io/v1/api/conversation/123456",
            headers={"Authorization": "Bearer None"}
        )

def test_completion(client):
    with patch('requests.post') as mock_post:
        mock_post.return_value.json.return_value = {'completion': 'result'}
        data = {
            'conversation_id': '123456',
            'messages': [{'role': 'user', 'content': 'Hello'}],
            'quote': True,
            'stream': False
        }
        response = client.post('/conversation/completion', json=data)
        assert response.status_code == 200
        assert json.loads(response.data) == {'completion': 'result'}
        mock_post.assert_called_once_with(
            "https://demo.ragflow.io/v1/api/completion",
            headers={"Authorization": "Bearer None"},
            json=data
        )

def test_get_document(client):
    with patch('requests.get') as mock_get:
        mock_get.return_value.content = b'document content'
        mock_get.return_value.headers = {'Content-Type': 'application/pdf'}
        response = client.get('/document/123456')
        assert response.status_code == 200
        assert response.data == b'document content'
        assert response.headers['Content-Type'] == 'application/pdf'
        mock_get.assert_called_once_with(
            "https://demo.ragflow.io/v1/document/get/123456",
            headers={"Authorization": "Bearer None"}
        )

def test_upload_document(client):
    with patch('requests.post') as mock_post:
        mock_post.return_value.json.return_value = {'doc_id': '123456'}
        data = {
            'kb_name': 'test_kb',
            'parser_id': 'test_parser',
            'run': 'true'
        }
        response = client.post('/document/upload', 
                               data=data, 
                               content_type='multipart/form-data',
                               buffered=True,
                               files={'file': (BytesIO(b'my file contents'), 'test.txt')})
        assert response.status_code == 200
        assert json.loads(response.data) == {'doc_id': '123456'}
        mock_post.assert_called_once()
        assert mock_post.call_args[0][0] == "https://demo.ragflow.io/v1/api/document/upload"
        assert mock_post.call_args[1]['headers'] == {"Authorization": "Bearer None"}
        assert 'files' in mock_post.call_args[1]
        assert 'data' in mock_post.call_args[1]

def test_list_chunks(client):
    with patch('requests.get') as mock_get:
        mock_get.return_value.json.return_value = {'chunks': ['chunk1', 'chunk2']}
        response = client.get('/document/list_chunks?doc_name=test.txt')
        assert response.status_code == 200
        assert json.loads(response.data) == {'chunks': ['chunk1', 'chunk2']}
        mock_get.assert_called_once_with(
            "https://demo.ragflow.io/v1/api/list_chunks",
            headers={"Authorization": "Bearer None"},
            params={'doc_name': 'test.txt'}
        )

def test_list_kb_docs(client):
    with patch('requests.post') as mock_post:
        mock_post.return_value.json.return_value = {'docs': ['doc1', 'doc2']}
        data = {
            'kb_name': 'test_kb',
            'page': 1,
            'page_size': 10
        }
        response = client.post('/kb/list_docs', json=data)
        assert response.status_code == 200
        assert json.loads(response.data) == {'docs': ['doc1', 'doc2']}
        mock_post.assert_called_once_with(
            "https://demo.ragflow.io/v1/api/list_kb_docs",
            headers={"Authorization": "Bearer None"},
            json=data
        )

def test_delete_documents(client):
    with patch('requests.delete') as mock_delete:
        mock_delete.return_value.json.return_value = {'deleted': ['doc1', 'doc2']}
        data = {
            'doc_names': ['doc1.txt', 'doc2.txt']
        }
        response = client.delete('/document/delete', json=data)
        assert response.status_code == 200
        assert json.loads(response.data) == {'deleted': ['doc1', 'doc2']}
        mock_delete.assert_called_once_with(
            "https://demo.ragflow.io/v1/api/document",
            headers={"Authorization": "Bearer None"},
            json=data
        )
