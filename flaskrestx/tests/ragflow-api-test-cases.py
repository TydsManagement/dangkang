import pytest
from unittest.mock import patch, MagicMock
from flask import Flask
from ragflow_flask_restx_wrapper import app, api

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_new_conversation(client):
    with patch('ragflow-flask-restx-wrapper.requests.get') as mock_get:
        mock_response = MagicMock()
        mock_response.json.return_value = {"conversation_id": "123"}
        mock_get.return_value = mock_response

        response = client.get('/conversation/new?user_id=test_user')
        assert response.status_code == 200
        assert response.json == {"conversation_id": "123"}

def test_get_conversation(client):
    with patch('ragflow-flask-restx-wrapper.requests.get') as mock_get:
        mock_response = MagicMock()
        mock_response.json.return_value = {"conversation": "data"}
        mock_get.return_value = mock_response

        response = client.get('/conversation/123')
        assert response.status_code == 200
        assert response.json == {"conversation": "data"}

def test_completion(client):
    with patch('ragflow-flask-restx-wrapper.requests.post') as mock_post:
        mock_response = MagicMock()
        mock_response.json.return_value = {"completion": "result"}
        mock_post.return_value = mock_response

        data = {
            "conversation_id": "123",
            "messages": [{"role": "user", "content": "Hello"}],
            "quote": False,
            "stream": False
        }
        response = client.post('/conversation/completion', json=data)
        assert response.status_code == 200
        assert response.json == {"completion": "result"}

def test_get_document(client):
    with patch('ragflow-flask-restx-wrapper.requests.get') as mock_get, \
         patch('ragflow-flask-restx-wrapper.send_file') as mock_send_file:
        mock_response = MagicMock()
        mock_response.content = b"document content"
        mock_response.headers = {'Content-Type': 'application/pdf'}
        mock_get.return_value = mock_response

        mock_send_file.return_value = "mocked_file"

        response = client.get('/document/123')
        mock_send_file.assert_called_once_with(b"document content", mimetype='application/pdf')

def test_upload_document(client):
    with patch('ragflow-flask-restx-wrapper.requests.post') as mock_post:
        mock_response = MagicMock()
        mock_response.json.return_value = {"document_id": "123"}
        mock_post.return_value = mock_response

        data = {
            'kb_name': 'test_kb',
            'file': (io.BytesIO(b"test content"), 'test.txt')
        }
        response = client.post('/document/upload', data=data, content_type='multipart/form-data')
        assert response.status_code == 200
        assert response.json == {"document_id": "123"}

def test_list_chunks(client):
    with patch('ragflow-flask-restx-wrapper.requests.get') as mock_get:
        mock_response = MagicMock()
        mock_response.json.return_value = {"chunks": ["chunk1", "chunk2"]}
        mock_get.return_value = mock_response

        response = client.get('/document/list_chunks?doc_name=test.txt')
        assert response.status_code == 200
        assert response.json == {"chunks": ["chunk1", "chunk2"]}

def test_list_kb_docs(client):
    with patch('ragflow-flask-restx-wrapper.requests.post') as mock_post:
        mock_response = MagicMock()
        mock_response.json.return_value = {"documents": ["doc1", "doc2"]}
        mock_post.return_value = mock_response

        data = {"kb_name": "test_kb"}
        response = client.post('/kb/list_docs', json=data)
        assert response.status_code == 200
        assert response.json == {"documents": ["doc1", "doc2"]}

def test_delete_documents(client):
    with patch('ragflow-flask-restx-wrapper.requests.delete') as mock_delete:
        mock_response = MagicMock()
        mock_response.json.return_value = {"deleted": ["doc1", "doc2"]}
        mock_delete.return_value = mock_response

        data = {"doc_names": ["doc1.txt", "doc2.txt"]}
        response = client.delete('/document/delete', json=data)
        assert response.status_code == 200
        assert response.json == {"deleted": ["doc1", "doc2"]}
