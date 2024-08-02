import os
import requests
from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename

app = Flask(__name__)

BASE_URL = "https://demo.ragflow.io/v1/"
API_KEY = os.environ.get("RAGFLOW_API_KEY")

HEADERS = {
    "Authorization": f"Bearer {API_KEY}"
}

@app.route('/new_conversation', methods=['GET'])
def new_conversation():
    user_id = request.args.get('user_id')
    if not user_id:
        return jsonify({"error": "user_id is required"}), 400

    response = requests.get(f"{BASE_URL}api/new_conversation", headers=HEADERS, params={"user_id": user_id})
    return jsonify(response.json())

@app.route('/conversation/<id>', methods=['GET'])
def get_conversation(id):
    response = requests.get(f"{BASE_URL}api/conversation/{id}", headers=HEADERS)
    return jsonify(response.json())

@app.route('/completion', methods=['POST'])
def get_answer():
    data = request.json
    response = requests.post(f"{BASE_URL}api/completion", headers=HEADERS, json=data)
    return jsonify(response.json())

@app.route('/document/<id>', methods=['GET'])
def get_document(id):
    response = requests.get(f"{BASE_URL}document/get/{id}", headers=HEADERS)
    return send_file(response.content, mimetype=response.headers['Content-Type'])

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    kb_name = request.form.get('kb_name')
    parser_id = request.form.get('parser_id')
    run = request.form.get('run')

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file and kb_name:
        filename = secure_filename(file.filename)
        files = {'file': (filename, file)}
        data = {'kb_name': kb_name}
        if parser_id:
            data['parser_id'] = parser_id
        if run:
            data['run'] = run

        response = requests.post(f"{BASE_URL}api/document/upload", headers=HEADERS, files=files, data=data)
        return jsonify(response.json())

@app.route('/list_chunks', methods=['GET'])
def list_chunks():
    doc_name = request.args.get('doc_name')
    doc_id = request.args.get('doc_id')

    if not doc_name and not doc_id:
        return jsonify({"error": "Either doc_name or doc_id is required"}), 400

    params = {}
    if doc_name:
        params['doc_name'] = doc_name
    if doc_id:
        params['doc_id'] = doc_id

    response = requests.get(f"{BASE_URL}api/list_chunks", headers=HEADERS, params=params)
    return jsonify(response.json())

@app.route('/list_kb_docs', methods=['POST'])
def list_kb_docs():
    data = request.json
    response = requests.post(f"{BASE_URL}api/list_kb_docs", headers=HEADERS, json=data)
    return jsonify(response.json())

@app.route('/delete_documents', methods=['DELETE'])
def delete_documents():
    data = request.json
    response = requests.delete(f"{BASE_URL}api/document", headers=HEADERS, json=data)
    return jsonify(response.json())

if __name__ == '__main__':
    app.run(debug=True)
