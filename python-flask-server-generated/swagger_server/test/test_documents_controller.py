# coding: utf-8

from __future__ import absolute_import

from flask import json
from six import BytesIO

from swagger_server.models.document_details_response import DocumentDetailsResponse  # noqa: E501
from swagger_server.models.document_parsing_status_response import DocumentParsingStatusResponse  # noqa: E501
from swagger_server.models.documents_parse_body import DocumentsParseBody  # noqa: E501
from swagger_server.models.list_documents_response import ListDocumentsResponse  # noqa: E501
from swagger_server.models.success_response import SuccessResponse  # noqa: E501
from swagger_server.models.update_document_request import UpdateDocumentRequest  # noqa: E501
from swagger_server.models.upload_documents_response import UploadDocumentsResponse  # noqa: E501
from swagger_server.test import BaseTestCase


class TestDocumentsController(BaseTestCase):
    """DocumentsController integration test stubs"""

    def test_delete_document(self):
        """Test case for delete_document

        Delete document
        """
        response = self.client.open(
            '/v1/api/dataset/{dataset_id}/documents/{document_id}'.format(dataset_id='dataset_id_example', document_id='document_id_example'),
            method='DELETE')
        self.assert200(response,
                       'Response body is : ' + response.data.decode('utf-8'))

    def test_download_document(self):
        """Test case for download_document

        Download document
        """
        response = self.client.open(
            '/v1/api/dataset/{dataset_id}/documents/{document_id}'.format(dataset_id='dataset_id_example', document_id='document_id_example'),
            method='GET')
        self.assert200(response,
                       'Response body is : ' + response.data.decode('utf-8'))

    def test_get_document_parsing_status(self):
        """Test case for get_document_parsing_status

        Show parsing status of document
        """
        response = self.client.open(
            '/v1/api/dataset/{dataset_id}/documents/{document_id}/status'.format(dataset_id='dataset_id_example', document_id='document_id_example'),
            method='GET')
        self.assert200(response,
                       'Response body is : ' + response.data.decode('utf-8'))

    def test_list_documents(self):
        """Test case for list_documents

        List documents
        """
        query_string = [('offset', 56),
                        ('count', 56),
                        ('order_by', 'order_by_example'),
                        ('descend', true),
                        ('keywords', 'keywords_example')]
        response = self.client.open(
            '/v1/api/dataset/{dataset_id}/documents'.format(dataset_id='dataset_id_example'),
            method='GET',
            query_string=query_string)
        self.assert200(response,
                       'Response body is : ' + response.data.decode('utf-8'))

    def test_start_parsing_document(self):
        """Test case for start_parsing_document

        Start parsing a document
        """
        response = self.client.open(
            '/v1/api/dataset/{dataset_id}/documents/{document_id}/parse'.format(dataset_id='dataset_id_example', document_id='document_id_example'),
            method='POST')
        self.assert200(response,
                       'Response body is : ' + response.data.decode('utf-8'))

    def test_start_parsing_multiple_documents(self):
        """Test case for start_parsing_multiple_documents

        Start parsing multiple documents
        """
        body = DocumentsParseBody()
        response = self.client.open(
            '/v1/api/dataset/{dataset_id}/documents/parse'.format(dataset_id='dataset_id_example'),
            method='POST',
            data=json.dumps(body),
            content_type='application/json')
        self.assert200(response,
                       'Response body is : ' + response.data.decode('utf-8'))

    def test_update_document_details(self):
        """Test case for update_document_details

        Update document details
        """
        body = UpdateDocumentRequest()
        response = self.client.open(
            '/v1/api/dataset/{dataset_id}/documents/{document_id}'.format(dataset_id='dataset_id_example', document_id='document_id_example'),
            method='PUT',
            data=json.dumps(body),
            content_type='application/json')
        self.assert200(response,
                       'Response body is : ' + response.data.decode('utf-8'))

    def test_upload_documents(self):
        """Test case for upload_documents

        Upload documents
        """
        data = dict(files='files_example')
        response = self.client.open(
            '/v1/api/dataset/{dataset_id}/documents'.format(dataset_id='dataset_id_example'),
            method='POST',
            data=data,
            content_type='multipart/form-data')
        self.assert200(response,
                       'Response body is : ' + response.data.decode('utf-8'))


if __name__ == '__main__':
    import unittest
    unittest.main()
