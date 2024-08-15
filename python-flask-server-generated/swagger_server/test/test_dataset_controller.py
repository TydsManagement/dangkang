# coding: utf-8

from __future__ import absolute_import

from flask import json
from six import BytesIO

from swagger_server.models.dataset_body import DatasetBody  # noqa: E501
from swagger_server.models.dataset_details_response import DatasetDetailsResponse  # noqa: E501
from swagger_server.models.dataset_list_response import DatasetListResponse  # noqa: E501
from swagger_server.models.dataset_response import DatasetResponse  # noqa: E501
from swagger_server.models.success_response import SuccessResponse  # noqa: E501
from swagger_server.models.update_dataset_request import UpdateDatasetRequest  # noqa: E501
from swagger_server.test import BaseTestCase


class TestDatasetController(BaseTestCase):
    """DatasetController integration test stubs"""

    def test_create_dataset(self):
        """Test case for create_dataset

        Create dataset
        """
        body = DatasetBody()
        response = self.client.open(
            '/v1/api/dataset',
            method='POST',
            data=json.dumps(body),
            content_type='application/json')
        self.assert200(response,
                       'Response body is : ' + response.data.decode('utf-8'))

    def test_delete_dataset(self):
        """Test case for delete_dataset

        Delete dataset
        """
        response = self.client.open(
            '/v1/api/dataset/{dataset_id}'.format(dataset_id='dataset_id_example'),
            method='DELETE')
        self.assert200(response,
                       'Response body is : ' + response.data.decode('utf-8'))

    def test_get_dataset_details(self):
        """Test case for get_dataset_details

        Get dataset details
        """
        response = self.client.open(
            '/v1/api/dataset/{dataset_id}'.format(dataset_id='dataset_id_example'),
            method='GET')
        self.assert200(response,
                       'Response body is : ' + response.data.decode('utf-8'))

    def test_get_dataset_list(self):
        """Test case for get_dataset_list

        Get dataset list
        """
        response = self.client.open(
            '/v1/api/dataset',
            method='GET')
        self.assert200(response,
                       'Response body is : ' + response.data.decode('utf-8'))

    def test_update_dataset_details(self):
        """Test case for update_dataset_details

        Update dataset details
        """
        body = UpdateDatasetRequest()
        response = self.client.open(
            '/v1/api/dataset/{dataset_id}'.format(dataset_id='dataset_id_example'),
            method='PUT',
            data=json.dumps(body),
            content_type='application/json')
        self.assert200(response,
                       'Response body is : ' + response.data.decode('utf-8'))


if __name__ == '__main__':
    import unittest
    unittest.main()
