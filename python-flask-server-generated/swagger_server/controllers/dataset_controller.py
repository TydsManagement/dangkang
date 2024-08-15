import connexion
import six

from swagger_server.models.dataset_body import DatasetBody  # noqa: E501
from swagger_server.models.dataset_details_response import DatasetDetailsResponse  # noqa: E501
from swagger_server.models.dataset_list_response import DatasetListResponse  # noqa: E501
from swagger_server.models.dataset_response import DatasetResponse  # noqa: E501
from swagger_server.models.success_response import SuccessResponse  # noqa: E501
from swagger_server.models.update_dataset_request import UpdateDatasetRequest  # noqa: E501
from swagger_server import util


def create_dataset(body):  # noqa: E501
    """Create dataset

     # noqa: E501

    :param body: 
    :type body: dict | bytes

    :rtype: DatasetResponse
    """
    if connexion.request.is_json:
        body = DatasetBody.from_dict(connexion.request.get_json())  # noqa: E501
    return 'do some magic!'


def delete_dataset(dataset_id):  # noqa: E501
    """Delete dataset

     # noqa: E501

    :param dataset_id: 
    :type dataset_id: str

    :rtype: SuccessResponse
    """
    return 'do some magic!'


def get_dataset_details(dataset_id):  # noqa: E501
    """Get dataset details

     # noqa: E501

    :param dataset_id: 
    :type dataset_id: str

    :rtype: DatasetDetailsResponse
    """
    return 'do some magic!'


def get_dataset_list():  # noqa: E501
    """Get dataset list

     # noqa: E501


    :rtype: DatasetListResponse
    """
    return 'do some magic!'


def update_dataset_details(body, dataset_id):  # noqa: E501
    """Update dataset details

     # noqa: E501

    :param body: 
    :type body: dict | bytes
    :param dataset_id: 
    :type dataset_id: str

    :rtype: DatasetDetailsResponse
    """
    if connexion.request.is_json:
        body = UpdateDatasetRequest.from_dict(connexion.request.get_json())  # noqa: E501
    return 'do some magic!'
