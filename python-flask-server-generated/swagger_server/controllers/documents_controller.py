import connexion
import six

from swagger_server.models.document_details_response import DocumentDetailsResponse  # noqa: E501
from swagger_server.models.document_parsing_status_response import DocumentParsingStatusResponse  # noqa: E501
from swagger_server.models.documents_parse_body import DocumentsParseBody  # noqa: E501
from swagger_server.models.list_documents_response import ListDocumentsResponse  # noqa: E501
from swagger_server.models.success_response import SuccessResponse  # noqa: E501
from swagger_server.models.update_document_request import UpdateDocumentRequest  # noqa: E501
from swagger_server.models.upload_documents_response import UploadDocumentsResponse  # noqa: E501
from swagger_server import util


def delete_document(dataset_id, document_id):  # noqa: E501
    """Delete document

     # noqa: E501

    :param dataset_id: 
    :type dataset_id: str
    :param document_id: 
    :type document_id: str

    :rtype: SuccessResponse
    """
    return 'do some magic!'


def download_document(dataset_id, document_id):  # noqa: E501
    """Download document

     # noqa: E501

    :param dataset_id: 
    :type dataset_id: str
    :param document_id: 
    :type document_id: str

    :rtype: str
    """
    return 'do some magic!'


def get_document_parsing_status(dataset_id, document_id):  # noqa: E501
    """Show parsing status of document

     # noqa: E501

    :param dataset_id: 
    :type dataset_id: str
    :param document_id: 
    :type document_id: str

    :rtype: DocumentParsingStatusResponse
    """
    return 'do some magic!'


def list_documents(dataset_id, offset=None, count=None, order_by=None, descend=None, keywords=None):  # noqa: E501
    """List documents

     # noqa: E501

    :param dataset_id: 
    :type dataset_id: str
    :param offset: 
    :type offset: int
    :param count: 
    :type count: int
    :param order_by: 
    :type order_by: str
    :param descend: 
    :type descend: bool
    :param keywords: 
    :type keywords: str

    :rtype: ListDocumentsResponse
    """
    return 'do some magic!'


def start_parsing_document(dataset_id, document_id):  # noqa: E501
    """Start parsing a document

     # noqa: E501

    :param dataset_id: 
    :type dataset_id: str
    :param document_id: 
    :type document_id: str

    :rtype: SuccessResponse
    """
    return 'do some magic!'


def start_parsing_multiple_documents(body, dataset_id):  # noqa: E501
    """Start parsing multiple documents

     # noqa: E501

    :param body: 
    :type body: dict | bytes
    :param dataset_id: 
    :type dataset_id: str

    :rtype: SuccessResponse
    """
    if connexion.request.is_json:
        body = DocumentsParseBody.from_dict(connexion.request.get_json())  # noqa: E501
    return 'do some magic!'


def update_document_details(body, dataset_id, document_id):  # noqa: E501
    """Update document details

     # noqa: E501

    :param body: 
    :type body: dict | bytes
    :param dataset_id: 
    :type dataset_id: str
    :param document_id: 
    :type document_id: str

    :rtype: DocumentDetailsResponse
    """
    if connexion.request.is_json:
        body = UpdateDocumentRequest.from_dict(connexion.request.get_json())  # noqa: E501
    return 'do some magic!'


def upload_documents(files, dataset_id):  # noqa: E501
    """Upload documents

     # noqa: E501

    :param files: 
    :type files: List[strstr]
    :param dataset_id: 
    :type dataset_id: str

    :rtype: UploadDocumentsResponse
    """
    return 'do some magic!'
