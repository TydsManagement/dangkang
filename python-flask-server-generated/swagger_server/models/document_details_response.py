# coding: utf-8

from __future__ import absolute_import
from datetime import date, datetime  # noqa: F401

from typing import List, Dict  # noqa: F401

from swagger_server.models.base_model_ import Model
from swagger_server.models.document_details import DocumentDetails  # noqa: F401,E501
from swagger_server import util


class DocumentDetailsResponse(Model):
    """NOTE: This class is auto generated by the swagger code generator program.

    Do not edit the class manually.
    """
    def __init__(self, code: int=None, data: DocumentDetails=None, message: str=None):  # noqa: E501
        """DocumentDetailsResponse - a model defined in Swagger

        :param code: The code of this DocumentDetailsResponse.  # noqa: E501
        :type code: int
        :param data: The data of this DocumentDetailsResponse.  # noqa: E501
        :type data: DocumentDetails
        :param message: The message of this DocumentDetailsResponse.  # noqa: E501
        :type message: str
        """
        self.swagger_types = {
            'code': int,
            'data': DocumentDetails,
            'message': str
        }

        self.attribute_map = {
            'code': 'code',
            'data': 'data',
            'message': 'message'
        }
        self._code = code
        self._data = data
        self._message = message

    @classmethod
    def from_dict(cls, dikt) -> 'DocumentDetailsResponse':
        """Returns the dict as a model

        :param dikt: A dict.
        :type: dict
        :return: The DocumentDetailsResponse of this DocumentDetailsResponse.  # noqa: E501
        :rtype: DocumentDetailsResponse
        """
        return util.deserialize_model(dikt, cls)

    @property
    def code(self) -> int:
        """Gets the code of this DocumentDetailsResponse.


        :return: The code of this DocumentDetailsResponse.
        :rtype: int
        """
        return self._code

    @code.setter
    def code(self, code: int):
        """Sets the code of this DocumentDetailsResponse.


        :param code: The code of this DocumentDetailsResponse.
        :type code: int
        """

        self._code = code

    @property
    def data(self) -> DocumentDetails:
        """Gets the data of this DocumentDetailsResponse.


        :return: The data of this DocumentDetailsResponse.
        :rtype: DocumentDetails
        """
        return self._data

    @data.setter
    def data(self, data: DocumentDetails):
        """Sets the data of this DocumentDetailsResponse.


        :param data: The data of this DocumentDetailsResponse.
        :type data: DocumentDetails
        """

        self._data = data

    @property
    def message(self) -> str:
        """Gets the message of this DocumentDetailsResponse.


        :return: The message of this DocumentDetailsResponse.
        :rtype: str
        """
        return self._message

    @message.setter
    def message(self, message: str):
        """Sets the message of this DocumentDetailsResponse.


        :param message: The message of this DocumentDetailsResponse.
        :type message: str
        """

        self._message = message