# coding: utf-8

from __future__ import absolute_import
from datetime import date, datetime  # noqa: F401

from typing import List, Dict  # noqa: F401

from swagger_server.models.base_model_ import Model
from swagger_server import util


class DatasetDetails(Model):
    """NOTE: This class is auto generated by the swagger code generator program.

    Do not edit the class manually.
    """
    def __init__(self, id: str=None, name: str=None, description: str=None, created_by: str=None, create_time: int=None, update_time: int=None):  # noqa: E501
        """DatasetDetails - a model defined in Swagger

        :param id: The id of this DatasetDetails.  # noqa: E501
        :type id: str
        :param name: The name of this DatasetDetails.  # noqa: E501
        :type name: str
        :param description: The description of this DatasetDetails.  # noqa: E501
        :type description: str
        :param created_by: The created_by of this DatasetDetails.  # noqa: E501
        :type created_by: str
        :param create_time: The create_time of this DatasetDetails.  # noqa: E501
        :type create_time: int
        :param update_time: The update_time of this DatasetDetails.  # noqa: E501
        :type update_time: int
        """
        self.swagger_types = {
            'id': str,
            'name': str,
            'description': str,
            'created_by': str,
            'create_time': int,
            'update_time': int
        }

        self.attribute_map = {
            'id': 'id',
            'name': 'name',
            'description': 'description',
            'created_by': 'created_by',
            'create_time': 'create_time',
            'update_time': 'update_time'
        }
        self._id = id
        self._name = name
        self._description = description
        self._created_by = created_by
        self._create_time = create_time
        self._update_time = update_time

    @classmethod
    def from_dict(cls, dikt) -> 'DatasetDetails':
        """Returns the dict as a model

        :param dikt: A dict.
        :type: dict
        :return: The DatasetDetails of this DatasetDetails.  # noqa: E501
        :rtype: DatasetDetails
        """
        return util.deserialize_model(dikt, cls)

    @property
    def id(self) -> str:
        """Gets the id of this DatasetDetails.


        :return: The id of this DatasetDetails.
        :rtype: str
        """
        return self._id

    @id.setter
    def id(self, id: str):
        """Sets the id of this DatasetDetails.


        :param id: The id of this DatasetDetails.
        :type id: str
        """

        self._id = id

    @property
    def name(self) -> str:
        """Gets the name of this DatasetDetails.


        :return: The name of this DatasetDetails.
        :rtype: str
        """
        return self._name

    @name.setter
    def name(self, name: str):
        """Sets the name of this DatasetDetails.


        :param name: The name of this DatasetDetails.
        :type name: str
        """

        self._name = name

    @property
    def description(self) -> str:
        """Gets the description of this DatasetDetails.


        :return: The description of this DatasetDetails.
        :rtype: str
        """
        return self._description

    @description.setter
    def description(self, description: str):
        """Sets the description of this DatasetDetails.


        :param description: The description of this DatasetDetails.
        :type description: str
        """

        self._description = description

    @property
    def created_by(self) -> str:
        """Gets the created_by of this DatasetDetails.


        :return: The created_by of this DatasetDetails.
        :rtype: str
        """
        return self._created_by

    @created_by.setter
    def created_by(self, created_by: str):
        """Sets the created_by of this DatasetDetails.


        :param created_by: The created_by of this DatasetDetails.
        :type created_by: str
        """

        self._created_by = created_by

    @property
    def create_time(self) -> int:
        """Gets the create_time of this DatasetDetails.


        :return: The create_time of this DatasetDetails.
        :rtype: int
        """
        return self._create_time

    @create_time.setter
    def create_time(self, create_time: int):
        """Sets the create_time of this DatasetDetails.


        :param create_time: The create_time of this DatasetDetails.
        :type create_time: int
        """

        self._create_time = create_time

    @property
    def update_time(self) -> int:
        """Gets the update_time of this DatasetDetails.


        :return: The update_time of this DatasetDetails.
        :rtype: int
        """
        return self._update_time

    @update_time.setter
    def update_time(self, update_time: int):
        """Sets the update_time of this DatasetDetails.


        :param update_time: The update_time of this DatasetDetails.
        :type update_time: int
        """

        self._update_time = update_time
