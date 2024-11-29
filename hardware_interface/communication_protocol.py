"""
communication_protocol.py
    description
Methods

Attributes

Dependencies

"""

__version__ = '0.0.0'
__organization__ = 'MarineAI'
__project__ = 'AIMS_racing'
__tested__ = 'N'

# Standard Packages
import json
from enum import Enum


class CarStatus(Enum):
    """Car status Enum for cars"""
    DEFAULT: int = 0
    PRERACE: int = 1
    INRACE: int = 2
    POSTRACE: int = 3



class RaceCommunicationProtocol:
    """
    Base class for RaceCommunicationProtocol

    ...
    Attributes
    ----------
    header: str
        Topic header for MQTT communication
    msg_name: str
        Message name for MQTT communication
    qos: int
        Quality of service

    ...
    Methods
    -------
    serialise
    deserialise
    """

    def __init__(self, header: str = "", msg_name: str = "", qos: int = 0):
        """Constructor for RaceCommunicationProtocol"""
        self.header = header
        self.msg_name = msg_name
        self.qos = qos

    @property
    def header(self) -> str:
        return self._header

    @header.setter
    def header(self, value: str):
        if isinstance(value, str):
            self._header = value

    @property
    def msg_name(self) -> str:
        return self._msg_name

    @msg_name.setter
    def msg_name(self, value: str):
        if isinstance(value, str):
            self._msg_name = value

    @property
    def qos(self) -> int:
        return self._qos

    @qos.setter
    def qos(self, value: int):
        if isinstance(value, int):
            self._qos = value
        else:
            self._qos = 0

    def serialise(self):
        """Serialise communication ready for MQTT publishing"""
        return json.dumps(self._get_dict())

    def _get_dict(self):
        """Returns dictionary representation of the class"""
        return {
            "MNAME": self.msg_name,
            "QOS": self.qos,
        }

    def deserialise(self, mqtt_msg: str):
        """Deserialise received MQTT message"""
        json_msg: dict = json.loads(mqtt_msg)
        if "MNAME" in json_msg:
            self.msg_name = json_msg.get("MNAME")
        if "QOS" in json_msg:
            self.qos = json_msg.get("QOS")


class RaceCar(RaceCommunicationProtocol):
    """
    Race Car message class

    ...
    Attributes
    ----------

    ...
    Methods
    -------

    """

    def __init__(self, header: str = "", msg_name: str = "", qos: int = 0, car_status: CarStatus):
        """Constructor for RaceCar"""
        super().__init__(header=header, msg_name=msg_name, qos=qos)
        self.car_status = car_status

    @property
    def car_status(self):
        return self._car_status

    @car_status.setter
    def car_status(self, value):
        if isinstance(value, CarStatus):
            self._car_status = value
        else:
            self._car_status = CarStatus.DEFAULT