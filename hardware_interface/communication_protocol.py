"""
communication_protocol.py
    Communication protocol module
Methods
    CarStatus
    RaceCommunicationProtocol
    RaceCar
"""

__version__ = '0.0.0'
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
    get_dict
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
        return json.dumps(self.get_dict())

    def get_dict(self):
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
    car_status: CarStatus
        The status of the car
    id: int
        Car ID
    ...
    Methods
    -------
    serialise
    deserialise
    get_dict
    """

    topic: str = "race/car"
    _msg_name: str = "RaceCar"

    def __init__(self, header: str = "", msg_name: str = "", qos: int = 0, car_status: CarStatus = CarStatus.DEFAULT,
                 id: int = -1):
        """Constructor for RaceCar"""
        super().__init__(header=header, msg_name=msg_name if msg_name else self._msg_name, qos=qos)
        self.car_status = car_status
        self.id = id

    @property
    def id(self):
        """
        description
            Car ID number
        type
            int
        default
            -1
        """
        return self._id

    @id.setter
    def id(self, value):
        if isinstance(value, int) or isinstance(value, float):
            self._id = value if value >= 0 else -1
        elif isinstance(value, str):
            try:
                self._id = int(value)
            except ValueError:
                self._id = -1
        else:
            self._id = -1

    @property
    def car_status(self):
        return self._car_status

    @car_status.setter
    def car_status(self, value):
        if isinstance(value, CarStatus):
            self._car_status = value
        else:
            self._car_status = CarStatus.DEFAULT

    def serialise(self):
        """Serialise communication ready for MQTT publishing"""
        return json.dumps(self.get_dict())

    def get_dict(self):
        """Returns dictionary representation of the class"""
        d = super().get_dict()
        d.update({
            "STS": self.car_status.value,
            "ID": self.id,
        })
        return d

    def deserialise(self, mqtt_msg: str):
        """Deserialise received MQTT message"""
        super().deserialise(mqtt_msg)
        json_data = json.loads(mqtt_msg)
        if "STS" in json_data:
            self.car_status = CarStatus(json_data.get("STS"))
        if "ID" in json_data:
            self.id = json_data.get("ID")

