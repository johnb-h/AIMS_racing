"""
communication_protocol.py
    Communication protocol module
Methods
    CarStatus
    RaceCommunicationProtocol
    LedMode
    RaceCar
    LedCtrl
"""

# Standard Packages
import json
from enum import Enum

class CarStatus(Enum):
    """Car status Enum for cars"""
    DEFAULT = 0
    PRERACE = 1
    INRACE = 2
    POSTRACE = 3

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

    ...
    Methods
    -------
    serialise
    get_dict
    deserialise
    """

    def __init__(self, header: str = "", msg_name: str = ""):
        """Constructor for RaceCommunicationProtocol"""
        self.header = header
        self.msg_name = msg_name

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

    def serialise(self) -> str:
        """Serialise communication ready for MQTT publishing"""
        return json.dumps(self.get_dict())

    def get_dict(self) -> dict:
        """Returns dictionary representation of the class"""
        return {
            "MNAME": self.msg_name
        }

    def deserialise(self, mqtt_msg: str) -> None:
        """Deserialise received MQTT message"""
        json_msg: dict = json.loads(mqtt_msg)
        if "MNAME" in json_msg:
            self.msg_name = json_msg.get("MNAME")

class LedMode(Enum):
    """LED mode Enum"""
    ALL_OFF = 0
    ALL_ON = 1
    INIT = 2
    RACE_START = 3
    DEFAULT = 0

class LedCtrl(RaceCommunicationProtocol):
    """
    Control for LEDs

    Attributes
    ----------
    mode: LedMode

    Methods
    -------
    serialise
    deserialise
    get_dict
    """

    topic: str = "race/led"
    _msg_name: str = "LedCtrl"

    def __init__(self, header: str = "", msg_name: str = "", mode: LedMode = LedMode.DEFAULT):
        """Constructor for LedCtrl"""
        super().__init__(header=header, msg_name=msg_name if msg_name else self._msg_name)
        self.mode = mode

    @property
    def mode(self):
        """Led Mode Enum"""
        return self._mode

    @mode.setter
    def mode(self, value):
        if isinstance(value, LedMode):
            self._mode = value
        elif isinstance(value, int) and value in [e.value for e in LedMode]:
            self._mode = LedMode(value)
        else:
            self._mode = LedMode.DEFAULT

    def serialise(self):
        """Serialise communication ready for MQTT publishing"""
        return json.dumps(self.get_dict())

    def get_dict(self):
        """Returns dictionary representation of the class"""
        d = super().get_dict()
        d.update({
            "MODE": self.mode.value
        })
        return d

    def deserialise(self, mqtt_msg: str):
        """Deserialise received MQTT message"""
        super().deserialise(mqtt_msg)
        json_data = json.loads(mqtt_msg)
        if "MODE" in json_data:
            try:
                self.mode = LedMode(json_data.get("MODE"))
            except ValueError:
                self.mode = LedMode.DEFAULT

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

    def __init__(self, header: str = "", msg_name: str = "", car_status: CarStatus = CarStatus.DEFAULT,
                 id: int = -1):
        """Constructor for RaceCar"""
        super().__init__(header=header, msg_name=msg_name if msg_name else self._msg_name)
        self.car_status = car_status
        self.id = id

    @property
    def id(self):
        """Car ID number"""
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
            "ID": self.id
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

