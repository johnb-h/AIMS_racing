"""
__init__.py
    Hardware and communication code for racing
Methods
    communication_protocol
        CarStatus
        RaceCommunicationProtocol
        RaceCar
    mqtt_communication
        MQTTClient
Dependencies
    paho-mqtt
"""

__version__ = '0.0.0'
__project__ = 'AIMS_racing'
__tested__ = 'N'

from .communication_protocol import RaceCar, CarStatus, RaceCommunicationProtocol
from .mqtt_communication import MQTTClient