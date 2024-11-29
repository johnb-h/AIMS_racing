"""
mqtt_communication.py
    MQTT communication file
"""

__version__ = '0.0.0'
__organization__ = 'MarineAI'
__project__ = 'AIMS_racing'
__tested__ = 'N'

import paho.mqtt.client as mqtt


class MQTTClient:
    """
    MQTT Client class
    """

    def __int__(self, config: dict):
        """
        MQTTClient constructor
        :param config: Configuration dictionary
        :type config: dict
        """
        self._client = mqtt.Client()

    def publish_message(self, topic: str, message: str):
        """Publishes a message to MQTT Broker"""
        raise NotImplementedError
