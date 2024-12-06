"""
mqtt_communication.py
    MQTT communication file
Methods
    MQTTClient
"""

__version__ = '0.0.0'
__project__ = 'AIMS_racing'
__tested__ = 'N'

# Standard Packages
import time

# Non-Standard Packages
import paho.mqtt.client as mqtt


class MQTTClient:
    """
    MQTT Client class

    ...
    Methods
    -------
    connect
    publish_message
    subscribe
    start_loop
    stop_loop
    disconnect

    ...
    Attributes
    ----------
    config: dict
        Configuration dictionary for the broker communication

    """

    def __init__(self, mqtt_config: dict, verbose: bool = False) -> None:
        """
        MQTTClient constructor
        :param mqtt_config: Configuration dictionary
        :type mqtt_config: dict
        :param verbose: Allow verbose output from MQTT communication
        :type verbose: bool
        """
        self._client = mqtt.Client()
        self.config = mqtt_config
        self._setup_client()
        self._verbose = verbose

    def _setup_client(self):
        """
        Sets up the MQTT client with optional authentication and callbacks
        """
        if 'username' in self.config and 'password' in self.config:
            self._client.username_pw_set(self.config['username'], self.config['password'])

        # Assign default callbacks
        self._client.on_connect = self._on_connect
        self._client.on_message = self._on_message

    def _on_connect(self, client, userdata, flags, rc):
        """
        Callback for when the client receives a CONNACK response from the broker
        :param client: The client instance
        :param userdata: User-defined data of any type
        :param flags: Response flags sent by the broker
        :param rc: Connection result
        """
        if self._verbose:
            if rc == 0:
                print("Connected to broker successfully")
            else:
                print(f"Failed to connect to broker. Return code: {rc}")

    def _on_message(self, client, userdata, msg):
        """
        Callback for when a PUBLISH message is received from the broker
        :param client: The client instance
        :param userdata: User-defined data of any type
        :param msg: An instance of MQTTMessage
        """
        if self._verbose:
            print(f"Message received on topic {msg.topic}: {msg.payload.decode()}")

    def connect(self):
        """
        Connects to the MQTT broker
        """
        self._client.connect(self.config['broker'], self.config['port'], self.config.get('keepalive', 60))

    def publish_message(self, topic: str, message: str):
        """
        Publishes a message to the MQTT broker
        :param topic: Topic to publish the message to
        :param message: The message payload
        """
        result = self._client.publish(topic, message)
        status = result[0]
        if self._verbose:
            if status == 0:
                print(f"Message sent to topic {topic}")
            else:
                print(f"Failed to send message to topic {topic}")

    def subscribe(self, topic: str):
        """
        Subscribes to a topic
        :param topic: The topic to subscribe to
        """
        self._client.subscribe(topic)
        if self._verbose:
            print(f"Subscribed to topic {topic}")

    def start_loop(self):
        """
        Starts the MQTT client loop to process network traffic
        """
        self._client.loop_start()

    def stop_loop(self):
        """
        Stops the MQTT client loop
        """
        self._client.loop_stop()

    def disconnect(self):
        """
        Disconnects from the MQTT broker
        """
        self._client.disconnect()


if __name__ == '__main__':
    """Example connection"""
    config = {
        'broker': '127.0.0.1',
        'port': 1883,
        'username': '',
        'password': '',
        'keepalive': 60
    }

    client = MQTTClient(config)
    client.connect()

    # Start the loop in a separate thread to handle callbacks
    client.start_loop()

    # Publish and subscribe
    client.subscribe("test/topic")
    client.publish_message("test/topic", "Hello MQTT")

    time.sleep(5)

    # Clean up
    client.stop_loop()
    client.disconnect()

