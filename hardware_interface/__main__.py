"""
__main__.py
    description
Methods

Attributes

Dependencies

"""

__version__ = '0.0.0'
__project__ = 'AIMS_racing'
__tested__ = 'N'

# Standard Packages
import time

# Module Imports
from hardware_interface import MQTTClient, RaceCar, CarStatus

if __name__ == '__main__':
    """Example communication using coms protocol and local mqtt client instance"""
    config: dict = {
        'broker': '127.0.0.1',
        'port': 1883,
        'username': '',
        'password': '',
        'keepalive': 60
    }

    # Connect to broker
    client = MQTTClient(mqtt_config=config, verbose=True)
    client.connect()

    # Start the loop in a separate thread to handle callbacks
    client.start_loop()

    # Publish and subscribe
    client.subscribe(RaceCar.topic)

    # Creat car objet and publish
    car: RaceCar = RaceCar(id=1, car_status=CarStatus.INRACE)
    client.publish_message(RaceCar.topic, car.serialise())

    # Wait to receive communication
    time.sleep(1)

    # Clean up
    client.stop_loop()
    client.disconnect()