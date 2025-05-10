"""
dummy_car_selection.py
    Test script to run to check if botton integration is working for the game
Methods

Attributes

Dependencies

"""

__version__ = '0.0.0'
__organization__ = 'AIMS'
__project__ = 'AIMS_racing'
__tested__ = 'N'

# Standard Packages
import json
import time
import sys

sys.path.append("..")

# Relative Imports
from hardware_interface.mqtt_communication import MQTTClient
from hardware_interface.communication_protocol import RaceCar

if __name__ == '__main__':
    """Demo script"""
    with open("configs/mqtt_config.json", "r", encoding="utf-8") as f:
        mqtt_config = json.load(f)
        f.close()
    mqtt_client = MQTTClient(mqtt_config=mqtt_config)
    mqtt_client.connect()
    race_car = RaceCar(id=1)
    time.sleep(0.2)
    mqtt_client.publish_message(race_car.topic, race_car.serialise())
    print("Topic: ", race_car.topic, "\nMsg: ", race_car.serialise())
    time.sleep(0.2)
    mqtt_client.disconnect()