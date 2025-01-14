"""
interface_manager.py
    Initialises a manager to control the IO of the PI
"""

__version__ = '0.0.0'
__organization__ = 'MarineAI'
__project__ = 'AIMS_racing'
__tested__ = 'N'

# Standard Packages
import json
import os
import time
import sys

sys.path.append("..")

# Non-Standard Packages
import pandas as pd

# Relative Imports
from hardware_interface import MQTTClient, RaceCar, CarStatus
from gpio_controller import GPIOController, LightButton


class InterfaceManager:
    """
    Manages the IO of the Pi

    ...
    Attributes
    ----------
    buttons: list[LightButton]
        Controls the interacting of buttons on the Pi

    ...
    Methods
    -------
    """

    def __init__(self, mqtt_client: MQTTClient, pin_configuration: str):
        """
        Initialises the interface manager
        :param mqtt_client: An initialized MQTT Client
        :type mqtt_client: MQTTClient
        :param pin_configuration: Pin configuration file path
        :type pin_configuration: str
        """
        with open(pin_configuration, "r", encoding="utf-8") as f:
            pin_setup = json.load(f)
            f.close()
        pins = self._fecth_pin_config(pin_setup)
        self._gpio_controller: GPIOController = GPIOController(led_pins=pins[0], button_pins=pins[1])
        self.buttons = {}

        for key in pin_setup['BUTTONS']:
            led_pin, trig_pin = pin_setup['BUTTONS'].get(key)
            self.buttons.update({key: LightButton(led_pin=led_pin, trigger_pin=trig_pin,
                                                  gpio_controller=self._gpio_controller)})
        self.init_sequence()
        self.mqtt_client = mqtt_client
        self._kill = False

    @staticmethod
    def _fecth_pin_config(pin_setup: dict) -> tuple[list[int], list[int]]:
        """
        Returns the list of pins for LEDs and buttons for GPIO
        :param pin_setup: Pin setup dictionary
        :type pin_setup: dict
        :return: List of pins for LEDs and buttons
        :rtype: tuple[list[int], list[int]]
        """
        df = pd.DataFrame(pin_setup["BUTTONS"])
        return df.iloc[0].tolist(), df.iloc[1].tolist()

    def init_sequence(self):
        """Init light sequence"""
        sleep_time: float = 0.5
        while sleep_time > 0:
            for button in self.buttons.values():
                button.toggle_light()
                time.sleep(sleep_time)
                button.toggle_light()
            sleep_time -= 0.1

    def loop(self):
        """Loops through checking the buttons and toggle lights where needed"""
        while not self._kill:
            for key, button in self.buttons.items():
                if button.check_trigger():
                    button.toggle_light()
                    msg = RaceCar(id=key)
                    self.mqtt_client.publish_message(RaceCar.topic, msg.serialise())
                    time.sleep(0.3)
            time.sleep(0.05)

    def stop(self):
        """Stops manager"""
        self._kill = True
        time.sleep(0.05)


if __name__ == '__main__':
    """Demo script"""
    with open("../configs/mqtt_config.json", "r", encoding="utf-8") as f:
        mqtt_config = json.load(f)
        f.close()
    mqtt_client = MQTTClient(mqtt_config=mqtt_config)
    mqtt_client.connect()
    manager = InterfaceManager(mqtt_client=mqtt_client,
                               pin_configuration="pins.json")
    manager.loop()
    while True:
        try:
            time.sleep(0.05)
        except KeyboardInterrupt:
            manager.stop()
            mqtt_client.disconnect()
