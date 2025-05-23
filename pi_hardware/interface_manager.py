"""
interface_manager.py
    Initialises a manager to control the IO of the PI
"""

# Standard Packages
import json
import time
import sys
import threading

# Non-Standard Packages
import pandas as pd

sys.path.append("..")

# Relative Imports
from hardware_interface import MQTTClient, RaceCar, LedCtrl, LedMode
from gpio_controller import GPIOController, LightButton


class InterfaceManager:
    """
    Manages the IO of the Pi

    ...
    Attributes
    ----------
    buttons: list[LightButton]
        Controls the interacting of buttons on the Pi
    mqtt_client: MQTTClient
        Initialised MQTT client

    ...
    Methods
    -------
    hardware_loop
    mqtt_loop
    stop
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
        self._init_sequence()
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

    def _init_sequence(self):
        """Init light sequence"""
        sleep_time: float = 0.12
        while sleep_time > 0:
            for button in self.buttons.values():
                button.toggle_light()
                time.sleep(sleep_time)
                button.toggle_light()
            for i in range(3):
                time.sleep(sleep_time)
                self._all_on()
                time.sleep(sleep_time)
                self._all_off()

            sleep_time -= 0.04

    def hardware_loop(self):
        """Loops through checking the buttons and toggle lights where needed"""
        while not self._kill:
            for key, button in self.buttons.items():
                if button.check_trigger():
                    button.toggle_light()
                    msg = RaceCar(id=key)
                    self.mqtt_client.publish_message(RaceCar.topic, msg.serialise())
                    while button.check_trigger():
                        time.sleep(0.05)
            time.sleep(0.05)

    def mqtt_loop(self):
        """Monitors the MQTT client for messages and deals with them appropriately"""
        self.mqtt_client.start_loop()
        print("Listen Thread started")
        self.mqtt_client.subscribe(LedCtrl.topic)
        while not self._kill:
            while not self.mqtt_client.queue_empty():
                topic, msg = self.mqtt_client.pop_queue()
                print(f"{topic}: {msg}")
                if topic == LedCtrl.topic:
                    led_msg = LedCtrl()
                    led_msg.deserialise(msg)
                    if led_msg.mode == LedMode.ALL_ON:
                        self._all_on()
                    elif led_msg.mode == LedMode.INIT:
                        self._init_sequence()
                    elif led_msg.mode == LedMode.RACE_START:
                        self._race_start()
                    else:
                        self._all_off()

            time.sleep(0.05)
        self.mqtt_client.stop_loop()

    def _all_on(self):
        """Turns on all the LEDs"""
        for button in self.buttons.values():
            button.on()

    def _all_off(self):#
        """Turns off all the LEDs"""
        for button in self.buttons.values():
            button.off()

    def _race_start(self):
        """Start of race light sequence"""
        row_len: int = len(self.buttons)
        row_end = row_len // 2
        # Countdown
        for i in range(row_end):
            self.buttons[str(i)].on()
            time.sleep(1)
        for i in range(row_end, row_len):
            self.buttons[str(i)].on()
        time.sleep(1)

        # Row flashes
        for _ in range(10):
            for i in range(row_end):
                self.buttons[str(i)].toggle_light()
            time.sleep(0.04)
            for i in range(row_end, row_len):
                self.buttons[str(i)].toggle_light()
            time.sleep(0.04)

        time.sleep(1)
        self._all_off()

    def stop(self):
        """Stops manager"""
        self._kill = True
        time.sleep(0.05)


if __name__ == '__main__':
    """Demo script"""
    with open("../configs/mqtt_config.json", "r", encoding="utf-8") as f:
        mqtt_config = json.load(f)
        f.close()
    mqtt_client = MQTTClient(mqtt_config=mqtt_config, verbose=True)
    mqtt_client.connect()
    manager = InterfaceManager(mqtt_client=mqtt_client,
                               pin_configuration="pins.json")
    time.sleep(0.5)
    listen_thread = threading.Thread(target=manager.mqtt_loop)
    listen_thread.start()
    hardware_thread = threading.Thread(target=manager.hardware_loop)
    hardware_thread.start()
    while True:
        try:
            time.sleep(0.05)
        except KeyboardInterrupt:
            manager.stop()
            listen_thread.join()
            hardware_thread.join()
            mqtt_client.disconnect()

