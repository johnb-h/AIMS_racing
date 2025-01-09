"""
gpio_controller.py
    Interfaces the GPIO ports on the Raspberry Pi for communication via communication protocol
"""

__version__ = '0.0.0'
__project__ = 'AIMS_racing'
__tested__ = 'N'

import RPi.GPIO as GPIO
import time
from threading import Thread
from hardware_interface import MQTTClient
from hardware_interface import RaceCar, CarStatus, RaceCommunicationProtocol


class GPIOController:
    """
    GPIO Controller for the Raspberry Pi

    ...
    Attributes
    ----------
    led_pins: list[int]
        A list of LED pins
    button_pins: list[int]
        A list of button pins
    mqtt_client: MQTTClient
        MQTT Client to communicate with the Raspberry Pi
    running: bool
        Whether or not the Raspberry Pi should be running

    ...
    Methods
    ----------


    """
    def __init__(self, led_pins: list[int], button_pins: list[int], mqtt_client: MQTTClient):
        """
        Initialise the GPIOController.

        :param led_pins: List of GPIO pins connected to LEDs.
        :type led_pins: list[int]
        :param button_pins: List of GPIO pins connected to Buttons.
        :type button_pins: list[int]
        :param mqtt_client: An instance of your MQTT client.
        :type mqtt_client: MQTTClient
        """
        # Unpack args
        self.led_pins = led_pins
        self.button_pins = button_pins
        self.mqtt_client = mqtt_client

        # Initialise Tracker Var
        self.running: bool = True

        # Setup GPIO mode
        GPIO.setmode(GPIO.BCM)

        # Initialize LED pins as output
        for pin in led_pins:
            GPIO.setup(pin, GPIO.OUT)
            GPIO.output(pin, GPIO.LOW)

        # Initialize Button pins as input with pull-up resistor
        for pin in button_pins:
            GPIO.setup(pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)

    def set_led(self, led_index: int, state: bool) -> None:
        """
        Set the state of an LED.

        :param led_index: Index of the LED in led_pins list.
        :type led_index: int
        :param state: True to turn on, False to turn off.
        :type state: bool
        :return: None
        """
        if 0 <= led_index < len(self.led_pins):
            GPIO.output(self.led_pins[led_index], GPIO.HIGH if state else GPIO.LOW)
        else:
            raise ValueError("Invalid LED index")

    def read_button(self, button_index: int) -> bool:
        """
        Read the state of a button.

        :param button_index: Index of the button in button_pins list.
        :type button_index: int
        :return: True if button is pressed, False otherwise.
        :rtype: bool
        """
        if 0 <= button_index < len(self.button_pins):
            return not GPIO.input(self.button_pins[button_index])  # Button pressed = LOW
        else:
            raise ValueError("Invalid button index")

    def monitor_buttons(self):
        """Monitor buttons and publish their status via MQTT."""
        while self.running:
            for i, pin in enumerate(self.button_pins):
                if not GPIO.input(pin):  # Button pressed
                    message = CustomMessage(type="button_press", id=i, state=True)
                    self.mqtt_client.publish_message(message)
                    time.sleep(0.1)  # Debounce delay

    def stop(self):
        """Stop the GPIOController and clean up GPIO settings."""
        self.running = False
        GPIO.cleanup()


# Example Usage
if __name__ == "__main__":
    # Define GPIO pins for LEDs and Buttons
    led_pins = [17, 27, 22]
    button_pins = [5, 6, 13]

    config: dict = {
        'broker': '127.0.0.1',
        'port': 1883,
        'username': '',
        'password': '',
        'keepalive': 60
    }

    # Connect to broker
    client = MQTTClient(mqtt_config=config, verbose=False)
    client.connect()

    # Initialize GPIOController
    gpio_controller = GPIOController(led_pins, button_pins, client)

    try:
        # Start monitoring buttons in a separate thread
        button_thread = Thread(target=gpio_controller.monitor_buttons)
        button_thread.start()

        # Example: Control LEDs via code
        gpio_controller.set_led(0, True)  # Turn on LED 0
        time.sleep(1)
        gpio_controller.set_led(0, False)  # Turn off LED 0

    except KeyboardInterrupt:
        print("Stopping GPIO Controller")

    finally:
        gpio_controller.stop()
        client.stop_loop()
        client.disconnect()
        time.sleep(1)
