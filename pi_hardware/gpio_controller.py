"""
gpio_controller.py
    Interfaces the GPIO ports on the Raspberry Pi for communication via communication protocol
"""

__version__ = '0.0.0'
__project__ = 'AIMS_racing'
__tested__ = 'N'

import sys

sys.path.append('../')
import RPi.GPIO as GPIO
import time
from threading import Thread


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
    running: bool
        Whether or not the Raspberry Pi should be running

    ...
    Methods
    ----------
    set_led
        Set the state of the GPIO pin controlling an LED
    read_button
        Read the state of the GPIO pin connected to a button
    monitor_buttons
        Loops over the buttons reading GPIO pins
    stop
        Stops the GPIO listening
    """

    def __init__(self, led_pins: list[int], button_pins: list[int]):
        """
        Initialise the GPIOController.

        :param led_pins: List of GPIO pins connected to LEDs.
        :type led_pins: list[int]
        :param button_pins: List of GPIO pins connected to Buttons.
        :type button_pins: list[int]
        """
        # Unpack args
        self.led_pins = led_pins
        self.button_pins = button_pins
        print(f"LED pins: {led_pins}\nButton pins: {button_pins}")

        # Initialise Tracker Var
        self.running: bool = True

        GPIO.setmode(GPIO.BCM)

        # Initialize LED pins as output
        GPIO.setup(led_pins, GPIO.OUT)
        GPIO.output(led_pins, GPIO.LOW)

        # Initialize Button pins as input with pull-up resistor
        GPIO.setup(button_pins, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)

    def set_led(self, led_pin: int, state: bool) -> None:
        """
        Set the state of an LED.
        :param led_pin: GPIO pin of the LED
        :type led_pin: int
        :param state: True to turn on, False to turn off.
        :type state: bool
        :return: None
        """
        if led_pin in self.led_pins:
            print(f"LED {state}: {led_pin}")
            GPIO.output(led_pin, GPIO.HIGH if state else GPIO.LOW)
        else:
            raise ValueError(f"Invalid LED {led_pin}")

    def read_button(self, button_pin: int) -> bool:
        """
        Read the state of a button.
        :param button_pin: GPIO pin for the button
        :type button_pin: int
        :return: True if button is pressed, False otherwise.
        :rtype: bool
        """
        if button_pin in self.button_pins:
            return GPIO.input(button_pin)  # Button pressed = LOW
        else:
            raise ValueError(f"Invalid button {button_pin}")

    def monitor_buttons(self):
        """Monitor buttons and publish their status via MQTT."""
        while self.running:
            for pin in self.button_pins:
                if self.read_button(pin):  # Button pressed
                    print(f"Button Clicked: {pin}")
                    time.sleep(0.5)  # Debounce delay

    def stop(self):
        """Stop the GPIOController and clean up GPIO settings."""
        self.running = False
        time.sleep(0.5)
        GPIO.cleanup()


class LightButton:
    """
    Used for interfacing the light-up buttons via GPIO

    ...
    Attributes
    ----------
    led_pin: int
    trigger_pin: int

    ...
    Methods
    -------
    toggle_light
        Toggle light on or off and returns set state
    check_trigger
        Read GPIO port to check for trigger. Returns True if triggered, False otherwise.
    """

    def __init__(self, led_pin: int, trigger_pin: int, gpio_controller: GPIOController):
        """
        LightButton constructor
        :param led_pin: LED GPIO pin
        :type led_pin: int
        :param trigger_pin: Button trigger pin
        :type trigger_pin: int
        :param gpio_controller: GPIO controller with pins initialised
        :type gpio_controller: GPIOController
        """
        # Unpack args
        self.led_pin = led_pin
        self.trigger_pin = trigger_pin
        self._gpio_controller = gpio_controller

        # Tracker vars
        self._light_state: bool = False

    def toggle_light(self) -> bool:
        """Toggles the light on or off."""
        if self._light_state:
            self.off()
        else:
            self.on()
        return self._light_state

    def off(self):
        """Turn the light off"""
        self._light_state = False
        self._gpio_controller.set_led(self.led_pin, self._light_state)

    def on(self):
        """Turn the light on"""
        self._light_state = True
        self._gpio_controller.set_led(self.led_pin, self._light_state)

    def check_trigger(self) -> bool:
        """Reads the state of the button trigger"""
        return self._gpio_controller.read_button(self.trigger_pin)


# Example Usage
if __name__ == "__main__":
    # Define GPIO pins for LEDs and Buttons
    led_pins = [17, 27, 22, 14, 15, 18, 10, 9, 11, 23, 24, 25]
    button_pins = [5, 6, 13]

    # Initialize GPIOController
    gpio_controller = GPIOController(led_pins, button_pins)
    while True:
        try:
            # Start monitoring buttons in a separate thread
            button_thread = Thread(target=gpio_controller.monitor_buttons)
            button_thread.start()

            # Example: Control LEDs via code
            gpio_controller.set_led(17, True)  # Turn on LED 0
            time.sleep(1)
            gpio_controller.set_led(17, False)  # Turn off LED 0
            time.sleep(1)

        except KeyboardInterrupt:
            print("Stopping GPIO Controller")
            break

    gpio_controller.stop()
    time.sleep(1)
