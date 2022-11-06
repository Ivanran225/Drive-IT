import RPi.GPIO as gpio
import time

FRONT_RIGHT_GO_FRONT = ("FRONT_RIGHT_GO_FRONT", 17)
FRONT_LEFT_GO_BACK = ("FRONT_LEFT_GO_BACK", 24)
FRONT_RIGHT_GO_BACK = ("FRONT_RIGHT_GO_BACK", 22)
FRONT_LEFT_GO_FRONT = ("FRONT_LEFT_GO_FRONT", 23)

BACK_LEFT_GO_FRONT = ("BACK_LEFT_GO_FRONT", 5)
BACK_LEFT_GO_BACK = ("BACK_LEFT_GO_BACK", 6)
BACK_RIGHT_GO_BACK = ("BACK_RIGHT_GO_BACK", 26)
BACK_RIGHT_GO_FRONT = ("BACK_RIGHT_GO_FRONT", 27)

all_motors = [BACK_LEFT_GO_FRONT, BACK_LEFT_GO_BACK,
              BACK_RIGHT_GO_BACK, BACK_RIGHT_GO_FRONT,
              FRONT_LEFT_GO_FRONT, FRONT_LEFT_GO_BACK,
              FRONT_RIGHT_GO_BACK, FRONT_RIGHT_GO_FRONT]


def init():
    gpio.setmode(gpio.BCM)
    for action, pin in all_motors:
        gpio.setup(pin, gpio.OUT)
        gpio.output(pin, False)


def forward(sec=None):
    init()
    for action, pin in all_motors:
        if action.endswith("GO_FRONT"):
            gpio.output(pin, True)
        else:
            gpio.output(pin, False)
    
    if sec:
        time.sleep(sec)
        gpio.cleanup()


def backwards(sec=None):
    init()
    for action, pin in all_motors:
        if action.endswith("GO_BACK"):
            gpio.output(pin, True)
        else:
            gpio.output(pin, False)

    if sec:
        time.sleep(sec)
        gpio.cleanup()


def turn_right(sec=None):
    init()
    for action, pin in all_motors:
        if action.endswith("LEFT_GO_BACK") or action.endswith("RIGHT_GO_FRONT"):
            gpio.output(pin, True)
        else:
            gpio.output(pin, False)

    if sec:
        time.sleep(sec)
        gpio.cleanup()


def turn_left(sec=None):
    init()
    for action, pin in all_motors:
        if action.endswith("LEFT_GO_FRONT") or action.endswith("RIGHT_GO_BACK"):
            gpio.output(pin, True)
        else:
            gpio.output(pin, False)

    if sec:
        time.sleep(sec)
        gpio.cleanup()

