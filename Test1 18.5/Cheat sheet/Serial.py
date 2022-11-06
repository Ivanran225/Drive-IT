import serial
import time
import keyboard
arduino = serial.Serial(port='COM4', baudrate=9600, timeout=.1)
def write_read(x):
    arduino.write(bytes(x, 'utf-8'))
    time.sleep(0.05)

while True:
    for key in ["w", "a", "s", "d"]:
        if keyboard.is_pressed(key):
            movement = key
            arduino.write(bytes(movement, 'utf-8'))
            time.sleep(0.05)

            print(movement) # printing the value
    if keyboard.is_pressed("q"):
        break