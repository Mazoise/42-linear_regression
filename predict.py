import pandas as pd
import numpy as np


try:
    open_file = open("thetas.txt", "r")
    thetas = open_file.read()
    open_file.close()
    thetas = np.array(thetas.split(",")).astype(float)
    assert len(thetas) == 2
except Exception as e:
    print("No theta set")
    thetas = np.array([0, 0])
print("Type \"exit\" to quit the program")
while True:
    mile = input("Mileage: ")
    if mile == "exit" or mile == "Exit":
        break
    try:
        mile = float(mile)
    except Exception as e:
        print("Error: ", e)
        continue
    print("Predicted price : ", thetas[0] + thetas[1] * mile)
