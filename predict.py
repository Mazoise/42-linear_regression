import pandas as pd
import numpy as np


try:
    model = pd.read_csv("model.csv")
    thetas = np.array(model["thetas"].values).reshape(-1, 1)
    bounds = np.array(model["bounds"].values).reshape(-1, 1)
except Exception as e:
    print("No theta set", e)
    thetas = np.array([0, 0])
    bounds = np.array([0, 0])
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
    print("Predicted price : ", thetas[0] + thetas[1] * (mile - bounds[0]) / (bounds[1] - bounds[0]))
