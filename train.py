import pandas as pd
import numpy as np
from my_linear_regression import MyLinearRegression as MyLr

try:
    data = pd.read_csv("data.csv")
    Mileage = np.array(data["km"]).reshape(-1, 1)
    Price = np.array(data["price"]).reshape(-1, 1)
    open_file = open("thetas.txt", "w")
    myLR = MyLr(np.array([[0], [0]]), 1e-1, 10000)
    print(myLR.fit_(Mileage, Price))
    model = { "thetas": myLR.thetas.squeeze(), "bounds": myLR.bounds }
    modelDF = pd.DataFrame(data=model)
    modelDF.to_csv("model.csv")
    myLR.plot_(Mileage, Price)
except Exception as e:
    print("Error: ", e)
