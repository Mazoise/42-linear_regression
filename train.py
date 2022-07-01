import pandas as pd
import numpy as np
from my_linear_regression import MyLinearRegression as MyLr

try:
    data = pd.read_csv("data.csv")
    Mileage = np.array(data["km"]).reshape(-1, 1)
    Price = np.array(data["price"]).reshape(-1, 1)
    open_file = open("thetas.txt", "w")
    myLR = MyLr(np.array([[0], [0]]), 1e-10, 10000000)
    myLR.fit_(Mileage, Price)
    open_file.write(str(myLR.thetas[0]) + "," + str(myLR.thetas[1]))
    open_file.close()
    myLR.plot_(Mileage, Price)
except Exception as e:
    print("Error: ", e)
