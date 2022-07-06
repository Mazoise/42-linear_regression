import numpy as np
import matplotlib.pyplot as plt


class MyLinearRegression():
    """
    Description:
        My personnal linear regression class to fit like a boss.
    """

    def __init__(self, thetas, alpha=0.001, max_iter=1000):
        self.alpha = alpha
        self.max_iter = max_iter
        self.thetas = thetas
        self.bounds = None

    def fit_(self, x, y):
        """
        Description:
            Fits the model to the training dataset contained in x and y.
        Args:
            x: has to be a numpy.array, a vector of shape m * 1:
            (number of training examples, 1).
            y: has to be a numpy.array, a vector of shape m * 1:
            (number of training examples, 1).
            theta: has to be a numpy.array, a vector of shape 2 * 1.
            alpha: has to be a float, the learning rate
            max_iter: has to be an int, the number of
            iterations done during the gradient descent
        Return:
            new_theta: numpy.array, a vector of shape 2 * 1.
            None if there is a matching shape problem.
            None if x, y, theta, alpha or max_iter is not of the expected type.
        Raises:
            This function should not raise any Exception.
        """
        try:
            self.thetas = self.thetas.astype(float)
            for i in range(self.max_iter):
                self.thetas -= self.alpha * self.gradient_(self.minmax_(x), y)
            return self.thetas
        except Exception as e:
            print("Error in fit: ", e)
            return None

    def minmax_(self, x):
        if (type(x) != np.ndarray or len(x) == 0):
            print("TypeError in minmax")
            return None
        try:
            if self.bounds is None:
                self.bounds = np.array([x.min(), x.max()])
            return (x - self.bounds[0]) / (self.bounds[1] - self.bounds[0])
        except Exception as e:
            print("Error in minmax: ", e)
            return None

    def predict_thetas_(self, x, thetas):
        if (type(x) != np.ndarray or type(thetas) != np.ndarray
           or len(x) == 0 or len(thetas) == 0
           or len(x.shape) != 2 or len(thetas.shape) != 2
           or thetas.shape[1] != 1
           or thetas.shape[0] != x.shape[1] + 1):
            print("TypeError in predict")
            return None
        try:
            x = self.add_intercept_(x)
            return x.dot(thetas)
        except Exception as e:
            print("Error in predict: ", e)
            return None

    def predict_(self, x):
        return self.predict_thetas_(x, self.thetas)

    @staticmethod
    def loss_elem_(y, y_hat):
        if (type(y) != np.ndarray or type(y_hat) != np.ndarray
           or len(y.shape) != 2 or y.shape != y_hat.shape or y.shape[1] != 1):
            print("TypeError in loss_elem")
            return None
        try:
            return (y_hat - y) ** 2
        except Exception as e:
            print("Error in loss_elem: ", e)
            return None

    @staticmethod
    def loss_(y, y_hat):
        if (type(y) != np.ndarray or type(y_hat) != np.ndarray
           or len(y.shape) != 2 or y.shape != y_hat.shape
           or y.shape[1] != 1):
            print("TypeError in loss")
            return None
        try:
            return (np.swapaxes(y_hat - y, 0, 1).dot(y_hat - y)
                    / (2 * len(y)))[0][0]
        except Exception as e:
            print("Error in loss: ", e)
            return None

    def gradient_(self, x, y):
        try:
            return (np.swapaxes(self.add_intercept_(x), 0, 1)
                    .dot(self.predict_(x) - y) / len(x))
        except Exception as e:
            print("Error in gradient: ", e)
            return None

    @staticmethod
    def add_intercept_(x):
        if (type(x) != np.ndarray or len(x) == 0
           or len(x.shape) != 2):
            print("TypeError in add intercept")
            return None
        try:
            return np.insert(x, 0, 1, axis=1).astype(float)
        except Exception as e:
            print("Error in add_interceptor: ", e)
            return None

    def mse_(self, y, y_hat):
        try:
            return self.loss_(y, y_hat) * 2
        except Exception as e:
            print("Error in mse: ", e)
            return None

    def plot_(self, x, y, xlabel="x", ylabel="y", units="units"):
        plt.plot(x, self.predict_(self.minmax_(x)), 'X-',
                 color='limegreen',
                 label="$s_{predict}(" + units + ")$")
        plt.plot(x, y, 'o',
                 label="$s_{true}(" + units + ")$",
                 color="deepskyblue")
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
                   ncol=2, borderaxespad=0.)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid()
        plt.show()

    def plot_loss_(self, x, y, xborns=(-10, 8), yborns=(-5, 15)):
        thetas0 = np.linspace(self.thetas[0] + xborns[0],
                              self.thetas[0] + xborns[1], 6)
        thetas1 = np.linspace(self.thetas[1] + yborns[0],
                              self.thetas[1] + yborns[1], 1000)
        J = np.zeros((1000))
        for i in range(thetas0.shape[0]):
            for j in range(thetas1.shape[0]):
                J[j] = self.mse_(y,
                                  self.predict_thetas_(x,
                                                       np.array([thetas0[i],
                                                                 thetas1[j]])))
            plt.plot(thetas1, J, str(0.1 + i / 7.0),
                     label="$J(θ_0=c_" + str(i) + "θ_1)$")
        plt.ylabel("cost function J($θ_0, θ_1$)")
        plt.xlabel("$θ_1$")
        plt.legend(loc="lower right")
        plt.xlim([-14.5, -3.5])
        plt.ylim([10, 150])
        plt.grid()
        plt.show()
