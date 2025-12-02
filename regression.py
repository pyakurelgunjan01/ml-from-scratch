class LinearRegressionScratch:

    def __init__(self):
        self.m = 0
        self.c = 0

    def mean(self, values):
        return sum(values) / len(values)

    def variance(self, values):
        mean_val = self.mean(values)
        return sum((x - mean_val) ** 2 for x in values)

    def covariance(self, X, Y):
        mean_x = self.mean(X)
        mean_y = self.mean(Y)
        return sum((X[i] - mean_x) * (Y[i] - mean_y) for i in range(len(X)))

    def fit(self, X, Y):
        # Slope m = covariance / variance
        self.m = self.covariance(X, Y) / self.variance(X)
        mean_x = self.mean(X)
        mean_y = self.mean(Y)

        # Intercept c = meanY - m * meanX
        self.c = mean_y - self.m * mean_x

    def predict(self, x):
        return self.m * x + self.c

    def mse(self, X, Y):
        total = 0
        for i in range(len(X)):
            y_pred = self.predict(X[i])
            total += (Y[i] - y_pred) ** 2
        return total / len(X)


# ---------- MAIN CODE ----------
if __name__ == "__main__":
    # Example dataset (Height vs Weight)
    X = [150, 160, 170, 180]
    Y = [50, 55, 65, 75]

    model = LinearRegressionScratch()
    model.fit(X, Y)

    print("Slope (m):", model.m)
    print("Intercept (c):", model.c)

    print("\nPrediction for height 175 cm =", model.predict(175))

    print("\nModel MSE =", model.mse(X, Y))
