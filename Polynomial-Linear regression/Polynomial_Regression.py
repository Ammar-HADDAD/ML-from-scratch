import numpy as np

class Polynomial_Regressor:
    def __init__(self, degree=1, learning_rate=0.001, n_iter=1000): # Degree=1 for linear regression, degree=2 for quadratic regression ...  
        self.degree = degree  # The degree of the polynomial to fit
        self.learning_rate = learning_rate  # Learning rate for gradient descent
        self.n_iter = n_iter  # Number of iterations for gradient descent
        
        # Initialize the coefficients randomly with shape (degree + 1, 1) 
        self.args = np.random.rand(degree + 1, 1)
        
        # Create an array to store the error function values during training
        self.error_func = np.zeros(self.n_iter)
    # Method to fit the polynomial regression model to the given data
    def fit(self, X, y):
        # X should be a single-column array, reshape it to a 2D column vector
        n_dim, _ = X.shape
        X = X.reshape((n_dim, 1))
        
        # Calculate the constant C for gradient descent (negative reciprocal of the number of data points)
        C = -np.divide(1, n_dim)
        
        # Transform the input X by combining its powers according to the specified degree
        X_met = self.combiner(X)

        # Gradient descent algorithm to update coefficients and minimize the error
        for n in range(self.n_iter):
            Y = y - self.model(X_met, self.args)  # Calculate the difference between predictions and actual y values
            Jc = C * X_met.T.dot(Y)  # Compute the gradient of the error function
            self.args = np.subtract(self.args, np.multiply(self.learning_rate, Jc))  # Update coefficients using the gradient
            J = (-C/2) * (Y.T.dot(Y))  # Compute the error function value
            self.error_func[n] = J  # Store the error function value at this iteration
        return self.error_func  # Return the array of error function values for analysis

    # Method to combine the input X with powers according to the polynomial degree
    def combiner(self, X):
        X_ = np.full((X.shape[0], 1), 1)  # Start with a column vector of ones (corresponding to X^0)
        for i in range(self.degree):
            X_ = np.hstack((X**(i + 1), X_))  # Append X^i to the existing matrix
        return X_

    # Method to predict output values based on the fitted model
    def model(self, X, args):
        return X.dot(args)

    # Method to predict output values for new input data using the fitted model
    def predict(self, X):
        return self.combiner(X).dot(self.args)  # Use the fitted coefficients to predict new values
        
