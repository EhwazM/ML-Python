import numpy as np

class LogisticRegression:
    def __init__(self, epochs=1000, eta0=0.1, umbral=0.5, tol = 1e-3):
        self.epochs = epochs
        self.eta0 = eta0
        self.umbral = umbral
        self.tol = tol
        self.theta = None
        self.intercept = None
        self.coefficent= None

    def sigmoidea(self,z):
        return (1/(1 + np.exp(-z)))

    def softmax(self, z):
        exp_z = np.exp(z)
        return exp_z/ np.sum(exp_z, axis=1, keepdims=True)

    def learn(self, eta0, epoch):
        return(self.eta0/(1 + epoch))

    def error_calc(self, x_b, y):
        prediction = x_b.dot(self.theta)
        error = np.mean((prediction - y)**2)
        return error

    def fit(self, x, y):
        m, n = x.shape
        num_classes = len(np.unique(y))

        if num_classes == 2:
            x_b = np.c_[np.ones((m, 1)), x]
            self.theta = np.zeros((n+1, 1))
                                  
            for epoch in range (self.epochs):
                theta_p = self.theta.copy()
                
                for i in range(m):        
                    randomIndex =  np.random.randint(m)
                    xi = x_b[randomIndex : randomIndex + 1]
                    yi = y[randomIndex : randomIndex + 1]
                    zi = np.dot(xi, self.theta)
                    hi = self.sigmoidea(zi)
                    gra = np.dot(xi.T, (hi-yi))
                    eta = self.learn(self.eta0, epoch)
                    self.theta -= eta*gra
                    self.intercept = self.theta[0,0]
                    self.coef = self.theta[1:,0]

                if(np.linalg.norm(self.theta - theta_p) < self.tol):
                    break

        else:
            num_data = len(y)
            y_one = np.zeros((num_data, num_classes))
            for i in range(num_data):
                y_one[i, y[i]] = 1

            x_b = np.c_[np.ones((m,1)), x]
            self.theta = np.zeros((n+1, num_classes))
                                  
            for epoch in range (self.epochs):
                theta_p = self.theta.copy()
                for i in range(m):        
                    randomIndex =  np.random.randint(m)
                    xi = x_b[randomIndex : randomIndex + 1]
                    yi = y_one[randomIndex : randomIndex + 1]
                    zi = np.dot(xi, self.theta)
                    hi = self.softmax(zi)
                    gra = np.dot(xi.T, (hi-yi))
                    eta = self.learn(self.eta0, epoch)
                    self.theta -= eta*gra
                    self.intercept = self.theta[0,0:]
                    self.coef = np.transpose(self.theta[1:,0:])

                if(np.linalg.norm(self.theta - theta_p) < self.tol):
                    break


    def predict(self, x):
        m,n = x.shape
        x_b = np.c_[np.ones((m,1)), x]
        z = np.dot(x_b, self.theta)
        prob = self.sigmoidea(z)
        y_pred = np.where(prob >= self.umbral, 1, 0)
        return y_pred
                
                