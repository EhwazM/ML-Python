import numpy as np

class LinearRegression:
    def __init__(self):
        self.theta = None
        self.intercept = None
        self.coef = None

    def fit(self, x, y):
        n = len(x)
        m = len(y)

        x_b = np.c_[np.ones((n,1)), x]

        self.theta = np.linalg.pinv(x_b.T.dot(x_b)).dot(x_b.T).dot(y)

        self.intercept = self.theta[0]
        self.coef = self.theta[1:]

    def predict(self, x):
        n = len(x)
        x_b = np.c_[np.ones((n, 1)), x]
        y_pred = x_b.dot(self.theta)
        
        return y_pred

class DGRegression:
    def __init__(self, epochs = 1000, lil_batch = None, seed = None, eta0 = 0.1, pacience = 10, tol = 1e-3):
        self.epochs = epochs
        self.lil_batch = lil_batch
        self.seed = seed
        self.eta0 = eta0
        self.pacience = pacience
        self.tol = tol
        self.theta = None
        self.intercept = None
        self.coef = None

    def Learn(self, epoch, eta0):
        return(eta0/(1 + epoch))

    def predict(self, x):
        n = len(x)
        x_b = np.c_[np.ones((n, 1)), x]
        y_pred = x_b.dot(self.theta)
        
        return y_pred

    def error_calc(self, x_b, y):
        prediction = x_b.dot(self.theta)
        error = np.mean((prediction - y)**2)
        return error

    def fit(self, x, y):
        m, n = x.shape
        x_b = np.c_[np.ones((m,1)), x]
        self.theta = np.random.rand(n+1, 1)
        best_error = float("inf")
        pacience_counter = 0

        for epoch in range(self.epochs):
            if (self.lil_batch is None and self.seed is None ):
                grad = (2/m)*(x_b.T.dot(x_b.dot(self.theta) - y))
                    
            elif (self.lil_batch is not None and self.seed is None):
                index_m = np.random.permutation(m)
                x_bm = x_b[index_m]
                y_m = y[index_m]

                for j in range(0, m, self.lil_batch):
                    xi = x_bm[j:j + self.lil_batch]
                    yi = y_m[j:j + self.lil_batch]

                    eta = self.eta0
                    grad = (2/self.lil_batch)*(xi.T.dot(xi.dot(self.theta)-yi))
    
            elif (self.lil_batch is None and self.seed is not None):
                for j in range(m):
                    random_index = np.random.randint(m)
                    xi = x_b[random_index:random_index + 1]
                    yi = y[random_index:random_index + 1]
                    
                    grad = 2*(xi.T.dot(xi.dot(self.theta) - yi))
                    
            self.theta -= self.Learn(epoch, self.eta0)*grad
            self.intercept = self.theta[0]
            self.coef = self.theta[1:]
            
            currenly_error = self.error_calc(x_b, y)
            
            if (abs(best_error - currenly_error) < self.tol): 
                print(f"Se detuvo por convergencia en iteración {epoch + 1}")
                break
                
            if (currenly_error  < best_error): 
                best_error = currenly_error
                pacience_counter = 0
            else:
                pacience_counter += 1

            if (pacience_counter >= self.pacience):
                print(f"Se detuvo por convergencia por detención anticipada en la iteración {epoch + 1}")
                break
class Ridge():
    def __init__(self, alpha = 0.1):
        self.alpha = alpha
        self.theta = None
        self.intercept = None
        self.coef = None

    def fit(self, x, y):
        m, n = x.shape
        x_b = np.c_[np.ones((m,1)), x]
        I = np.identity(n + 1)
        I[0, 0] = 0
        self.theta = np.linalg.inv(x_b.T.dot(x_b) + self.alpha*I).dot(x_b.T).dot(y)
        self.intercept = self.theta[0]
        self.coef = self.theta[1:]

    def predict(self, x):
        m, n = x.shape
        x_b = np.c_[np.ones((m,1)), x]
        y_pred = x_b.dot(self.theta)
        return y_pred

class Lasso:
    def __init__(self, alpha=0.1, epochs=1000, lil_batch=None, seed=None, eta0=0.1, pacience=10, tol=1e-3):
        self.alpha = alpha
        self.epochs = epochs
        self.lil_batch = lil_batch
        self.seed = seed
        self.eta0 = eta0
        self.pacience = pacience
        self.tol = tol
        self.theta = None
        self.intercept = None
        self.coef = None

    def learn(self, epoch, eta0):
        return eta0 / (1 + epoch)

    def fit(self, x, y):
        m, n = x.shape
        x_b = np.c_[np.ones((m,1)), x]
        self.theta = np.random.rand(n+1, 1)
        best_error = float("inf")
        pacience_counter = 0

        for epoch in range(self.epochs):
            theta_p = self.theta.copy()

            grad = (2/m) * x_b.T.dot(x_b.dot(self.theta) - y)
            grad += self.alpha * np.sign(self.theta) 

            self.theta -= self.learn(epoch, self.eta0) * grad
            self.intercept = self.theta[0]
            self.coef = self.theta[1:]

            currently_error = self.error_calc(x_b, y)

            if np.linalg.norm(self.theta - theta_p) < self.tol:
                print(f"Se detuvo por convergencia en iteración {epoch + 1}")
                break

            if currently_error < best_error:
                best_error = currently_error
                pacience_counter = 0
            else:
                pacience_counter += 1

            if pacience_counter >= self.pacience:
                print(f"Se detuvo por detención anticipada en iteración {epoch + 1}")
                break
# class Lasso():
#     def __init__(self, alpha = 0.1, epochs = 1000, lil_batch = None, seed = None, eta0 = 0.1, pacience = 10, tol = 1e-3):
#         self.alpha = alpha
#         self.epochs = epochs
#         self.lil_batch = lil_batch
#         self.seed = seed
#         self.eta0 = eta0
#         self.pacience = pacience
#         self.tol = tol
#         self.theta = None
#         self.intercept = None
#         self.coef = None

#     def learn(self, epoch, eta0):
#         return(eta0/(1 + epoch))

#     def fit(self, x, y):
#         m, n = x.shape
#         x_b = np.c_[np.ones((m,1)), x]
#         self.theta = np.random.rand(n+1, 1)
#         best_error = float("inf")
#         pacience_counter = 0
        
#         for epoch in range(self.epochs):
#             theta_p = self.theta.copy()

#             if (self.lil_batch is None and self.seed is None ):
#                 for i in range(n):
#                     eta = self.learn(i, self.eta0) #Tasa de aprendizaje
#                     if i == 0:
#                         grad = (2/m)*(x_b.T.dot(x_b.dot(self.theta) - y))
                        
#                     else:
#                         grad = (2/m) * x_b.T.dot(x_b.dot(self.theta) - y)
#                         grad += self.alpha * np.sign(self.theta)  # fuera del dot

#             elif (self.lil_batch is not None and self.seed is None):
#                 index_m = np.random.permutation(m)
#                 x_bm = x_b[index_m]
#                 y_m = y[index_m]

#                 for j in range(0, m, self.lil_batch):
#                     xi = x_bm[j:j + self.lil_batch]
#                     yi = y_m[j:j + self.lil_batch]

#                     for i in range(n):
#                         eta = self.learn(i, self.eta0) #Tasa de aprendizaje
    
#                         if i == 0:
#                             grad = (2/self.lil_batch)*(xi.T.dot(xi.dot(self.theta) - yi))
    
#                         else:
#                             grad = (2/self.lil_batch)*(xi.T.dot(xi.dot(self.theta) - yi + self.alpha*np.sign(self.theta[i])))


#             elif (self.lil_batch is None and self.seed is not None):
#                 for j in range(m):
#                     random_index = np.random.randint(m)
#                     xi = x_b[random_index:random_index + 1]
#                     yi = y[random_index:random_index + 1]
                    
#                     for i in range(n):
#                         eta = self.learn(i, self.eta0) #Tasa de aprendizaje

#                         if i == 0:
#                             grad = 2*(xi.T.dot(xi.dot(self.theta) - yi))
    
#                         else:
#                             grad = 2*(xi.T.dot(xi.dot(self.theta) - yi + self.alpha*np.sign(self.theta[i])))

                
#             if(np.linalg.norm(self.theta - theta_p) < self.tol):
#                 break
                    
#             self.theta -= self.learn(epoch, self.eta0)*grad
#             self.intercept = self.theta[0]
#             self.coef = self.theta[1:]
            
#             currenly_error = self.error_calc(x_b, y)


    def predict(self, x):
        m, n = x.shape
        x_b = np.c_[np.ones((m,1)), x]
        y_pred = x_b.dot(self.theta)
        return y_pred

    def error_calc(self, x_b, y):
        prediction = x_b.dot(self.theta)
        error = np.mean((prediction - y)**2)
        return error

            # if (abs(best_error - currenly_error) < self.tol): 
            #     print(f"Se detuvo por convergencia en iteración {epoch + 1}")
            #     break
                
            # if (currenly_error  < best_error): 
            #     best_error = currenly_error
            #     pacience_counter = 0
            # else:
            #     pacience_counter += 1

            # if (pacience_counter >= self.pacience):
            #     print(f"Se detuvo por convergencia por detención anticipada en la iteración {epoch + 1}")
            #     break