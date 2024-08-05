import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.isotonic import IsotonicRegression
from scipy.linalg import pinv
from sklearn.linear_model import LinearRegression

import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.isotonic import IsotonicRegression
from scipy.linalg import pinv
from sklearn.linear_model import LinearRegression

class PRINCALS:
    def __init__(self, X, n_components, var_types):

        self.X = X
        self.n, self.p = X.shape
        self.r = n_components
        self.var_types = var_types
        self.JS = [i for i, v in enumerate(var_types) if v in ['ordinal', 'numerical']]
        self.JM = [i for i, v in enumerate(var_types) if v == 'multi_nominal']
        self.Z = np.random.rand(self.n, self.r)
        self.W = [np.random.rand(self.get_indicator_matrix(X[:, j], var_types[j]).shape[1], self.r) for j in range(self.p)]
        self.A = np.random.rand(self.p, self.r)

    
    def get_indicator_matrix(self, X_j, var_type):
        if var_type in ['nominal', 'multi_nominal']:
            encoder = OneHotEncoder(sparse=False, drop='first')
            X_j_reshaped = X_j.reshape(-1, 1)
            G_j = encoder.fit_transform(X_j_reshaped)
        elif var_type == 'ordinal':
            encoder = OneHotEncoder(sparse=False)
            X_j_reshaped = X_j.reshape(-1, 1)
            G_j = encoder.fit_transform(X_j_reshaped)
        elif var_type == 'numerical':
            G_j = X_j.reshape(-1, 1)
        else:
            raise ValueError(f"Unknown variable type: {var_type}")
        return G_j

    def center_normalize(self, X):
        scaler = StandardScaler()
        return scaler.fit_transform(X)
    
    def optimal_scaling(self, X_star, Z, A):
        new_X = np.zeros_like(X_star)
        for j in range(self.p):
            var_type = self.var_types[j]
            Gj = self.get_indicator_matrix(X_star[:, j], var_type)
            Aj = A[:, j].reshape(-1, 1)
            y = Z @ Aj
            if var_type == 'ordinal':
                new_X[:, j] = IsotonicRegression().fit_transform(y, X_star[:, j])
            elif var_type == 'numerical':
                model = LinearRegression()
                model.fit(y, X_star[:, j])
                new_X[:, j] = model.predict(y)
            elif var_type in ['nominal', 'multi_nominal']:
                new_X[:, j] = y.flatten()
            else:
                raise ValueError(f"Unknown variable type: {var_type}")
        return self.center_normalize(new_X)


    def loss_function(self, Z, A, X_star):
        loss = 0
        for j in range(self.p):
            X_star_j = X_star[:, j]
            ZA_j = Z @ A[:, j]
            loss += np.sum((X_star_j - ZA_j) ** 2)
        return loss

    def fit(self, max_iter=100, tol=1e-6):
        X_star = self.center_normalize(self.X)
        Z = self.Z.copy()
        prev_loss = float('inf')

        for iteration in range(max_iter):
            # Update W for all variables
            W = [self.get_indicator_matrix(X_star[:, j], self.var_types[j]).T @ Z for j in range(self.p)]
            
            # Update W for multiple nominal variables
            for j in self.JM:
                Gj = self.get_indicator_matrix(X_star[:, j], self.var_types[j])
                W[j] = pinv(Gj.T @ Gj) @ Gj.T @ Z
            
            # Update A and X_star for each variable
            for j in self.JS:
                Gj = self.get_indicator_matrix(X_star[:, j], self.var_types[j])
                A_j = W[j].T @ pinv(Gj.T @ Gj) @ Gj.T @ X_star[:, j]
                Y_j = W[j] @ A_j
                if self.var_types[j] == 'ordinal':
                    Y_j = IsotonicRegression().fit_transform(Y_j, X_star[:, j])
                elif self.var_types[j] == 'numerical':
                    Y_j = np.linalg.lstsq(A_j, X_star[:, j], rcond=None)[0]
                X_star[:, j] = Gj @ Y_j
            
            # Update Z
            Gj_list = [self.get_indicator_matrix(X_star[:, j], self.var_types[j]) for j in range(self.p)]
            Z_new = np.mean(np.hstack(Gj_list), axis=1)
            Z_new -= np.mean(Z_new, axis=0)
            Z_new, _ = np.linalg.qr(Z_new)
            
            # Compute loss and check for convergence
            current_loss = self.loss_function(Z_new, np.hstack(W), X_star)
            print(f'Iteration {iteration + 1}, Loss: {current_loss}')
            
            if abs(prev_loss - current_loss) < tol:
                print('Converged')
                break
            prev_loss = current_loss
        
        self.Z = Z_new
        self.W = W
        return self.Z, self.W

