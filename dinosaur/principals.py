import numpy as np
from scipy.linalg import svd, pinv
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

class PRINCIPALS:

    def __init__(self, X, n_components, var_types):
        
        self.X = X
        self.n, self.p = X.shape
        self.r = n_components
        self.var_types = var_types  # List of variable types
        self.Z = np.random.rand(self.n, self.r)
        self.A = np.random.rand(self.p, self.r)
        self.X_star = np.copy(X)

    def center_normalize(self, X):
        scaler = StandardScaler()
        return scaler.fit_transform(X)
    
    def loss_function(self, Z, A, X_star):
        loss = 0
        for j in range(self.p):
            X_star_j = X_star[:, j]
            ZA_j = Z @ A[:, j]
            loss += np.sum((X_star_j - ZA_j) ** 2)
        return loss

    def optimal_scaling(self, X_star, Z, A):
        new_X = np.zeros_like(X_star)

        for j in range(self.p):
            var_type = self.var_types[j]
            Aj = A[:, j].reshape(-1, 1)
            y = Z @ Aj

            if var_type == 'ordinal':
                new_X[:, j] = IsotonicRegression().fit_transform(y, X_star[:, j])
            elif var_type == 'numerical':
                model = LinearRegression()
                model.fit(y, X_star[:, j])
                new_X[:, j] = model.predict(y)
            elif var_type == 'nominal':
                new_X[:, j] = y.flatten()
            else:
                raise ValueError(f"Unknown variable type: {var_type}")

        return self.center_normalize(new_X)

    def fit(self, max_iter=100, tol=1e-6):

        #Step 0. Initialization
        X_star = self.center_normalize(self.X_star)
        A = self.A.copy()
        Z = self.Z.copy()
        prev_loss = float('inf')

        for iteration in range(max_iter):
            
            # Step 1. Model Estimation step
            U, Sigma, Vt = svd(X_star, full_matrices=False)
            U_r = U[:, :self.r]
            Sigma_r = Sigma[:self.r]
            V_r = Vt[:self.r, :]
            Z = U_r
            A = np.dot(np.diag(Sigma_r), V_r)
            
            # Step 2. Optimal Scaling
            X_star = self.optimal_scaling(X_star, Z, A)
            
            current_loss = self.loss_function(Z, A, X_star)
            print(f'Loss: {current_loss}')
            
            if abs(prev_loss - current_loss) < tol:
                print('Converged')
                break
            prev_loss = current_loss
        
        self.Z = Z
        self.A = A
        return self.Z, self.A
