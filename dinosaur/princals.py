import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import SplineTransformer

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
            encoder = OneHotEncoder(sparse_output=False, drop='first')
            X_j_reshaped = X_j.reshape(-1, 1)
            G_j = encoder.fit_transform(X_j_reshaped)
        elif var_type == 'ordinal':
            encoder = OneHotEncoder(sparse_output=False)
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

    def initialize_A(self, Z_0, G, Y_0):
        A = []

        for j in range(self.p):
            Gj = G[j] 
            Yj = Y_0[j].reshape(-1, 1)
            Z0 = Z_0.T
            Aj = Z0 @ Gj @ Yj

            normAj = np.linalg.norm(Aj)
            if normAj > 0:
                A_j_0_normalized = Aj / normAj
            else:
                A_j_0_normalized = Aj

            A.append(normAj)

        return A

    def update_W(self, Z_t, G):
        W = []
        for j in range(self.p):
            Gj = G[j]

            Gj_Gj_T_inv = np.linalg.pinv(Gj.T @ Gj)
            W_j_t1 = Gj_Gj_T_inv @ Gj.T @ Z_t

            W.append(W_j_t1)
        return W

    def update_A_and_Y(self, W_t1, G, Y_t):
        A = []
        Y = []
        for j in range(self.p):
            W_j_t1 = W_t1[j]
            Gj = G[j]
            Y_j_t = Y_t[j]
            numerator = W_j_t1.T @ Gj.T @ Gj @ Y_j_t
            denominator = Y_j_t.T @ Gj.T @ Gj @ Y_j_t
            A_j_t1 = numerator / denominator
            A.append(A_j_t1)
            Y_j_t1 = W_j_t1 @ A_j_t1
            Y.append(Y_j_t1)

        return A, Y

    def update_Z(self, W_t1, G):

        Z_t1 = np.mean([G[j] @ W_t1[j] for j in range(self.p)], axis=0)
        Z_t1_centered = Z_t1 - np.mean(Z_t1, axis=0)
        Z_t1_orthonormalized, _ = np.linalg.qr(Z_t1_centered)

        return Z_t1_orthonormalized

    def fit(self, max_iter=100, tol=1e-6):
        X_star = self.center_normalize(self.X)
        Z_0 = np.random.rand(self.n, self.r)
        G = [self.get_indicator_matrix(X_star[:, j], self.var_types[j]) for j in range(self.p)]
        Y_0 = [np.arange(1, G[j].shape[1] + 1) for j in range(self.p)]

        self.A = self.initialize_A(Z_0, G, Y_0)
        Z = Z_0
        prev_loss = float('inf')

        for iteration in range(max_iter):

            W = self.update_W(Z, G)
            tmp_loss = 0

            self.A,self.Y = self.update_A_and_Y(W, G, Y_0)

            for j in self.JS:
                Gj = G[j]
                Aj = self.A[j].reshape(-1, 1)
                Y_j = W[j] @ Aj
                if self.var_types[j] == 'ordinal':
                    spline_transformer = SplineTransformer(degree=3, include_bias=False, n_knots=5)
                    spline_transformer.fit(W[j])
                    W_j_transformed = spline_transformer.transform(W[j])
                    
                    model = LinearRegression()
                    model.fit(W_j_transformed, Y_j)
                    Y_j = model.predict(W_j_transformed)
                elif self.var_types[j] == 'numerical':
                    Y_j = np.linalg.lstsq(Aj, X_star[:, j], rcond=None)[0]
                Y_0[j] = Y_j

                residual = Z - Gj @ W[j]
                tmp_loss += np.trace(residual.T @ residual)
                X_star[:, j] = (Gj @ Y_j).reshape(-1)

            # Update Z
            Z_new = self.update_Z(W, G)
            print(f'Iteration {iteration + 1}, Loss: {tmp_loss}')
            Z = Z_new

            if abs(prev_loss - tmp_loss) < tol:
                print('Converged')
                break
            prev_loss = tmp_loss

        self.Y = Y_0
        self.Z = Z_new
        self.W = W
        return self.Z, self.W

