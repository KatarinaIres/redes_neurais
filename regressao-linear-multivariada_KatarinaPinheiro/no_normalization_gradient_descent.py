import numpy as np
import time

# Gerar dados sintéticos de exemplo
np.random.seed(42)
X = 2 * np.random.rand(100, 2)  # 100 amostras, 2 variáveis independentes
y = 4 + 3 * X[:, 0] + 5 * X[:, 1] + np.random.randn(100)  # y = 4 + 3x1 + 5x2 + ruído

# Adiciona a coluna de 1s para o termo de bias (intercepto)
X_b = np.c_[np.ones((X.shape[0], 1)), X]

# Hiperparâmetros
learning_rate = 0.01
n_iterations = 1000
m = X_b.shape[0]  # número de amostras

# Inicialização dos parâmetros (theta)
theta = np.random.randn(X_b.shape[1])

# Gradiente Descendente
start_time = time.time()
for iteration in range(n_iterations):
    gradients = (2/m) * X_b.T.dot(X_b.dot(theta) - y)
    theta -= learning_rate * gradients
end_time = time.time()

# Predição e erro
y_pred = X_b.dot(theta)
mse = np.mean((y - y_pred) ** 2)

# Resultados
print("Parâmetros (theta):", theta)
print("Tempo de execução:", round(end_time - start_time, 4), "s")
print("Erro quadrático médio (MSE):", round(mse, 4))
