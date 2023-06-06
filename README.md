# Faculdade-Projeto
Projeto simples

1.	Programa de Aprendizado de Máquina Supervisionado: Previsão de Preços de Casas
Este programa utiliza um conjunto de dados de preços de casas e suas características para treinar um modelo de aprendizado de máquina supervisionado capaz de prever os preços de casas novas, com base em suas características. O programa utiliza a biblioteca scikit-learn para implementar um modelo de regressão linear.


# importando as bibliotecas necessárias
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score

# carregando o conjunto de dados
data = pd.read_csv('casas.csv')

# explorando o conjunto de dados
print(data.head())

# visualizando a distribuição dos preços das casas
sns.distplot(data['preco'])
plt.show()

# dividindo o conjunto de dados em treinamento e teste
train, test = train_test_split(data, test_size=0.2, random_state=42)

# separando as variáveis independentes e dependentes
x_train = train.drop(['preco'], axis=1)
y_train = train['preco']
x_test = test.drop(['preco'], axis=1)
y_test = test['preco']

# criando um pipeline de pré-processamento e modelo
pipe = make_pipeline(
    StandardScaler(),
    PolynomialFeatures(),
    LinearRegression()
)

# ajustando o pipeline ao conjunto de treinamento
pipe.fit(x_train, y_train)

# avaliando o modelo no conjunto de teste
y_pred = pipe.predict(x_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print('MSE: ', mse)
print('R2: ', r2)

# realizando uma busca em grade para encontrar os melhores hiperparâmetros do modelo Ridge
param_grid = {
    'polynomialfeatures__degree': [1, 2, 3],
    'ridge__alpha': [0.1, 1, 10]
}
ridge_pipe = make_pipeline(
    StandardScaler(),
    PolynomialFeatures(),
    Ridge()
)
grid = GridSearchCV(ridge_pipe, param_grid, cv=5)
grid.fit(x_train, y_train)
print(grid.best_params_)
print(grid.best_score_)

# avaliando o modelo Ridge no conjunto de teste
y_pred = grid.predict(x_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print('MSE: ', mse)
print('R2: ', r2)
