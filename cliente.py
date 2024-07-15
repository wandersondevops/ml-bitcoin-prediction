# Projeto 2 - Construção e Deploy de API - Machine Learning Para Prever o Preço do Bitcoin
# APP Cliente

# Import
import requests

# Dados que serão passados para a API
payload = {"Model" : "Machine Learning"}

# Faz a requisição à API
resposta = requests.post("http://localhost:3000/predict", json = payload).json()

# Imprime a resposta
print('\nAcessando a API do Projeto 2 da DSA Para Prever o Preço do Bitcoin!')
print('\nResposta da API:\n')
print(resposta)
print('\nObrigado Por Usar Esta API!\n')
