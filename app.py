# Projeto 2 - Construção e Deploy de API - Machine Learning Para Prever o Preço do Bitcoin
# Script da API

# Importa a biblioteca uvicorn para rodar o servidor WSGI
import uvicorn

# Importa a classe FastAPI para criar a aplicação
from fastapi import FastAPI

# Importa a classe BaseModel do Pydantic para validar os dados
from pydantic import BaseModel

# Importa a função load do joblib para carregar modelos salvos
from joblib import load

# Importa o módulo personalizado para calcular indicadores financeiros
from indicadores import dsa_calcula_indicadores

# Importa a biblioteca yfinance para baixar os dados de cotação do Bitcoin
import yfinance as yf

# Define metadados para a documentação da API
tags_metadata = [{"name": "DSA-Projeto2", "description": "Prevendo o Preço de Bitcoin com Machine Learning"}]

# Cria a instância da aplicação FastAPI com informações de metadados
app = FastAPI(title = "Bitcoin Price API",
              description = "DSA - Projeto2",
              version = "1.0",
              contact = {"name": "DSA", "url": "https://www.datascienceacademy.com.br"},
              openapi_tags = tags_metadata)

# Define um modelo Pydantic para validar os dados de entrada
class Features(BaseModel):
    Model: str

# Define uma rota raiz que retorna uma mensagem
# Este método define uma rota HTTP GET para a raiz do seu servidor (ou seja, "/"). 
# Quando alguém acessa a URL base da sua API, essa função é chamada e retorna o texto "Prevendo o Preço de Bitcoin com Machine Learning". 
# É uma forma simples de verificar se a sua API está funcionando corretamente.
@app.get("/")
def message():
    texto = "Esta é uma API Para Prever o Preço de Bitcoin com Machine Learning. Use o Método Adequado."
    return texto

# O async (usado abaixo) antes da definição da função em um aplicativo web construído com um framework como FastAPI 
# indica que a função é assíncrona. Isso significa que a função pode ser pausada e retomada, permitindo que Python execute 
# outras tarefas enquanto espera por uma operação de entrada/saída (I/O) ser concluída, como acessar um banco de dados ou 
# fazer uma requisição de rede. O uso de async permite que o servidor web lide com muitas requisições simultaneamente de forma mais 
# eficiente, sem bloquear o processamento enquanto espera que as tarefas de I/O sejam concluídas.

# Define uma rota para a previsão que utiliza o modelo especificado
# Este método define uma rota HTTP POST para "/predict". 
# É usado para fazer previsões com base nas características (features) enviadas no corpo da requisição. 
# A função predict recebe um objeto Features como argumento, que contém os valores das características necessárias para fazer a previsão. 
# Dentro dessa função, você vai extrair essas características, fazer algum pré-processamento necessário e então passá-las para 
# o seu modelo de machine learning para obter a previsão. Finalmente, a previsão é retornada como resposta à requisição.
@app.post("/predict", tags = ["DSA-Projeto2"])
async def predict(Features: Features):

    # Features é uma classe que define a estrutura das características esperadas na requisição para a rota /predict. 
    # Ela herda de BaseModel, que é uma classe do Pydantic usada para definir modelos de dados com validação de tipos. 
    # Cada atributo da classe corresponde a uma característica que o modelo espera.
    # Em nosso caso não passamos atributos na chamada da API pois os atributos são calculados dentro da API.

    # Define que iremos buscar dados de cotação do Bitcoin
    btc_ticker = yf.Ticker("BTC-USD")

    # Baixa os últimos 200 dias de dados de preço do Bitcoin sem incluir ações (splits/dividends)
    valor_historico_btc = btc_ticker.history(period = "200d", actions = False)

    # Remove a localização de fuso horário dos dados
    valor_historico_btc = valor_historico_btc.tz_localize(None)

    # Calcula indicadores financeiros nos dados históricos
    valor_historico_btc = dsa_calcula_indicadores(valor_historico_btc)

    # Ordena os dados pelo índice de forma decrescente
    valor_historico_btc = valor_historico_btc.sort_index(ascending = False)

    # Seleciona as linhas a partir do índice zero e todas as colunas
    dados_entrada = valor_historico_btc.iloc[0, :]

    # Substitui valores ausentes por 0
    dados_entrada = dados_entrada.fillna(0)

    # Converte os dados de entrada para um array
    dados_entrada = dados_entrada.array

    # Redimensiona o array para o formato esperado pelo modelo
    dados_entrada = dados_entrada.reshape(1, -1)

    # Carrega o objeto scaler para normalizar os dados de entrada
    scaler = load("scaler_dsa.bin")

    # Padroniza os dados de entrada
    dados_entrada = scaler.transform(dados_entrada)

    # Atribui o modelo especificado pelo usuário
    Model = Features.Model

    # Carrega o modelo de Machine Learning se especificado
    if Model == "Machine Learning":
        arquivo = "modelo_dsa.joblib"
        model = load(arquivo)

    # Faz a previsão usando o modelo carregado
    previsao = model.predict(dados_entrada)

    # Obtém o último preço conhecido do Bitcoin
    ultimo_preco = valor_historico_btc.iloc[0, 3]

    # Monta a resposta com o modelo usado, o último preço e a previsão
    response = {"Modelo": Model, 
                "Último Preço": round(ultimo_preco, 2), 
                "Previsão Para o Próximo Dia": round(previsao.tolist()[0], 2)}

    # Retorna a resposta
    return response

# Inicia o servidor em qualquer ip da máquina e na porta 3000
if __name__ == "__main__":
    uvicorn.run(app, host = "0.0.0.0", port = 3000)




    
