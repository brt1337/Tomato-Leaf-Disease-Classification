# Classificação de Doenças em Plantas

Este projeto visa classificar doenças em folhas de tomate usando uma rede neural convolucional implementada com TensorFlow/Keras e apresentada através de uma aplicação web utilizando Streamlit.

## Visão Geral

O objetivo deste projeto é ajudar na identificação de diferentes doenças que afetam folhas de tomateiro. A aplicação permite que usuários façam upload de imagens de folhas de tomate para análise, onde a rede neural treinada irá prever se a folha está saudável ou afetada por uma das seguintes doenças:

- Acaro
- Enrolamento Fisiologico das Folhas
- Mancha Alvo
- Mancha Bacteriana
- Septoriose
- Pinta Preta
- Requeima
- Tomato Yellow Leaf Curl Virus
- Mosaico Virus Y

Para cada previsão, a aplicação também fornece um link útil com informações sobre a doença identificada.

## Explicação dos Parâmetros

Aqui estão os parâmetros utilizados na definição e compilação do modelo de rede neural convolucional:

### Camadas Convolucionais (Conv2D)

- **filters**: Número de filtros convolucionais aplicados à entrada.
- **kernel_size**: Dimensão da janela de convolução.
- **activation**: Função de ativação aplicada após cada operação de convolução.

### Camadas de Pooling (MaxPooling2D)

- **pool_size**: Dimensão da janela de pooling.
- **strides**: Passo do pooling.
- **padding**: 'valid' ou 'same'. Determina se o padding é aplicado para manter as dimensões da saída.

### Camada de Flattening (Flatten)

Transforma a saída das camadas convolucionais e de pooling em um vetor unidimensional para alimentar as camadas densas (fully connected).

### Camada Fully Connected (Dense)

- **units**: Número de neurônios na camada.
- **activation**: Função de ativação aplicada à saída.

### Dropout

- **rate**: Fração dos neurônios a serem desligados aleatoriamente durante o treinamento para reduzir overfitting.

### Camada de Saída

- **units**: Número de classes de saída.
- **activation**: 'softmax' para problemas de classificação multiclasse. Gera uma distribuição de probabilidade sobre as classes.

### Compilação do Modelo

- **optimizer**: Algoritmo de otimização usado para ajustar os pesos do modelo durante o treinamento (por exemplo, 'adam', 'sgd', 'rmsprop').
- **loss**: Função de perda utilizada para medir o quão bem o modelo está se saindo durante o treinamento.
- **metrics**: Métricas utilizadas para monitorar o desempenho do modelo durante o treinamento (por exemplo, 'accuracy' para precisão).

Estes parâmetros são fundamentais para configurar e compilar um modelo de rede neural convolucional para classificação de imagens. Eles definem a arquitetura da rede, como ela aprende durante o treinamento e como ela é avaliada durante o treinamento e teste.

## Pré-requisitos

- Python 3.6 ou superior
- Bibliotecas Python: Streamlit, TensorFlow, numpy, PIL

## Instalação

1. Clone o repositório:

   ```bash
   git clone https://github.com/brt1337/Tomato-Leaf-Disease-Classification.git

2. Instale as dependências:
- pip install -r requirements.txt

## Como Usar

1. Navegue até o diretório do projeto:
- cd nome-do-repositorio

2. Execute a aplicação Streamlit:
- streamlit run app.py

## Exemplo de Uso

1. Faça o upload de uma imagem de uma folha de tomate afetada por uma das doenças treindas ou uma folha saudavel .

- A aplicação irá exibir a imagem carregada e a previsão da doença, juntamente com um link útil para mais informações sobre a doença identificada.


## Contribuições
- Contribuições são bem-vindas! Para sugestões de melhorias, por favor abra uma issue primeiro para discutir o que você gostaria de mudar.
