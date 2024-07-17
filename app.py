import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.backend import clear_session
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout,Input
from tensorflow.keras.preprocessing import image
from PIL import Image
import tempfile
import os


#tf.config.set_visible_devices([], 'GPU')




#model = tf.keras.models.load_model('models/cnn/rede_neural_convolucional_epochs_100.h5')


def build_model():
    model = Sequential()
    model.add(Input(shape=(256, 256, 3)))  # Camada de entrada com a forma desejada
    model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', name='Camada_Convolucional_1'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid', name='Camada_MaxPooling_1'))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', name='Camada_Convolucional_2'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid', name='Camada_MaxPooling_2'))
    model.add(Flatten(name='Camada_Flattening'))
    model.add(Dense(units=1024, activation='relu', name='Camada_FullConnect'))
    model.add(Dropout(rate=0.5, name='Camada_de_Dropout'))
    model.add(Dense(units=10, activation='softmax', name='Camada_de_saida'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

model = build_model()
model.load_weights('models/cnn/rede_neural_convolucional_weights_epochs.h5')


disease_links = {
    0: ("Acaro", "https://agriculture.basf.com/br/pt/conteudos/cultivos-e-sementes/tomate/tipos-acaro.html"),
    1: ("Enrolamento_Fisiologico_das_Folhas", "http://ephytia.inra.fr/pt/C/5334/Tomate-Enrolamento-Fisiologico-das-Folhas"),
    2: ("Saudavel", None),
    3: ("Mancha_Alvo", "https://agriculture.basf.com/br/pt/conteudos/cultivos-e-sementes/soja/mancha-alvo-principais-sintomas.html"),
    4: ("Mancha_Bacteriana", "https://www.agrolink.com.br/problemas/mancha-bacteriana_1666.html"),
    5: ("Septoriose", "https://agriculture.basf.com/br/pt/conteudos/cultivos-e-sementes/tomate/saiba-mais-sobre-a-septoriose-na-cultura-do-tomate-e-manejo.html"),
    6: ("Pinta_Preta", "https://revistacultivar.com.br/artigos/como-controlar-pinta-preta-no-tomateiro"),
    7: ("Requeima", "https://revistacultivar.com.br/artigos/manejo-de-requeima-em-tomate"),
    8: ("Tomato_Yellow_Leaf_Curl_Virus", "https://plantix.net/pt/library/plant-diseases/200036/tomato-yellow-leaf-curl-virus/"),
    9: ("Mosaico_Virus_Y", "https://www.embrapa.br/agencia-de-informacao-tecnologica/cultivos/tomate/producao/doencas-e-pragas/doencas/virus/mosaico-do-virus-y")
}

def predict_image(img_path):
    with Image.open(img_path) as img:
        img = img.resize((256, 256))
        img_array = np.asarray(img)
        img_array = img_array.astype('float32') / 255.0  # Normalizando os pixels para o intervalo [0, 1]
        img_array = np.expand_dims(img_array, axis=0)  # Adicionando dimensão para o batch
        
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)[0]
    confidence = prediction[0][predicted_class] * 100  # Confiança associada à classe predita

    disease_name, disease_link = disease_links.get(predicted_class, ("Desconhecida", None))
    
    if disease_name == "Saudavel":
        return "A imagem refere-se a uma folha saudável", confidence
    elif disease_link:
        disease_name = disease_name.replace('_', ' ').upper()
        return f"A imagem é referente a doença de {disease_name} e o cuidado para tal você pode encontrar aqui: {disease_link}", confidence
    else:
        return "Doença desconhecida ou não mapeada", confidence

def get_disease_names():
    dataset_path = 'dataset/train'
    disease_names = [disease for disease in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, disease))]
    return disease_names

# Mostrar nomes das doenças treinadas
disease_names = get_disease_names()
st.sidebar.title("Doenças Treinadas")
for disease in disease_names:
    st.sidebar.write(disease.replace('_', ' ').title())

st.title("Classificação de Doenças em Plantas")
st.write("Faça upload de uma imagem de uma folha de tomate para verificar se há alguma doença.")

# Upload da imagem
uploaded_file = st.file_uploader("Escolha uma imagem...", type="jpg")

if uploaded_file is not None:
    # Mostrar a imagem carregada
    img = Image.open(uploaded_file)
    st.image(img, caption='Imagem Carregada', use_column_width=True)
    
    # Verificar se a imagem excede 256x256 pixels
    if img.width > 256 or img.height > 256:
        st.error("Por favor, escolha uma imagem com dimensões iguais ou inferiores a 256x256 pixels.")
    else:
        # Criar um arquivo temporário para a imagem carregada
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            img_path = tmp_file.name
            img.save(img_path)

        # Fazer a predição
        result, confidence = predict_image(img_path)
        st.write(result)
        st.write(f"Confiança da previsão: {confidence:.2f}%")
        
        # Remover o arquivo temporário após a predição
        os.remove(img_path)
