# -*- coding: utf-8 -*-
"""
Desafio Tractian
Candidato: Msc. Marcos Hiroshi Takahama
Curriculum:
https://www.linkedin.com/in/mhtakahama/
http://lattes.cnpq.br/8034933372506302
Created on Sat Sep 30 17:12:30 2023
"""
"""
### ### **Parte 2 - Machine Learning, Deep learning e Data Driven Solutions:**

Assim como na etapa anterior você deve utilizar os arquivos disponibilizados, os quais podem ser encontrados aqui.

**Zip instructions** - Arquivos de coletas de vibração processados:

- Os arquivos se encontram no formato CSV “*.csv”.
- **collects.csv**: Contém uma lista de objetos que representam as coletas obtidas para diferentes ativos.
- **assets.csv**: Contém as informações sobre os ativos aos quais as coletas fornecidas pertencem.

<aside>
📌 **Considerando os dados contidos nesses arquivos você deve completar as seguintes etapas:**

1. Apresentar visualmente os dados contidos em cada arquivo, juntamente com as informações do ativo a que pertencem.
2. Desenvolver um modelo/função capaz de calcular o tempo de downtime e uptime para um ativo qualquer.
3. Desenvolver um modelo/função capaz de identificar mudanças nos padrões de vibração para um ativo qualquer.
4. Identificar possíveis falhas nos ativos utilizando o modelo desenvolvido no item 3 ou um novo modelo (a identificação deve ser autônoma e não uma análise visual).
</aside>

"""

"""
Comments in English, 
Variables in Portuguese to better a logical interpretation in case of abbreviations names
"""

# not common libraries :

# Libraries Used:  
import pandas as pd
import os
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


def calc_tempo_updown(data_frame_acessado,modelo_tipo): 
    if modelo_tipo=='heaterFurnace' or modelo_tipo=='transformer':
        sinal_analisado=data_frame_acessado['temp']
        rsq_medias=data_frame_acessado['temp'].mean()
    else:
        media_accel_x = data_frame_acessado['params.accelRMS.x'].mean()
        media_accel_y = data_frame_acessado['params.accelRMS.y'].mean()
        media_accel_z = data_frame_acessado['params.accelRMS.z'].mean()
        media_vel_x = data_frame_acessado['params.velRMS.x'].mean()
        media_vel_y = data_frame_acessado['params.velRMS.y'].mean()
        media_vel_z = data_frame_acessado['params.velRMS.z'].mean()
        
        medias=[media_accel_x,media_accel_y,media_accel_z,media_vel_x,media_vel_y,media_vel_z]
        
        sinal_analisado=[]
        # Assumindo que critério de utilização baseado no desvio e média dos sinais 
        # Varredura manual linha a linha
        for indice, linha in data_frame_acessado.iterrows():
            var1 = linha['params.accelRMS.x']
            var2=  linha['params.accelRMS.y']
            var3 = linha['params.accelRMS.z']
            var4 = linha['params.velRMS.x']
            var5 = linha['params.velRMS.y']
            var6 = linha['params.velRMS.z']
            
            raiz_soma_quadrados = np.sqrt(var1**2 + var2**2+ var3**2+ var4**2+ var5**2+ var6**2) #Root mean square
            sinal_analisado.append(raiz_soma_quadrados)
    
        rsq_medias = np.sqrt(np.sum(np.array(medias)**2)) 

    up_down=[] 
    for valor in sinal_analisado:  # Varredura de pontos
        if valor >= (rsq_medias): #1
               up_down.append(1)
        else:
               up_down.append(0)

    data_frame_acessado['up_down'] = up_down

    # Criar um booelano para verificar a mudança de estado
    data_frame_acessado['mudanca_estado']= data_frame_acessado['up_down'] != data_frame_acessado['up_down'].shift()
    
    # Indices para cada posicão referente a mudanca de estado
    data_frame_acessado['boolean_index'] = data_frame_acessado['mudanca_estado'].cumsum()

    # # Calcule a diferença entre os valores máximos e mínimos de 'timestamp' em cada grupo
    # tempos = data_frame_acessado['timestamp'].agg(lambda x: x.max() - x.min())  
    tempos = data_frame_acessado.groupby(['mudanca_estado', 'boolean_index'])['createdAt'].agg(lambda x: x.max() - x.min()).reset_index()
   
    # Alterar nomenclaturas para melhor compreensão do usuario
    tempos = tempos.rename(columns={'createdAt': 'Tempo'})
    mapeamento = {True: 'ligado', False: 'desligado'}
    tempos['mudanca_estado'] = tempos['mudanca_estado'].replace(mapeamento)
    return tempos, up_down

def RNA_datasensor(data_frame_acessado_vibration,type_model,nome): # Redes neurais
        # type_model='pattern_vibration' #Para padrão de vibracão
        # type_model = 'failure_detection' #Para classificação de falha
        
        #Preparar dados
        X = data_frame_acessado_vibration[['params.accelRMS.x', 'params.accelRMS.y', 'params.accelRMS.z', 'params.velRMS.x', 'params.velRMS.y', 'params.velRMS.z', 'temp']].values
       
        # Simular falhas definindo os valores como o máximo mais 2 vezes o desvio padrão multiplicado por um valor randômico entre 0 e 1 para cada coluna, uma vez para cada coluna
        num_fault_samples = 15  # Número de amostras com falhas simuladas
        
        # Lista de índices para adicionar falhas
        fault_indices = [0, 1, 2, 3, 4, 5, 6]  # Índices das colunas onde queremos adicionar falhas
        # fault_indices = [1, 1, 1, 1, 1, 1, 1]  # Índices das colunas onde queremos adicionar falhas
        # eixo x - 1
        # eixo y - 2
        # eixo z - 3
        # temperatura - 4 
        # Criar uma matriz para armazenar amostras com falhas
        fault_samples = np.zeros((num_fault_samples * len(fault_indices), X.shape[1]))
        
        # Adicionar falhas em uma única amostra em cada coluna
        for col_index in fault_indices:
            for i in range(num_fault_samples):
                random_value = np.random.uniform(0, 1)  # Valor randômico entre 0 e 1
                fault_sample = X[0].copy()
                # Criar rótulos para os dados (0 para operação normal, 1 para falha)
                if type_model == 'pattern_vibration':
                    fault_sample[col_index] = X[:, col_index].max() + 2 * X[:, col_index].std() * random_value
                elif type_model == 'failure_detection' :
                    fault_sample[col_index] = 4
                fault_samples[i + col_index * num_fault_samples] = fault_sample
                            
        # Concatenar as amostras normais com as amostras com falhas
        X = np.vstack((X, fault_samples))
        
        # Normaliza os dados de entrada de 0 a 1
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)
        # X2 = scaler.fit_transform(X) #checa

        # Criar rótulos para os dados (0 para operação normal, 1 para falha)
        if type_model == 'pattern_vibration':
            y = np.zeros(X.shape[0])
        elif type_model == 'failure_detection' :
            y = np.zeros((X.shape[0],4))
        # y = np.zeros((num_samples, num_columns))

        # patern = np.array([])
        if type_model == 'pattern_vibration':
            y[-(num_fault_samples * len(fault_indices)):] = np.repeat(1, num_fault_samples * 7)
        elif type_model == 'failure_detection' :

            # Defina um contador inicial
            cont1 = 0
            pattern2 = np.zeros((num_fault_samples*len(fault_indices),4))

            for i in range(3):
                for j in range(num_fault_samples):
                    pattern2[cont1][i] = 1
                    pattern2[cont1+num_fault_samples*3][i] = 1
                    cont1 += 1
                    
            loc1=((num_fault_samples*len(fault_indices)-1))
            loc2=num_fault_samples*len(fault_indices)
            pattern2[loc1:loc2, 3] = 1

            # Preencha as saidas com o padrao de falhas
            y[-(num_fault_samples * len(fault_indices)):]  = pattern2

        # Dividir os dados em treinamento e teste
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

        # Criar e treinar a rede neural
        model = Sequential() 
        if type_model == 'pattern_vibration':
            model.add(Dense(64, activation='relu', input_shape=(7,))) #Hidden layer 1 e input of size considering inputs
            model.add(Dense(64, activation='relu')) #Hidden layer 2 
            model.add(Dense(1, activation='sigmoid')) #output layer
        
        elif type_model == 'failure_detection' :
            # y[-(num_fault_samples * len(fault_indices)):]  = np.repeat(range(1, 8), num_fault_samples)
            model.add(Dense(64, activation='relu', input_shape=(7,))) #Hidden layer 1 e input of size considering inputs
            model.add(Dense(64, activation='relu')) #Hidden layer 2 
            model.add(Dense(4, activation='Softmax')) #output layer 5 saidas e softmax para multiclasse
            
        # model.add(Dense(64, activation='relu', input_shape=(7,))) #Hidden layer 1 e input of size considering inputs
        # model.add(Dense(64, activation='relu')) #Hidden layer 2 
        # model.add(Dense(1, activation='sigmoid')) #output layer
        
        if type_model == 'pattern_vibration':
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']) #classificação binaria
        elif type_model == 'failure_detection' :
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) # para os 7 padroes
        
        history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1) #50 epocas, 10% dos dados de entrada para validação
        
        # Avaliar a rede neural nos dados de teste
        test_loss, test_accuracy = model.evaluate(X_test, y_test)
        print(f"Acurácia nos dados de teste: {test_accuracy}")
        
        plt.figure(figsize=(16,10))
        plt.subplot(2, 1, 1)
        graph_title1 = "Sensor {} - Treinamento Rede neural (RN) - LSTM \n ".format(nome)
        graph_title2 = " {} ".format(type_model)
        graph_title= graph_title1+graph_title2
        plt.title(graph_title)
        plt.plot(history.history['accuracy'], label='Acurácia no treinamento', color='blue')
        plt.plot(history.history['val_accuracy'], label='Acurácia na validação', color='red')
        plt.xlabel('Épocas')
        plt.ylabel('Acurácia')
        plt.tight_layout()
        plt.legend()
        
        # Aplicar a rede neural aos dados de entrada
        y_pred = model.predict(X_test)
        
        # Plotar os resultados
        if type_model == 'pattern_vibration':
            raiz_soma_quadrados_test = y_test
            raiz_soma_quadrados_pred = y_pred  
        elif type_model == 'failure_detection' :
            raiz_soma_quadrados_test = np.sqrt(y_test[:,0]**2 + y_test[:,1]**2+ y_test[:,2]**2+ y_test[:,3]**2) #Root mean square
            raiz_soma_quadrados_pred = np.sqrt(y_pred[:,0]**2 + y_pred[:,1]**2+ y_pred[:,2]**2+ y_pred[:,3]**2) #Root mean square  
        plt.subplot(2, 1, 2)
        plt.scatter(range(len(raiz_soma_quadrados_test)), raiz_soma_quadrados_test, label='Real', marker='o', color='blue')
        plt.scatter(range(len(raiz_soma_quadrados_pred)), raiz_soma_quadrados_pred, label='Previsão', marker='x', color='red')
        plt.xlabel('Amostras')
        plt.ylabel('Saída')
        plt.legend()
        plt.title('Teste de previsão nos dados de entrada \n Resultados da Previsão vs. Real')
        plt.show()

        return model 

diretorio_atual = os.getcwd()  # Dir of currently code

nome_do_diretorio = "RNA_criada" #Cria uma pasta para armazenar as redes neurais criadas
caminho_completo = os.path.join(os.getcwd(), nome_do_diretorio)
if not os.path.exists(caminho_completo):
    os.makedirs(caminho_completo)

# Section 1 - setup, load files and organize all data

# arquivos_csv = [arquivo for arquivo in arquivos_na_pasta if arquivo.endswith('.csv')] # Only .csv files
assets = pd.read_csv(os.path.join(diretorio_atual, 'assets.csv'))
collects = pd.read_csv(os.path.join(diretorio_atual, 'collects.csv'))

# processar para cada sensor
sensor_names = assets['sensors']

# Obter valores únicos na coluna 'Nome'
sensor_names = assets['sensors'].unique()

#removendo aspas e colchetes, considerando que é recorrente desta forma, caso contrário fazer varedura de string
cont_name=0
for nome in sensor_names:
    # print(nome[2:-2]) #check name
    sensor_names[cont_name] = nome[2:-2]
    cont_name+=1

# Criar um dicionário de DataFrames separados por nome
dataframes_por_sensor = {}

# Iterar pelos nomes únicos e criar DataFrames separados 
for nome in sensor_names:
    dataframes_por_sensor[nome] = collects[collects['sensorId'] == nome]


# Section 2 - Show data classified by sensor

contador_grafico=0

# Varredura por data frames
for nome in sensor_names:
    
    for contador, nome_sensor in enumerate(sensor_names):
        if nome == sensor_names[contador]:
            numero_correspondente=contador #encontra o numero correspondente ao nome para array

    data_frame_acessado=dataframes_por_sensor[nome] #Carrega os dados de um sensor
    
    data_frame_acessado = data_frame_acessado.copy() #deixar esta linha, o dataframe não permite alteração se não for explicito
    # See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
    # Converter a coluna Tempo_ISO8601 em objetos datetime
    data_frame_acessado['createdAt'] = pd.to_datetime(data_frame_acessado['createdAt'])
    
    # Apagar as linhas quando "nan" for encontrado (neste caso considerando qualquer ocorrencia)
    data_frame_acessado = data_frame_acessado.dropna(subset=['params.accelRMS.x'])
    # trocar por 0 quando "nan" for encontrado (neste caso considerando qualquer ocorrencia)
    # data_frame_acessado.fillna(0, inplace=True)
    
    # Ordenar o DataFrame com base na data (garantir que os dados não vieram embaralhados)
    data_frame_acessado.sort_values(by='createdAt', inplace=True)  
        
    # Assets
    empresa=assets['company'].iloc[numero_correspondente]
    maquina=assets['name'].iloc[numero_correspondente]
    
    modelo_tipo=assets['modelType'].iloc[numero_correspondente]
    inicio=assets['createdAt'].iloc[numero_correspondente]
    inicio_datetime = datetime.fromisoformat(inicio)
    inicio_dias = f'{inicio_datetime.day:02d}/{inicio_datetime.month:02d}/{inicio_datetime.year}'
    sensor_name=nome
    valor_maximo_accelrms = data_frame_acessado[['params.accelRMS.x', 'params.accelRMS.y', 'params.accelRMS.z']].max().max()
    valor_minimo_accelrms = data_frame_acessado[['params.accelRMS.x', 'params.accelRMS.y', 'params.accelRMS.z']].min().min()
    valor_maximo_vellrms = data_frame_acessado[['params.velRMS.x', 'params.velRMS.y', 'params.velRMS.z']].max().max()
    valor_minimo_vellrms = data_frame_acessado[['params.velRMS.x', 'params.velRMS.y', 'params.velRMS.z']].min().min()
    
    # Aceleração

    #Specs for each graph
    xyz='X'
    direcao_correspondente=assets['specifications.axisX'].iloc[numero_correspondente]
    maxrms=data_frame_acessado['params.accelRMS.x'].max()
    samprate_max=data_frame_acessado['params.sampRate'].max()
    samprate_min=data_frame_acessado['params.sampRate'].min()
    
    maxtemp_esp=assets['specifications.maxTemp'].iloc[numero_correspondente]
    maxtemp_reg=data_frame_acessado['temp'].max()
    
    # Cria um gráfico do sinal no dominio do tempo
    contador_grafico+= 1
    plt.figure(figsize=(20,10))
    plt.subplot(7, 1, 1)
    graph_title1 = " (Empresa {}) {} - {} \n Tipo: {} - {} - Sensor {} \n".format(contador_grafico, empresa, maquina, modelo_tipo, inicio_dias, sensor_name)
    graph_title2 = " Aceleração Direção '{}' - {} - {} máx rms (g)- Samprate {}  Hz~ {} Hz".format(xyz, direcao_correspondente, round(maxrms,2), round(samprate_max), round(samprate_min))
    graph_title= graph_title1+graph_title2
    plt.title(graph_title)
    plt.plot(data_frame_acessado['createdAt'], data_frame_acessado['params.accelRMS.x'], linestyle='-', color='red')
    plt.ylim(valor_minimo_accelrms, valor_maximo_accelrms)
    # plt.ylabel('Amplitude rms (g)')
    plt.grid(True)
    plt.tight_layout()
    
    #Specs for each graph
    xyz='Y'
    direcao_correspondente=assets['specifications.axisY'].iloc[numero_correspondente]
    maxrms=data_frame_acessado['params.accelRMS.y'].max()
    
    plt.subplot(7, 1, 2)
    graph_title = " Aceleração Direção '{}' - {} - {} máx rms (g)".format(xyz, direcao_correspondente, round(maxrms,2))
    plt.title(graph_title)
    plt.plot(data_frame_acessado['createdAt'], data_frame_acessado['params.accelRMS.y'], linestyle='-', color='red')
    plt.ylim(valor_minimo_accelrms, valor_maximo_accelrms)
    plt.ylabel('Amplitude rms (g)')
    plt.grid(True)
    plt.tight_layout()
    
    #Specs for each graph
    xyz='Z'
    direcao_correspondente=assets['specifications.axisZ'].iloc[numero_correspondente]
    maxrms=data_frame_acessado['params.accelRMS.z'].max()
    
    plt.subplot(7, 1, 3)
    graph_title = " Aceleração Direção '{}' - {} - {} máx rms (g)".format(xyz, direcao_correspondente, round(maxrms,2))
    plt.title(graph_title)
    plt.plot(data_frame_acessado['createdAt'], data_frame_acessado['params.accelRMS.z'], linestyle='-', color='red')
    plt.ylim(valor_minimo_accelrms, valor_maximo_accelrms)
    # plt.ylabel('Amplitude rms (g)')
    plt.grid(True)
    plt.tight_layout()

    # Velocidade

    #Specs for each graph
    xyz='X'
    direcao_correspondente=assets['specifications.axisX'].iloc[numero_correspondente]
    maxrms=data_frame_acessado['params.velRMS.x'].max()
    
    plt.subplot(7, 1, 4)
    graph_title = " Velocidade Direção '{}' - {} - {} máx rms (m/s)".format(xyz, direcao_correspondente, round(maxrms,2))
    plt.title(graph_title)
    plt.plot(data_frame_acessado['createdAt'], data_frame_acessado['params.velRMS.x'], linestyle='-', color='blue')
    plt.ylim(valor_minimo_vellrms, valor_maximo_vellrms)
    # plt.ylabel('Amplitude rms (m/s)')
    plt.grid(True)
    plt.tight_layout()
    
    #Specs for each graph
    xyz='Y'
    direcao_correspondente=assets['specifications.axisY'].iloc[numero_correspondente]
    maxrms=data_frame_acessado['params.velRMS.y'].max()
    
    plt.subplot(7, 1, 5)
    graph_title = " Velocidade Direção '{}' - {} - {} máx rms (m/s)".format(xyz, direcao_correspondente, round(maxrms,2))
    plt.title(graph_title)
    plt.plot(data_frame_acessado['createdAt'], data_frame_acessado['params.velRMS.y'], linestyle='-', color='blue')
    plt.ylim(valor_minimo_vellrms, valor_maximo_vellrms)
    plt.ylabel('Amplitude rms (m/s)')
    plt.grid(True)
    plt.tight_layout()
    
    #Specs for each graph
    xyz='Z'
    direcao_correspondente=assets['specifications.axisZ'].iloc[numero_correspondente]
    maxrms=data_frame_acessado['params.velRMS.z'].max()
    
    plt.subplot(7, 1, 6)
    graph_title = " Velocidade Direção '{}' - {} - {} máx rms (m/s)".format(xyz, direcao_correspondente, round(maxrms,2))
    plt.title(graph_title)
    plt.plot(data_frame_acessado['createdAt'], data_frame_acessado['params.velRMS.z'], linestyle='-', color='blue')
    plt.ylim(valor_minimo_vellrms, valor_maximo_vellrms)
    # plt.ylabel('Amplitude rms (m/s)')
    plt.grid(True)
    plt.tight_layout()
    
    #Specs for each graph
    xyz='Z'
    direcao_correspondente=assets['specifications.axisZ'].iloc[numero_correspondente]
    maxrms=data_frame_acessado['params.velRMS.z'].max()
    
    # Temperatura
    
    plt.subplot(7, 1, 7)
    graph_title = " Temperatura (°C) - Máxima registrada {} - Máxima especificada {}".format(maxtemp_reg, maxtemp_esp)
    plt.title(graph_title)
    plt.plot(data_frame_acessado['createdAt'], data_frame_acessado['temp'], linestyle='-', color='green', label='Temperatura')
    plt.ylabel('Temperatura (°C)')
    plt.grid(True)
    plt.tight_layout()
    
# Section 3 - Estimate downtime and uptime 
    
    #tempos armazena os up e downtimes e up_down os valores booleanos
    tempos,up_down=calc_tempo_updown(data_frame_acessado,modelo_tipo)
    
    up_down_normalizado = [x * maxtemp_reg*0.3 for x in up_down]    
    plt.plot(data_frame_acessado['createdAt'], up_down_normalizado, linestyle='-', color='orange', label='uptime') #plot updowntime com escala normalizada junto com a temperatura
    plt.legend()

# Section 4 - Pattern vibration 
    # Incluir o booleano que identfica o uptime e downtime no dataframe analisado
    data_frame_acessado_vibration=[]
    if modelo_tipo!='heaterFurnace' or modelo_tipo!='transformer':
            data_frame_acessado['up_down']=up_down
            # Exclua as linhas que consideram o ativo parado igual a 0
            data_frame_acessado_vibration = data_frame_acessado.loc[data_frame_acessado['up_down'] == 1]

            model = RNA_datasensor(data_frame_acessado_vibration,'pattern_vibration',nome)
            model.save(caminho_completo+'\RNA_PV_'+nome+'.h5')
            
# Section 5 - Failure type detector

            model = RNA_datasensor(data_frame_acessado_vibration,'failure_detection',nome)
            model.save(caminho_completo+'\RNA_DF_'+nome+'.h5')
  