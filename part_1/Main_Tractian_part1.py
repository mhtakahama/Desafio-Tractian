# -*- coding: utf-8 -*-
"""
Desafio Tractian
Candidato: Msc. Marcos Hiroshi Takahama
Curriculum:
https://www.linkedin.com/in/mhtakahama/
http://lattes.cnpq.br/8034933372506302
Created on Sat Sep 30 17:12:30 2023
"""

# Libraries Used:  
import os 
import pandas as pd 
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal #somente para filtros de frequencia


# Section 0 - Functions

def identificar_picos_fft(fft, frequencias, limiar, max_multiplo): #sem usar scipy
 
    # Identifica os picos na parte positiva da FFT com base no limiar
    picos = np.where(np.abs(fft[1:]) > limiar)[0] + 1  # encontra todos os picos com base no limiar, ignora o primeiro ponto
                                                        #  +1 -Corrigo os índices para incluir o primeiro ponto
    picos_positivos = [valor for valor in picos if valor < (len(fft)//2)] #somente positivos
    
    # # Identifica os harmonicos entre os picos positivos fft
    
    harmonicos = []
    for i in range(len(picos_positivos)):  #Varredura do valor de referencia
        valor_referencia = picos_positivos[i] 
        multiplos = []
    
        for j in range(len(picos_positivos)):
            if i != j and picos_positivos[j] % valor_referencia == 0: #i != j Compara os demais valores j em relação ao valor de referencia i
                multiplos.append(picos_positivos[j])  #lista os valores multiplos j do valor de referencia i
    
        harmonicos.append([valor_referencia, multiplos]) #armazena em uma tupla valor de referencia + multiplos
    
        if len(multiplos) >= 2:  # Verifique se a lista de picos tem 2 ou mais elementos
            harmonicos.append([valor_referencia, multiplos])
    
    lista_comparacao = [tupla for tupla in harmonicos if len(tupla[1]) > 0] # Exclui valores individuais e armazena só os picos que contem múltiplos
    
    lista_harmonicos = [[i[0]] + i[1] for i in lista_comparacao] # Ajusta o objeto 

    return picos_positivos,lista_harmonicos

# Section 1 - setup, load files and organize all data
 
diretorio_atual = os.getcwd()#  Dir of currently code
arquivos_na_pasta = os.listdir(diretorio_atual)# All files of current dir
arquivos_csv = [arquivo for arquivo in arquivos_na_pasta if arquivo.endswith('.csv')] # Only .csv files

contador_grafico=0
# List of content in each name file and get file info
for nome_arquivo in arquivos_na_pasta:
    # Verify if each file is a .csv and read
    if nome_arquivo.endswith('.csv'):
        caminho_arquivo = os.path.join(diretorio_atual, nome_arquivo)
        data_full = pd.read_csv(caminho_arquivo) #read all data in .csv file (first line is column name)
        num_rows, num_columns = data_full.shape

        # Title file name
        #indentify and split char separator
        partes = nome_arquivo.split("-")   
        
        data_hora = partes[0]
        data_hora =  datetime.utcfromtimestamp(int(data_hora)) #convert unix format to plot
        
        duracao_amostragem = int(partes[1])/1000 #total duration in seconds
        dt = duracao_amostragem/num_rows # sampling interval
        t = np.arange(0, duracao_amostragem, dt) #vetor temporal
        fs = 1 / dt # sampling frequency
        frequencies = np.fft.fftfreq(num_rows, 1/fs)  # Frequency vector
        positive_indices = frequencies >= 0

        sensor_name = partes[2]
        sensor_name = sensor_name.split(".")
        sensor_name = sensor_name[0] # Sensor name
        
# Section 2 - Visualizar sinal e FFT

# Apply Fast Fourier Transform on each column 'x', 'y' and 'z'
        for direcoes_sinal in ['x', 'y', 'z']:
                
            # Calcula a FFT dos dados na direcoes_sinal
            sinal = data_full[direcoes_sinal].values
            fft_ajustada= np.fft.fft(sinal)/ len(sinal) #fft em escala de magnitude

            limiar=max(np.abs(fft_ajustada[1:]))*0.2 #Define o limiar minimo para identificar o pico sendo 20% da magnitude do maior pico da rotação
            picos_positivos,lista_harmonicos= identificar_picos_fft(fft_ajustada, frequencies, limiar, 4)   #procura os picos e harmonicos com base no limiar            
            rms = np.sqrt(np.mean(sinal**2))             # Calcula o RMS

            # Cria um gráfico do sinal no dominio do tempo
            contador_grafico+= 1
            plt.figure(figsize=(20,10))
            plt.subplot(5, 1, 1)
            graph_title = "Grafico {} - sensor {} - Data da cooleta: {} \n Sinal em '{}' \n 100 primeiros pontos do sinal".format(contador_grafico,sensor_name,data_hora,direcoes_sinal)
            plt.title(graph_title)
            plt.plot(t[:100], sinal[:100])
            plt.xlabel('Tempo (s)')
            plt.ylabel('Amplitude (g)')
            plt.grid(True)
            plt.tight_layout()
            
            plt.subplot(5, 1, 2)
            graph_title = "Sinal completo - RMS {} (g)".format(round(rms,2))
            plt.title(graph_title)
            plt.plot(t, sinal)
            plt.xlabel('Tempo (s)')
            plt.ylabel('Amplitude (g)')
            plt.grid(True)
            plt.tight_layout() 
           
            plt.subplot(5, 1, 3)
            graph_title = "Espectro de frequência (FFT) completo direção '{}'".format(direcoes_sinal)
            plt.title(graph_title)
            indices = (frequencies >= 1)

            # indices=(frequencies > 0) #exclui os 2 primeiros pontos
            plt.plot(frequencies[indices],np.abs(fft_ajustada[indices])) #Apenas parte positiva, a partir do 2
                        
          
            # Plote cada conjunto de pontos
            frequencias_interesse = [] #Variavel caso queira armazenar a lista dos harmonicos
            amplitudes_harmonicas = [] #Variavel caso queira armazenar a amplitude respectiva das frequencias
            
            harmonicos=0 #contador provisorio, por algum motivo o contador direto bugou, resolver futuramente
            for i in lista_harmonicos:
                frequencias_harmonicas=frequencies[lista_harmonicos[harmonicos]]
                amplitudes_harmonicas= np.abs(fft_ajustada[lista_harmonicos[harmonicos]])
                frequencias_interesse.append(frequencias_harmonicas) #armazena em uma tupla valor de referencia + multiplos
                
                plt.scatter(frequencias_harmonicas,amplitudes_harmonicas)
                harmonicos+=1
                
            plt.xlabel('Frequência')
            plt.ylabel('Amplitude normalizada (magnitude)')
            plt.grid(True)
            plt.tight_layout()
     
         
# # Section 3 - Frequências de interesse e bandpass filters
        # Comuns
            # 1 - Frequências de Rotação e engrenagens, 60hz (motor eletrico 3600rpm, considerar 500hz para analise de mais harmonicos)
                # Desbalanceamento, Desalinhamento, Eixo Empenado,Falta de lubrificação
            # 2 - Usar filtro Notch caso queira eliminar as frequencias de 60hz e 120hz do sinal
        	
            # Frequência de corte do filtro passa-baixa
            frequencia_de_corte = 500  # Frequência de corte em Hz
            ordem = 4  # Ordem do filtro
            b, a = signal.butter(ordem, frequencia_de_corte / (fs / 2), btype='low')
            sinal_passabaixa = signal.lfilter(b, a, sinal)
            fft_passabaixa= np.fft.fft(sinal_passabaixa)/ len(sinal_passabaixa) #fft em escala de magnitude
            
            #Identifica os picos da fft por treshold baseado na magnitude do maior pico de fft
            limiar=max(np.abs(fft_passabaixa[1:]))*0.2 #procura os picos com base em 50% da magnitude do maior pico da rotação
            picos_positivos,_= identificar_picos_fft(fft_passabaixa, frequencies, limiar, 4)
            
            plt.subplot(5, 1, 4)
            graph_title = "Espectro de frequência até 500Hz (FFT), filtro passa baixa na direção '{}'".format(direcoes_sinal)
            plt.title(graph_title)
            
            # indices = (frequencies < 500)
            indices = (frequencies >= 1) 
            plt.plot(frequencies[indices],np.abs(fft_passabaixa[indices])) #Apenas parte positiva, a partir do 2
            # Identificar os picos na FFT
            plt.plot(frequencies[picos_positivos], np.abs(fft_passabaixa)[picos_positivos], 'ro', markersize=8, label='Picos') #picos da fft           
            plt.xlabel('Frequência')
            plt.ylabel('Amplitude normalizada (magnitude)')
            plt.grid(True)
            plt.tight_layout()
                 
        
            # 2 - Frequências de falha em rolamentos 500 a 2kHz BPFO e BPFI, varia de acordo com as dimensÕes do rolamento e rotação
            
            b, a = signal.butter(ordem, frequencia_de_corte / (fs / 2), btype='high')
            sinal_alta= signal.lfilter(b, a, sinal)
            fft_passaalta= np.fft.fft(sinal_alta)/ len(sinal_alta) #fft em escala de magnitude
            
            #Identifica os picos da fft por treshold baseado na magnitude do maior pico de fft
            limiar=max(np.abs(fft_passaalta[1:]))*0.3 #procura os picos com base em 50% da magnitude do maior pico da rotação
            picos_positivos,_= identificar_picos_fft(fft_passaalta, frequencies, limiar, 4)

            plt.subplot(5, 1, 5)
            graph_title = "Espectro de frequência BPFO e BPFI, filtro passa alta na direção '{}'".format(direcoes_sinal)
            plt.title(graph_title)
            
            # indices = (frequencies > 500)
            indices = (frequencies >= 1)
            plt.plot(frequencies[indices],np.abs(fft_passaalta[indices])) #Apenas parte positiva, a partir do 2
            # Identificar os picos na FFT
            plt.plot(frequencies[picos_positivos], np.abs(fft_passaalta)[picos_positivos], 'ro', markersize=8, label='Picos')
            plt.xlabel('Frequência')
            plt.ylabel('Amplitude normalizada (magnitude)')
            plt.grid(True)
            plt.tight_layout()
            plt.show()