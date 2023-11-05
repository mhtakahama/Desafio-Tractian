Os arquivos do desafio estão dentro das pastas referentes a cada parte do desafio. O código foi desenvolvido dentro do prazo de teste (7 dias). 

# Desafio-Tractian Vaga: Data Scientist

### Parte 1 -  Data ETL, Data **Wrangling e Data Exploration**

Para começar, realize o download dos arquivos necessários no link acima.

**Zip instructions** - Arquivos de coleta de vibração crua:

- Os arquivos se encontram no formato CSV “*.csv”.
- Em cada arquivo existe uma coleta de vibração realizada pelo sensor “band-aid”.
- O nome do arquivo traz as seguintes informações: “{start}-{interval}-{sensor_id}.bin”
    - Exemplo: 1623535615-3006-IAJ9206.csv
        - start: 1623535615 [epoch Unix]
        - interval - sampling duration: 3006 [ms]
        - sensor_id: IAJ9206 [string que identifica um sensor]
- As coletas dizem respeito aos dados de aceleração em g nos eixos X, Y, Z de um acelerômetro.

<aside>
📌 **Utilizando estes arquivos você deve completar as seguintes etapas:**

1. Apresentar os dados contidos nos arquivos no domínio do tempo.
2. Apresentar os dados contidos nos arquivos no domínio da frequência (fft).
3. Aplicar filtros, se necessário, para limpar e corrigir os sinais da melhor forma possível.
4. [Bônus] Criar uma função capaz de identificar os harmônicos e picos no domínio da frequência, de maneira a reduzir a quantidade de dados e filtrar a informação relevante.
</aside>

### **Parte 2 - Machine Learning, Deep learning e Data Driven Solutions:**

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


# Desafio Parte 1

<img src="https://github.com/mhtakahama/Desafio-Tractian/blob/main/Diagramas/Desafio%20Tractian.png" alt="Figure 3" width="300">


# Desafio Parte 2

<img src="https://github.com/mhtakahama/Desafio-Tractian/blob/main/Diagramas/Desafio%20Tractian2.png" alt="Figure 3" width="300">

