#==================================================================================================================================#
#                                                   PACOTES NECESSÁRIOS                                                            #
#==================================================================================================================================#

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

#==================================================================================================================================#
#                                                   TRATAMENTO DOS DADOS                                                           #
#==================================================================================================================================#

# Lendo o arquivo CSV
epl = pd.read_csv("epl_final.csv", sep=",", encoding="utf-8")

# Criando um dicionário de dados
dicionario= {
    'Temporada': 'A temporada de futebol',
    'Data da Partida': 'A data em que a partida foi disputada',
    'Time da Casa': 'Nome do time da casa',
    'Time Visitante': 'Nome do time visitante',
    'Gols Casa': 'Gols marcados pelo time da casa (tempo integral)',
    'Gols Fora': 'Gols marcados pelo time visitante (tempo integral)',
    'Resultado no Fim do Jogo': 'Resultado da partida (H = Vitória em casa, A = Vitória fora, D = Empate)',
    'Gols em Casa no Intervalo': 'Gols marcados pelo time da casa (intervalo)',
    'Gols Fora no Intervalo': 'Gols marcados pelo time visitante (intervalo)',
    'Resultado no Intervalo': 'Resultado do intervalo (H = vitória da casa, A = vitória do visitante, D = empate)',
    'Chutes Casa': 'Total de chutes do time da casa',
    'Chutes Fora': 'Total de chutes do time visitante',
    'Chutes em Casa no Gol': 'Chutes a gol do time da casa',
    'Chutes Fora no Gol': 'Chutes a gol do time visitante',
    'Escanteios em Casa': 'Número de escanteios ganhos pelo time da casa',
    'Escanteios Fora': 'Número de escanteios ganhos pelo time visitante',
    'Faltas em Casa': 'Número de faltas cometidas pelo time da casa',
    'Faltas Fora': 'Número de faltas cometidas pelo time visitante',
    'Cartões Amarelos Casa': 'Cartões amarelos recebidos pelo time da casa',
    'Cartões Amarelos Fora': 'Cartões amarelos recebidos pelo time visitante',
    'Cartões Vermelhos Casa': 'Cartões vermelhos recebidos pelo time da casa',
    'Cartões Vermelhos Fora': 'Cartões vermelhos recebidos pelo time visitante'
}

# Criando um DataFrame a partir do dicionário
dicionario = pd.DataFrame(
    list(dicionario.items()),
    columns=['Variável', 'Descrição'],
    index=None
)

# Resetando o índice do DataFrame
dicionario.reset_index(drop=True, inplace=True)

# Criando uma lista de colunas em português
colunas_pt = [
'Temporada', 'Data da Partida', 'Time da Casa', 'Time Visitante', 'Gols Casa',
'Gols Fora', 'Resultado no Fim do Jogo', 'Gols em Casa no Intervalo',
'Gols Fora no Intervalo', 'Resultado no Intervalo', 'Chutes Casa', 'Chutes Fora',
'Chutes em Casa no Gol', 'Chutes Fora no Gol', 'Escanteios em Casa', 'Escanteios Fora',
'Faltas em Casa', 'Faltas Fora', 'Cartões Amarelos Casa', 'Cartões Amarelos Fora',
'Cartões Vermelhos Casa', 'Cartões Vermelhos Fora'
]

# Renomeando as colunas
epl.columns = colunas_pt

# 1. Criar uma nova coluna com o ano inicial da temporada como número inteiro
epl['AnoInicial'] = epl['Temporada'].str[:4].astype(int)

# 2. Filtrar as temporadas entre 2009 e 2020
df_filtrado = epl[(epl['AnoInicial'] >= 2009) & (epl['AnoInicial'] <= 2023)]

# Filtrar jogos em que o time da casa venceu no intervalo
df_intervalo_vantagem = df_filtrado[df_filtrado['Resultado no Intervalo'] == 'H'].copy()

# Agrupar por temporada e calcular totais
confirmacao_vitoria = (
    df_intervalo_vantagem
    .groupby('Temporada')
    .agg(
        Total_Jogos=('Resultado no Fim do Jogo', 'count'),
        Vitorias_Confirmadas=('Resultado no Fim do Jogo', lambda x: (x == 'H').sum())
    )
    .reset_index()
)

# Calcular porcentagem de confirmação
confirmacao_vitoria['% Confirmaram Vitória'] = (
    100 * confirmacao_vitoria['Vitorias_Confirmadas'] / confirmacao_vitoria['Total_Jogos']
)

# Calcular a média ponderada
total_vitorias_confirmadas = confirmacao_vitoria['Vitorias_Confirmadas'].sum()
total_jogos = confirmacao_vitoria['Total_Jogos'].sum()

media_ponderada = (total_vitorias_confirmadas / total_jogos) * 100

# Agrupando por temporada
eficiencia_temporada = (
    df_filtrado
    .groupby('Temporada')
    .agg(
        ChutesTotais=('Chutes em Casa no Gol', 'sum'),
        ChutesVisitantes=('Chutes Fora no Gol', 'sum'),
        GolsCasa=('Gols Casa', 'sum'),
        GolsVisitante=('Gols Fora', 'sum')
    )
    .reset_index()
)

# Somando os valores para totalizar por temporada
eficiencia_temporada['ChutesTotal'] = eficiencia_temporada['ChutesTotais'] + eficiencia_temporada['ChutesVisitantes']
eficiencia_temporada['GolsTotal'] = eficiencia_temporada['GolsCasa'] + eficiencia_temporada['GolsVisitante']

# Eficiência de finalização
eficiencia_temporada['Eficiência (%)'] = (eficiencia_temporada['GolsTotal'] / eficiencia_temporada['ChutesTotal']) * 100

# Calculando eficiências separadas
eficiencia_temporada['Eficiência Casa (%)'] = (eficiencia_temporada['GolsCasa'] / eficiencia_temporada['ChutesTotais']) * 100
eficiencia_temporada['Eficiência Visitantes (%)'] = (eficiencia_temporada['GolsVisitante'] / eficiencia_temporada['ChutesVisitantes']) * 100

#==================================================================================================================================#
#                                                  CONSTRUÇÃO DOS GRÁFICOS                                                         #
#==================================================================================================================================#

# Gráfico 1: Total de Gols por Temporada - Casa vs Visitante
# Agrupar por temporada e somar os gols
gols_por_temporada = df_filtrado.groupby('Temporada').agg({
    'Gols Casa': 'sum',
    'Gols Fora': 'sum'
}).reset_index()

# Transformar em formato longo para facilitar o gráfico
gols_melted = gols_por_temporada.melt(id_vars='Temporada', 
                                      value_vars=['Gols Casa', 'Gols Fora'],
                                      var_name='Tipo de Gol',
                                      value_name='Quantidade de Gols')

# Ordenar corretamente a temporada (por ano inicial)
gols_melted['AnoInicial'] = gols_melted['Temporada'].str[:4].astype(int)
gols_melted = gols_melted.sort_values('AnoInicial')

# Criar o gráfico
fig1 = px.line(
    gols_melted,
    x='Temporada',
    y='Quantidade de Gols',
    color='Tipo de Gol',
    markers=True,
    #title='Quantidade de Gols por Temporada - Casa vs Visitante'
)

fig1.update_layout(
    xaxis_title='Temporada',
    yaxis_title='Total de Gols',
    template= 'plotly_white',
)

fig1.add_annotation(
    x='2020/21',
    y=gols_melted[(gols_melted['Temporada'] == '2020/21') & (gols_melted['Tipo de Gol'] == 'Gols Casa')]['Quantidade de Gols'].values[0],
    text="⬇ Redução nos gols dos mandantes<br>⬆ Aumento nos gols visitantes<br><b>Pandemia: jogos sem torcida</b>",
    showarrow=True,
    arrowhead=1,
    ax=-100,
    ay=-100,
    bgcolor="rgba(255,255,255,0.8)",
    bordercolor="black",
    font=dict(size=12)
)

# Gráfico 2: Total de Vitórias por Temporada - Casa vs Visitante
# Contar vitórias por temporada
vitorias_por_temporada = df_filtrado.groupby('Temporada')['Resultado no Fim do Jogo'].value_counts().unstack().fillna(0)

# Selecionar apenas vitórias da casa e fora
vitorias_por_temporada = vitorias_por_temporada[['H', 'A']].reset_index()
vitorias_por_temporada = vitorias_por_temporada.rename(columns={
    'H': 'Vitórias Casa',
    'A': 'Vitórias Fora'
})

# Formato longo para plotar
vitorias_melted = vitorias_por_temporada.melt(id_vars='Temporada',
                                              value_vars=['Vitórias Casa', 'Vitórias Fora'],
                                              var_name='Tipo de Vitória',
                                              value_name='Quantidade de Vitórias')

# Ordenar temporadas por ano
vitorias_melted['AnoInicial'] = vitorias_melted['Temporada'].str[:4].astype(int)
vitorias_melted = vitorias_melted.sort_values('AnoInicial')

# Gráfico
fig2 = px.line(
    vitorias_melted,
    x='Temporada',
    y='Quantidade de Vitórias',
    color='Tipo de Vitória',
    markers=True,
    #title='Quantidade de Vitórias por Temporada - Casa vs Fora'
)

fig2.update_layout(
    xaxis_title='Temporada',
    yaxis_title='Total de Vitórias',
    template='plotly_white',
)

# Adicionar anotação para 2020/21 (efeito pandemia)
fig2.add_annotation(
    x='2020/21',
    y=vitorias_melted[(vitorias_melted['Temporada'] == '2020/21') & (vitorias_melted['Tipo de Vitória'] == 'Vitórias Casa')]['Quantidade de Vitórias'].values[0],
    text="⬇ Menos vitórias da casa<br>⬆ Mais vitórias visitantes<br><b>Pandemia: jogos sem torcida</b>",
    showarrow=True,
    arrowhead=1,
    ax=-100,
    ay=-100,
    bgcolor="rgba(255,255,255,0.8)",
    bordercolor="black",
    font=dict(size=12)
)

# Gráfico 3: Porcentagem de Vitórias Confirmadas (Casa)
fig3 = px.line(
    confirmacao_vitoria,
    x='Temporada',
    y='% Confirmaram Vitória',
    markers=True,
    #title='Porcentagem de Vitórias Confirmadas - Casa vencendo no Intervalo',
)

fig3.update_layout(
    xaxis_title='Temporada',
    yaxis_title='% de Confirmação da Vitória',
    template='plotly_white',
    yaxis_tickformat=".1f"
)

# Alterando a escala do eixo y para minimo 0 e maximo 100
fig3.update_yaxes(range=[0, 100])
# Adicionando uma linha horizontal em 50%
fig3.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="70% de Confirmação", annotation_position="bottom right")

# Gráfico de Velocimetro
fig4 = go.Figure(go.Indicator(
    mode="gauge+number+delta",
    value=media_ponderada,
    #title={'text': "Média Ponderada de Confirmação de Vitórias"},
    gauge={
        'axis': {'range': [0, 100]},
        'bar': {'color': "royalblue"},
        'steps': [
            {'range': [0, 50], 'color': "lightcoral"},
            {'range': [50, 75], 'color': "gold"},
            {'range': [75, 100], 'color': "lightgreen"}
        ],
        'threshold': {
            'line': {'color': "red", 'width': 4},
            'thickness': 0.75,
            'value': media_ponderada
        }
    }
))

fig4.update_layout(
    height=400,
    margin=dict(t=50, b=0, l=0, r=0)
)

# Gráfico 5: Eficiência em Finalizações
# Gráfico original
fig5 = px.line(
    eficiencia_temporada,
    x='Temporada',
    y='Eficiência (%)',
    #title='Eficiência de Finalização por Temporada (Gols / Chutes Totais)',
    markers=True
)

# Encontrar valores das temporadas 2012/13 e 2013/14
x0 = '2012/13'
x1 = '2013/14'

y0 = eficiencia_temporada.loc[eficiencia_temporada['Temporada'] == x0, 'Eficiência (%)'].values[0]
y1 = eficiencia_temporada.loc[eficiencia_temporada['Temporada'] == x1, 'Eficiência (%)'].values[0]

# Adiciona seta
fig5.add_annotation(
    x=x1,
    y=y1,
    axref='x',
    ayref='y',
    ax=x0,
    ay=y0,
    showarrow=True,
    arrowhead=3,
    arrowsize=1,
    arrowwidth=2,
    arrowcolor='green'
)

# Adiciona anotação de texto
fig5.add_annotation(
    x=x1,
    y=y1 + 0.9,  # levemente acima da seta
    text='Aumento significativo de eficiência\n(2013/14)',
    showarrow=False,
    font=dict(size=12, color='green'),
    bgcolor='rgba(240,255,240,0.8)',
    bordercolor='green',
    borderwidth=1,
    borderpad=4
)

# Layout final
fig5.update_layout(
    xaxis_title='Temporada',
    yaxis_title='Eficiência (%)',
    template='plotly_white',
    yaxis_tickformat=".2f"
)

# Gráfico de chutes a gol
# Agrupar por temporada e somar os chutes a gol
chutes_a_gol_por_temporada = df_filtrado.groupby('Temporada').agg({
    'Chutes em Casa no Gol': 'sum',
    'Chutes Fora no Gol': 'sum'
}).reset_index()
# Transformar em formato longo para facilitar o gráfico
chutes_a_gol_melted = chutes_a_gol_por_temporada.melt(id_vars='Temporada', 
                                                         value_vars=['Chutes em Casa no Gol', 'Chutes Fora no Gol'],
                                                         var_name='Tipo de Chute',
                                                         value_name='Quantidade')
# Ordenar corretamente a temporada (por ano inicial)
chutes_a_gol_melted['AnoInicial'] = chutes_a_gol_melted['Temporada'].str[:4].astype(int)
chutes_a_gol_melted = chutes_a_gol_melted.sort_values('AnoInicial')
# Criar o gráfico
fig6 = px.line(
    chutes_a_gol_melted,
    x='Temporada',
    y='Quantidade',
    color='Tipo de Chute',
    markers=True,
    #title='Quantidade de Chutes a Gol por Temporada - Casa vs Visitante'
)
fig6.update_layout(
    xaxis_title='Temporada',
    yaxis_title='Total de Chutes a Gol',
    template= 'plotly_white',
)

#==================================================================================================================================#
#                                                  MACHINE LEARNING                                                                #
#==================================================================================================================================#

# Desempenho médio dos times jogando em casa
df_home = df_filtrado.groupby('Time da Casa').agg({
    'Gols Casa': 'mean',  # Gols feitos em casa
    'Chutes Casa': 'mean',    # Chutes em casa
    'Faltas em Casa': 'mean',     # Faltas cometidas em casa
    'Cartões Amarelos Casa': 'mean',    # Amarelos em casa
    'Cartões Vermelhos Casa': 'mean'     # Vermelhos em casa
}).rename(columns={
    'Gols Casa': 'Gols_Casa_Médio',
    'Chutes Casa': 'Chutes_Casa_Médio',
    'Faltas em Casa': 'Faltas_Casa_Médio',
    'Cartões Amarelos Casa': 'Amarelos_Casa_Médio',
    'Cartões Vermelhos Casa': 'Vermelhos_Casa_Médio'
})

# Desempenho médio dos times jogando fora
df_away = df_filtrado.groupby('Time Visitante').agg({
    'Gols Fora': 'mean',  # Gols feitos em casa
    'Chutes Fora': 'mean',    # Chutes em casa
    'Faltas Fora': 'mean',     # Faltas cometidas em casa
    'Cartões Amarelos Fora': 'mean',    # Amarelos em casa
    'Cartões Vermelhos Fora': 'mean'     # Vermelhos em casa
}).rename(columns={
    'Gols Fora': 'Gols_Fora_Médio',
    'Chutes Fora': 'Chutes_Fora_Médio',
    'Faltas Fora': 'Faltas_Fora_Médio',
    'Cartões Amarelos Fora': 'Amarelos_Fora_Médio',
    'Cartões Vermelhos Fora': 'Vermelhos_Fora_Médio'
})

# Base consolidada
df_merged = pd.merge(df_home, df_away, left_index=True, right_index=True, how='outer')
df_merged.reset_index(inplace=True)

# Treinamento do modelo K-Means
base_cluster = df_merged[
    [
        'Time da Casa',
        'Gols_Casa_Médio', 
        'Gols_Fora_Médio',
        'Chutes_Casa_Médio', 
        'Chutes_Fora_Médio',
    ]
]

# 2. Calcular médias por time (agregando casa e fora)
base_cluster['Time da Casa'] = base_cluster['Time da Casa'].astype(str)
colunas = ['Gols_Casa_Médio', 'Gols_Fora_Médio', 'Chutes_Casa_Médio', 'Chutes_Fora_Médio']
df_cluster = base_cluster.groupby('Time da Casa')[colunas].mean().reset_index()

# 3. Padronizar variáveis
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_cluster[colunas])

# 4. Método do cotovelo para definir k
inertias = []
K = range(1, 11)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)

# Gráfico 7: Cotovelo do K-Means
# Criando o gráfico de cotovelo com Plotly
fig7 = px.line(
    x=K,
    y=inertias,
    title='Método do Cotovelo',
    labels={'x': 'Número de Clusters (k)', 'y': 'Inércia'},
    markers=True
)
fig7.update_layout(
    xaxis_title='Número de Clusters (k)',
    yaxis_title='Inércia',
    template='plotly_white'
)

# Adicionando a linha do cotovelo
fig7.add_shape(
    type='line',
    x0=4,
    y0=min(inertias),
    x1=4,
    y1=max(inertias),
    line=dict(color='red', width=2, dash='dash'),
)

# 5. Aplicar K-Means com k ideal (ajuste com base no gráfico do cotovelo)
kmeans = KMeans(n_clusters=4, random_state=42)
df_cluster['Cluster'] = kmeans.fit_predict(X_scaled)

# 6. Redução de dimensionalidade para visualização
pca = PCA(n_components=2)
pca_result = pca.fit_transform(X_scaled)
df_cluster['PCA1'] = pca_result[:, 0]
df_cluster['PCA2'] = pca_result[:, 1]

# 8. Visualização com Plotly
fig8 = px.scatter(
    df_cluster,
    x='PCA1',
    y='PCA2',
    color='Cluster',
    #title='Times agrupados por desempenho (K-Means + PCA)',
    labels={'PCA1': 'Componente Principal 1', 'PCA2': 'Componente Principal 2'},
    hover_name='Time da Casa',
    color_continuous_scale=px.colors.qualitative.Set2
)
fig8.update_traces(marker=dict(size=10))
fig8.update_layout(
    xaxis_title='Componente Principal 1',
    yaxis_title='Componente Principal 2',
    template='plotly_white'
)

# Adicionando os rótulos dos clusters ao dataframe original
df_merged['Cluster'] = kmeans.labels_
df_merged.head(10)

# Calculando a média das variáveis por cluster
cluster_profiles = df_merged.groupby('Cluster').agg({
    'Gols_Casa_Médio': 'mean',
    'Gols_Fora_Médio': 'mean',
    'Chutes_Casa_Médio': 'mean',
    'Chutes_Fora_Médio': 'mean'
}).reset_index()

# Filtrando os times do grupo "Times Ofensivos de Alta Performance"
best_teams = df_merged[df_merged['Cluster'] == 1].copy()

#==================================================================================================================================#
#                                                   CONSTRUÇÃO DO DASHBOARD                                                        #
#==================================================================================================================================#

# Ajustando o leyout do Dashboard, de modo que os gráficos respeitem as colunas e ele fique um pouco mais espaçado

# Configuração da página
st.set_page_config(page_title="Estudo de Desempenho Ofensivo na Premier League", layout="wide")

#==================================================================================================================================#

# CSS para reduzir o espaço superior
st.markdown("""
    <style>
        .block-container {
            padding-top: 1rem;
        }
    </style>
""", unsafe_allow_html=True)

#==================================================================================================================================#
# Título principal
st.title("Estudo de Desempenho Ofensivo na Premier League") # Adicionando um título
st.subheader("Análise de Agrupamento com K-Means e PCA")
st.markdown("---")  # linha horizontal


#==================================================================================================================================#
# Estrutura básica com a barra lateral

# Barra lateral
st.sidebar.markdown("### Bem-vindo ao Dashboard") # Adicionar título + legenda na sidebar
st.sidebar.title("Opções") #Título do Filtro
opcao = st.sidebar.selectbox("Navegação:", ["🌐 Visão Geral", "📊 Análises", "🤖 Machine Learning", "📑 Documentação", "ℹ️ Sobre o Autor"])

#==================================================================================================================================#

if opcao == "🌐 Visão Geral": # Detalhando a página de exibição
    st.info("ℹ️ Este conjunto de dados é apenas para uso educacional e não comercial. Dados brutos obtidos do site football-data.co.uk.\n" 
        "\n 📥 Os dados foram extraídos diretamente do kaggle.\n"
        "\n 🔗 Link: https://www.kaggle.com/datasets/marcohuiii/english-premier-league-epl-match-data-2000-2025") # Adicionando uma caixa de informação
    
    st.subheader("Objetivo do Dashboard") # Adicionando um subtítulo
    st.write("⚽️ Analisar e comparar o desempenho dos times com base em estatísticas ofensivas, identificando padrões de performance e agrupando-os em perfis distintos utilizando técnicas de **aprendizado de máquina**\n"
    "\n com (**K-Means + PCA**).")

    st.subheader("Sobre o Conjunto de Dados") # Adicionando um subtítulo
    st.write('⚽️ A base utilizada possui dados de partidas da Premier League - Temporadas de 2009/10 até 2023/24') # Exibindo o número de linhas e colunas
    st.write('A base de dados possui 5700 linhas e 23 colunas') # Exibindo o número de linhas e colunas
    
    st.write('⚽️ Abaixo uma amostra da base de dados')
    with st.expander("Mostrar Tabela"):
    # Exibir a tabela dentro do expander
        st.dataframe(df_filtrado.head(5)) # Exibindo a tabela com as 5 primeiras linhas
    
    st.write('⚽️ Abaixo um dicionário com as variáveis da base de dados')
    with st.expander("Mostrar Dicionário"):
    # Exibir a tabela dentro do expander
        st.dataframe(dicionario)

elif opcao == "📊 Análises": # Detalhando a página de exibição
    tabs = st.tabs(["⚽ Gols por Temporada", "🏆 Vitórias por Temporada", "✅ % Vitórias Confirmadas", "📉 Chutes a Gol", "📈 Eficiência em Finalizações"])
    with tabs[0]:
        st.subheader("Quantidade de Gols por Temporada - Casa vs Visitante")
        st.write("A pandemia da COVID-19 provocou uma mudança significativa no comportamento dos jogos. Durante o período sem torcida, observou-se uma queda nos gols marcados pelos times da casa, possivelmente pela ausência do apoio da arquibancada. Por outro lado, os times visitantes passaram a marcar mais gols, aproveitando a neutralização do fator casa.")

        st.write("Após a retomada gradual dos públicos nos estádios, essa tendência se inverteu: os mandantes voltaram a demonstrar força ofensiva, com um crescimento acentuado na média de gols marcados, indicando a recuperação da vantagem de jogar em casa.")
        st.plotly_chart(fig1, use_container_width=True)

    with tabs[1]:
        st.subheader("Vitórias por Temporada - Casa vs Visitante")
        st.write("Com relação as vitórias o cenário é semelhante. **Os times visitantes venceram mais partidas do que os mandantes** — algo que não se repetiu em nenhum outro momento do período analisado.")
        st.write("A ausência da torcida, que normalmente exerce pressão sobre o adversário e apoio ao time da casa, parece ter equilibrado as forças, favorecendo os visitantes. Esse comportamento reforça a importância do fator casa no futebol e como ele foi drasticamente afetado pelas restrições da pandemia.")
        st.plotly_chart(fig2, use_container_width=True)

    with tabs[2]:
        with st.container():
            st.subheader("**Desempenho dos times mandantes ao irem para o intervalo vencendo**")
            st.write("Quando os times jogam em casa e terminam o primeiro tempo em vantagem, eles confirmam a vitória em mais de 70% das partidas.\n"
                     "\n Considerando a média geral, esse índice chega a 80,6%, reforçando a importância do mando de campo e a tendência de manutenção da liderança no segundo tempo.")


            col1, col2, col3 = st.columns([0.6, 0.05, 0.3])  # Retângulo maior e quadrado menor
            with col1:
                st.plotly_chart(fig3, use_container_width=True)
            with col2:
                st.write("")
            with col3:
                st.plotly_chart(fig4, use_container_width=True)

    with tabs[3]:
        st.subheader("Quantidade de Chutes a Gol por Temporada - Casa vs Visitante")
        st.write("A quantidade de chutes a gol reduziu significativamente na temporada 2013/14, mas a quantidade de gols permaneceu estável.\n"
                 "\n Isso sugere que os times passaram a ser mais eficientes em suas finalizações, como apresentado na tela a seguir.")
        st.plotly_chart(fig6, use_container_width=True)

    with tabs[4]:
        st.subheader("Eficiência em Finalizações por Temporada (Gols / Chutes Totais)")
        st.write("A partir da temporada 2013/14, os times passaram a demonstrar maior eficiência ofensiva, mesmo com um número menor de finalizações.\n"
                 "\n Isso indica uma evolução tática e técnica, onde a precisão passou a se sobressair à quantidade.\n"
                 "\n Os clubes parecem estar priorizando jogadas com maior chance real de gol, o que reflete uma modernização no estilo de jogo.")
        st.plotly_chart(fig5, use_container_width=True)

elif opcao == "🤖 Machine Learning": # Detalhando a página de exibição
    tabs_ml = st.tabs(["📄 Base do Modelo", "🔎 Definindo o Número de Clusters" ,"📊 Visualização dos Agrupamentos", "📈 Análise dos Agrupamentos", "🏆 Melhores Times"])
    
    with tabs_ml[0]:
        st.subheader("Agrupamento dos times por desempenho ofensivo - K-Means")
        st.write("O agrupamento dos times foi realizado com base em suas estatísticas ofensivas, utilizando o algoritmo K-Means. O objetivo é identificar padrões de desempenho e classificar os times em grupos distintos, permitindo uma análise mais aprofundada de suas características.")
        st.write("Abaixo estão as médias de desempenho dos times jogando em casa e fora, que foram utilizadas para o agrupamento.")
        with st.expander("Amostra dos Dados", expanded=True):
            st.dataframe(df_merged.head(10)) # Exibindo a tabela com as 10 primeiras linhas
    
    with tabs_ml[1]:
        st.subheader("Definindo o Número de Clusters")
        st.write("O método do cotovelo é uma técnica utilizada para determinar o número ideal de clusters em um algoritmo de agrupamento, como o K-Means. Ele envolve a execução do algoritmo para diferentes valores de k (número de clusters) e a análise da inércia (soma das distâncias quadradas entre os pontos e seus respectivos centróides).")
        st.write("No método do cotovelo, buscamos o ponto onde há uma diminuição acentuada na inércia, e após esse ponto as reduções são marginais – ou seja, o custo (inércia) continua caindo, mas com ganhos menores.")
        with st.expander("Mostrar Gráfico", expanded=False):
            st.plotly_chart(fig7, use_container_width=True)
        st.write("No Gráfico:\n"
                 "\n 🗸 A inércia despenca de k=1 para k=2, e cai forte até k=4.\n"
                 "\n 🗸 A partir de k=4 em diante, a redução na inércia é muito menor (curva suaviza).")
        st.write("Isso indica que 4 clusters é o ponto ótimo em que se ganha um bom nível de segmentação sem overfitting.")
    
    with tabs_ml[2]:
        st.subheader("Visualização dos Agrupamentos")
        st.write("Abaixo está a visualização dos agrupamentos dos times, utilizando PCA (Análise de Componentes Principais) para reduzir a dimensionalidade dos dados e facilitar a interpretação.")
        st.write("O gráfico mostra que os 4 clusters estão bem distribuídos no espaço projetado pelo PCA, com separações razoáveis entre alguns grupos — especialmente o Cluster 1 (Roxo), que se destaca mais à direita, sugerindo um grupo de times com características bem distintas.")
        st.plotly_chart(fig8, use_container_width=True)
    
    with tabs_ml[3]:
        st.subheader("Análise dos Agrupamentos")
        st.write("Abaixo estão as médias de desempenho dos times em cada cluster, permitindo uma análise mais detalhada das características de cada grupo.")
        with st.expander("Mostrar Tabela", expanded=True):
            st.dataframe(cluster_profiles)
        st.write("* 🟣 Cluster 1: Times que mantêm um alto volume ofensivo e marcam muitos gols, tanto em casa quanto fora.\n"
                 "\n * 🟡 Cluster 2: Times equilibrados, com ataque relativamente forte. Provavelmente ocupam posições intermediárias-alta na tabela.\n"
                 "\n * 🟢 Cluster 0: Times que até criam chances, mas são ineficientes nas finalizações. Provavelmente lutando no meio da tabela.\n"
                 "\n * 🟤 Cluster 3: Times com baixo volume ofensivo e baixo aproveitamento. Provavelmente na parte de baixo da tabela ou lutando contra o rebaixamento.")

    with tabs_ml[4]:
        st.subheader("Melhores Times - Times Ofensivos de Alta Performance")
        st.write("O grupo de times classificados como **Ofensivos de Alta Performance** (Cluster 1) se destaca por sua capacidade de marcar gols e criar oportunidades, tanto em casa quanto fora. Esses times possuem um desempenho ofensivo superior, com alta média de gols e chutes a gol.")
        st.write("Abaixo estão os times que se destacam nesse grupo:")
        with st.expander("Mostrar Tabela", expanded=True):
            st.dataframe(best_teams[['Time da Casa', 'Gols_Casa_Médio', 'Gols_Fora_Médio', 'Chutes_Casa_Médio', 'Chutes_Fora_Médio', 'Cluster']])
    
        

#==================================================================================================================================#
#                                                  DOCUMENTAÇÃO DO PROJETO                                                         #
#==================================================================================================================================#

elif opcao == "📑 Documentação":
    
    st.subheader("1️⃣ Bibliotecas Utilizadas:")
    st.write("1. **Pandas**: Para manipulação e análise de dados.")
    st.write("2. **Scikit-learn**: Para implementação de algoritmos de aprendizado de máquina.")
    st.write("3. **Plotly**: Para visualização interativa dos dados.")
    st.write("4. **Streamlit**: Para criação do dashboard interativo.")

    st.code("""
    import streamlit as st
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA""", language='python')

    st.subheader("2️⃣ Carregamento dos Dados:")
    st.write("* Os dados foram coletados do Kaggle e carregados em um DataFrame.\n"
             "\n * Link: https://www.kaggle.com/datasets/marcohuiii/english-premier-league-epl-match-data-2000-2025")
    st.write("**📌 Carregando a base de dados**")
    st.code("""
            # Atenção: O caminho do arquivo deve ser ajustado conforme o local onde o arquivo CSV está salvo
            epl_dados = pd.read_csv("epl_final.csv", sep=",", encoding="utf-8")
            epl_dados.head(5)""", 
            language='python')
    
    epl_dados = pd.read_csv("epl_final.csv", sep=",", encoding="utf-8")
    with st.expander("Resultado", expanded=True):
        st.dataframe(epl_dados.head(5))
    
    st.subheader("3️⃣ Limpeza dos Dados")
    st.write("* Para facilitar o trabalho, as colunas foram renomeadas para português (Brasil).\n"
             "\n * Os dados foram filtrados para incluir apenas as temporadas relevantes (2009-2023).")
    
    # Código Suporte =====================================================================================
    epl_dados.columns = colunas_pt
    # 1. Criar uma nova coluna com o ano inicial da temporada como número inteiro
    epl_dados['AnoInicial'] = epl_dados['Temporada'].str[:4].astype(int)

    # 2. Filtrar as temporadas entre 2009 e 2020
    epl_dados_filtrado = epl_dados[(epl['AnoInicial'] >= 2009) & (epl_dados['AnoInicial'] <= 2023)]
    # =====================================================================================================

    st.write("**📌 Criando uma lista de colunas em português**")
    st.code("""
            colunas_pt = [
            'Temporada', 'Data da Partida', 'Time da Casa', 'Time Visitante', 'Gols Casa',
            'Gols Fora', 'Resultado no Fim do Jogo', 'Gols em Casa no Intervalo',
            'Gols Fora no Intervalo', 'Resultado no Intervalo', 'Chutes Casa', 'Chutes Fora',
            'Chutes em Casa no Gol', 'Chutes Fora no Gol', 'Escanteios em Casa', 'Escanteios Fora',
            'Faltas em Casa', 'Faltas Fora', 'Cartões Amarelos Casa', 'Cartões Amarelos Fora',
            'Cartões Vermelhos Casa', 'Cartões Vermelhos Fora'
            ])""",
            language='python')
    
    st.write("**📌 Alterar os nomes das colunas**")
    st.code("""
            epl_dados.columns = colunas_pt
            )
            """,
            language='python')

    st.write("**📌 Filtrar a base de dados**")
    st.code("""
            # Criar uma nova coluna com o ano inicial da temporada como número inteiro
            epl_dados['AnoInicial'] = epl_dados['Temporada'].str[:4].astype(int)

            # Filtrar as temporadas entre 2009 e 2020
            epl_dados_filtrado = epl_dados[(epl['AnoInicial'] >= 2009) & (epl_dados['AnoInicial'] <= 2023)]
            
            # Visualizar uma amostra dos dados
            epl_dados_filtrado.head(5))""",
            language='python')
    with st.expander("Resultado", expanded=True):
        st.dataframe(epl_dados_filtrado.head(5))

    st.subheader("4️⃣ Análise Exploratória")
    st.write("* Verificar os tipos das colunas. \n"
             "\n * Verificar a quantidade de linhas e colunas. \n"
             "\n * Verificar se existem dados nulos. \n"
             "\n * Gráfico 1: Quantidade de Gols por Temporada - Casa vs Visitante. \n"
             "\n * Gráfico 2: Quantidade de Vitórias por Temporada - Casa vs Visitante. \n"
             "\n * Fórmula #1: % Confirmação Vitória. \n"
             "\n * Gráfico 3: Porcentagem de Vitórias Confirmadas - Casa vencendo no Intervalo. \n"
             "\n * Calcular a média geral. \n"
             "\n * Gráfico 4: Velocímetro com a média geral. \n"
             "\n * Gráfico 5: Quantidade de Chutes a Gol por Temporada - Casa vs Visitante. \n"
             "\n * Fórmula #2: Eficiência de Finalizações."
             "\n * Gráfico 6: Eficiência de Finalizações por Temporada (Gols / Chutes Totais).")
    
    # Código Suporte =====================================================================================
    # Obter os tipos das colunas
    tipos_colunas = epl_dados_filtrado.dtypes

    # Converter para DataFrame
    tipos_colunas_df = tipos_colunas.to_frame()

    # Renomear a coluna que por padrão é chamada de 0
    tipos_colunas_df.columns = ['Tipo de Dado']

    # Resetar o índice para transformar os nomes das colunas em uma coluna normal
    tipos_colunas_df = tipos_colunas_df.reset_index()

    # Renomear as colunas
    tipos_colunas_df.columns = ['Coluna', 'Tipo de Dado']
    
    # Verificando o Shape do DataFrame
    shape = f'✔️ A base de dados possui {epl_dados_filtrado.shape[0]} linhas e {epl_dados_filtrado.shape[1]} colunas'

    # Obter a contagem de valores nulos por coluna
    valores_nulos = epl_dados_filtrado.isnull().sum()

    # Converter para DataFrame
    valores_nulos_df = valores_nulos.to_frame()

    # Renomear a coluna (por padrão ela se chama 0)
    valores_nulos_df.columns = ['Valores Nulos']

    # (Opcional) Resetar o índice para transformar os nomes das colunas em uma coluna normal
    valores_nulos_df = valores_nulos_df.reset_index()

    # Renomear as colunas: a primeira coluna é o nome original da coluna do DataFrame original
    valores_nulos_df.columns = ['Coluna', 'Valores Nulos']
    # ==================================================================================================
    
    st.write("**📌 Verificando os tipos de dados**")
    st.code("""
            epl_dados_filtrado.dtypes
            """)
    with st.expander("Resultado", expanded=True):
        st.dataframe(tipos_colunas_df)

    st.write("**📌 Verificando o Shape do DataFrame**")
    st.code("""
            f'✔️ A base de dados possui {epl_dados_filtrado.shape[0]} linhas e {epl_dados_filtrado.shape[1]} colunas'""")
    with st.expander("Resultado", expanded=True):
        st.write(shape)

    st.write("**📌 Verificando a quantidade de valores nulos**")
    st.code("""
            epl_dados_filtrado.isnull().sum()""",
            language='python')
    with st.expander("Resultado", expanded=True):
        st.dataframe(valores_nulos_df)
    
    st.write("**📊 Gráfico 1: Quantidade de Gols por Temporada - Casa vs Visitante**")
    st.code("""
            # Agrupar por temporada e somar os gols
            gols_por_temporada = epl_dados_filtrado.groupby('Temporada').agg({
                'Gols Casa': 'sum',
                'Gols Fora': 'sum'
            }).reset_index()

            # Transformar em formato longo para facilitar o gráfico
            gols_melted = gols_por_temporada.melt(id_vars='Temporada', 
                                                value_vars=['Gols Casa', 'Gols Fora'],
                                                var_name='Tipo de Gol',
                                                value_name='Quantidade de Gols')

            # Ordenar corretamente a temporada (por ano inicial)
            gols_melted['AnoInicial'] = gols_melted['Temporada'].str[:4].astype(int)
            gols_melted = gols_melted.sort_values('AnoInicial')

            # Criar o gráfico
            fig1 = px.line(
                gols_melted,
                x='Temporada',
                y='Quantidade de Gols',
                color='Tipo de Gol',
                markers=True,
                title='Quantidade de Gols por Temporada - Casa vs Visitante'
            )

            fig1.update_layout(
                xaxis_title='Temporada',
                yaxis_title='Total de Gols',
                template= 'plotly_white',
            )

            fig1.add_annotation(
                x='2020/21',
                y=gols_melted[(gols_melted['Temporada'] == '2020/21') & (gols_melted['Tipo de Gol'] == 'Gols Casa')]['Quantidade de Gols'].values[0],
                text="⬇ Redução nos gols dos mandantes<br>⬆ Aumento nos gols visitantes<br><b>Pandemia: jogos sem torcida</b>",
                showarrow=True,
                arrowhead=1,
                ax=-100,
                ay=-100,
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="black",
                font=dict(size=12)
            )

            fig1.show()""", language='python')
    with st.expander("Resultado", expanded=True):
        st.plotly_chart(fig1, use_container_width=True)

    st.write("**📊 Gráfico 2: Quantidade de Vitórias por Temporada - Casa vs Visitante**")
    st.code("""
            # Contar vitórias por temporada
            vitorias_por_temporada = df_filtrado.groupby('Temporada')['Resultado no Fim do Jogo'].value_counts().unstack().fillna(0)

            # Selecionar apenas vitórias da casa e fora
            vitorias_por_temporada = vitorias_por_temporada[['H', 'A']].reset_index()
            vitorias_por_temporada = vitorias_por_temporada.rename(columns={
                'H': 'Vitórias Casa',
                'A': 'Vitórias Fora'
            })

            # Formato longo para plotar
            vitorias_melted = vitorias_por_temporada.melt(id_vars='Temporada',
                                                        value_vars=['Vitórias Casa', 'Vitórias Fora'],
                                                        var_name='Tipo de Vitória',
                                                        value_name='Quantidade de Vitórias')

            # Ordenar temporadas por ano
            vitorias_melted['AnoInicial'] = vitorias_melted['Temporada'].str[:4].astype(int)
            vitorias_melted = vitorias_melted.sort_values('AnoInicial')

            # Gráfico
            fig2 = px.line(
                vitorias_melted,
                x='Temporada',
                y='Quantidade de Vitórias',
                color='Tipo de Vitória',
                markers=True,
                title='Quantidade de Vitórias por Temporada - Casa vs Fora'
            )

            fig2.update_layout(
                xaxis_title='Temporada',
                yaxis_title='Total de Vitórias',
                template='plotly_white',
            )

            # Adicionar anotação para 2020/21 (efeito pandemia)
            fig2.add_annotation(
                x='2020/21',
                y=vitorias_melted[(vitorias_melted['Temporada'] == '2020/21') & (vitorias_melted['Tipo de Vitória'] == 'Vitórias Casa')]['Quantidade de Vitórias'].values[0],
                text="⬇ Menos vitórias da casa<br>⬆ Mais vitórias visitantes<br><b>Efeito pandemia?</b>",
                showarrow=True,
                arrowhead=1,
                ax=-100,
                ay=-100,
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="black",
                font=dict(size=12)
            )

            fig2.show()""", language='python')
    with st.expander("Resultado", expanded=True):
        st.plotly_chart(fig2, use_container_width=True)

    st.write("**🔢 Fórmula #1:** % Times da casa que foram para o intervalo vencendo e confirmaram a vitória.")
    with st.expander("Fórmula #1: % Vitórias Confirmadas", expanded=True):
        st.latex(r"\frac{\text{Vitórias\_Confirmadas}}{\text{Quantidade\_Total\_Jogos}} \times 100")
    
    st.write("**📊 Gráfico 3: Porcentagem de Vitórias Confirmadas - Casa vencendo no Intervalo**")
    st.code("""
            fig3 = px.line(
            confirmacao_vitoria,
            x='Temporada',
            y='% Confirmaram Vitória',
            markers=True,
            #title='Porcentagem de Vitórias Confirmadas - Casa vencendo no Intervalo',
        )

        fig3.update_layout(
            xaxis_title='Temporada',
            yaxis_title='% de Confirmação da Vitória',
            template='plotly_white',
            yaxis_tickformat=".1f"
        )

        # Alterando a escala do eixo y para minimo 0 e maximo 100
        fig3.update_yaxes(range=[0, 100])
        # Adicionando uma linha horizontal em 50%
        fig3.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="70% de Confirmação", annotation_position="bottom right")
        fig3.show()""", language='python')
    with st.expander("Resultado", expanded=True):
        st.plotly_chart(fig3, use_container_width=True)
    
    st.write("**📌 Calcular a média geral**")
    st.code("""
            total_vitorias_confirmadas = confirmacao_vitoria['Vitorias_Confirmadas'].sum()
            total_jogos = confirmacao_vitoria['Total_Jogos'].sum()

            media_geral = (total_vitorias_confirmadas / total_jogos) * 100)
            
            print(f"Média geral de confirmação de vitórias: {media_geral:.2f}%")""", language='python')
    with st.expander("Resultado", expanded=True):
        st.write(f"✔️ Média geral de confirmação de vitórias: {media_ponderada:.2f}%")

    st.write("**📊 Gráfico 4: Velocímetro com a média geral**")
    st.code("""
            fig4 = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=media_ponderada,
            title={'text': "Média Ponderada de Confirmação de Vitórias"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "royalblue"},
                'steps': [
                    {'range': [0, 50], 'color': "lightcoral"},
                    {'range': [50, 75], 'color': "gold"},
                    {'range': [75, 100], 'color': "lightgreen"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': media_ponderada
                }
            }
        ))

        fig4.update_layout(
            height=400,
            margin=dict(t=50, b=0, l=0, r=0)
        )

        fig4.show()""", language='python')
    
    with st.expander("Resultado", expanded=True):
        st.plotly_chart(fig4, use_container_width=True)
    
    st.write("**📊 Gráfico 5: Quantidade de Chutes a Gol por Temporada - Casa vs Visitante**")
    st.code("""
            # Criando um gráfico com o total de chutes a gol por temporada
            # Agrupar por temporada e somar os chutes a gol
            chutes_a_gol_por_temporada = df_filtrado.groupby('Temporada').agg({
                'Chutes em Casa no Gol': 'sum',
                'Chutes Fora no Gol': 'sum'
            }).reset_index()
            # Transformar em formato longo para facilitar o gráfico
            chutes_a_gol_melted = chutes_a_gol_por_temporada.melt(id_vars='Temporada', 
                                                                    value_vars=['Chutes em Casa no Gol', 'Chutes Fora no Gol'],
                                                                    var_name='Tipo de Chute',
                                                                    value_name='Quantidade')
            # Ordenar corretamente a temporada (por ano inicial)
            chutes_a_gol_melted['AnoInicial'] = chutes_a_gol_melted['Temporada'].str[:4].astype(int)
            chutes_a_gol_melted = chutes_a_gol_melted.sort_values('AnoInicial')
            # Criar o gráfico
            fig5 = px.line(
                chutes_a_gol_melted,
                x='Temporada',
                y='Quantidade',
                color='Tipo de Chute',
                markers=True,
                title='Quantidade de Chutes a Gol por Temporada - Casa vs Visitante'
            )
            fig5.update_layout(
                xaxis_title='Temporada',
                yaxis_title='Total de Chutes a Gol',
                template= 'plotly_white',
            ))

            fig5.show()""", language='python')
    with st.expander("Resultado", expanded=True):
        st.plotly_chart(fig6, use_container_width=True)

    st.write("**🔢 Fórmula #2:** % Quantidade de chutes que um time precisa para macar um gol.")
    with st.expander("Fórmula #2: % Eficiência Ofensiva", expanded=True):
        st.latex(r"\frac{\text{Gols\_Total}}{\text{Chutes\_Total}} \times 100")

    st.write("**📌 Cálculo de eficiência de finalizações**")
    st.code("""
            # Agrupando por temporada
            eficiencia_temporada = (
                df_filtrado
                .groupby('Temporada')
                .agg(
                    ChutesTotais=('Chutes em Casa no Gol', 'sum'),
                    ChutesVisitantes=('Chutes Fora no Gol', 'sum'),
                    GolsCasa=('Gols Casa', 'sum'),
                    GolsVisitante=('Gols Fora', 'sum')
                )
                .reset_index()
            )

            # Somando os valores para totalizar por temporada
            eficiencia_temporada['ChutesTotal'] = eficiencia_temporada['ChutesTotais'] + eficiencia_temporada['ChutesVisitantes']
            eficiencia_temporada['GolsTotal'] = eficiencia_temporada['GolsCasa'] + eficiencia_temporada['GolsVisitante']

            # Eficiência de finalização
            eficiencia_temporada['Eficiência (%)'] = (eficiencia_temporada['GolsTotal'] / eficiencia_temporada['ChutesTotal']) * 100

            # Calculando eficiências separadas
            eficiencia_temporada['Eficiência Casa (%)'] = (eficiencia_temporada['GolsCasa'] / eficiencia_temporada['ChutesTotais']) * 100
            eficiencia_temporada['Eficiência Visitantes (%)'] = (eficiencia_temporada['GolsVisitante'] / eficiencia_temporada['ChutesVisitantes']) * 100)""",language='python')

    st.write("**📊 Gráfico 6: Eficiência de Finalizações por Temporada (Gols / Chutes Totais)**")
    st.code("""
            fig6 = px.line(
            eficiencia_temporada,
            x='Temporada',
            y='Eficiência (%)',
            title='Eficiência de Finalização por Temporada (Gols / Chutes Totais)',
            markers=True
        )

        # Encontrar valores das temporadas 2012/13 e 2013/14
        x0 = '2012/13'
        x1 = '2013/14'

        y0 = eficiencia_temporada.loc[eficiencia_temporada['Temporada'] == x0, 'Eficiência (%)'].values[0]
        y1 = eficiencia_temporada.loc[eficiencia_temporada['Temporada'] == x1, 'Eficiência (%)'].values[0]

        # Adiciona seta
        fig6.add_annotation(
            x=x1,
            y=y1,
            axref='x',
            ayref='y',
            ax=x0,
            ay=y0,
            showarrow=True,
            arrowhead=3,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor='green'
        )

        # Adiciona anotação de texto
        fig6.add_annotation(
            x=x1,
            y=y1 + 0.9,  # levemente acima da seta
            text='Aumento significativo de eficiência (2013/14)',
            showarrow=False,
            font=dict(size=12, color='green'),
            bgcolor='rgba(240,255,240,0.8)',
            bordercolor='green',
            borderwidth=1,
            borderpad=4
        )

        # Layout final
        fig6.update_layout(
            xaxis_title='Temporada',
            yaxis_title='Eficiência (%)',
            template='plotly_white',
            yaxis_tickformat=".2f"
        )

        fig6.show()""", language='python')   
    with st.expander("Resultado", expanded=True):
        st.plotly_chart(fig5, use_container_width=True)

    st.subheader("5️⃣ Machine Leaning")
    st.write("* Base com a média de desempenho dos times jogando em casa. \n"
             "\n * Base com a média de desempenho dos times jogando fora de casa. \n"
             "\n * Consolidando as bases de dados (Média de desempenho em casa + Média de desempenho fora). \n"
             "\n * Realizando o agrupamento com K-Means e PCA. \n"
             "\n * Gráfico 7: Método Cotovelo. \n"
             "\n * Aplicando K-Means com k ideal. \n"
             "\n * Gráfico 8:Times agrupados por desempenho (K-Means + PCA)")

    st.write("**📌 Base com a média de desempenho dos times jogando em casa**")
    st.code("""
            df_home = df_filtrado.groupby('Time da Casa').agg({
            'Gols Casa': 'mean',  # Gols feitos em casa
            'Chutes Casa': 'mean',    # Chutes em casa
            'Faltas em Casa': 'mean',     # Faltas cometidas em casa
            'Cartões Amarelos Casa': 'mean',    # Amarelos em casa
            'Cartões Vermelhos Casa': 'mean'     # Vermelhos em casa
        }).rename(columns={
            'Gols Casa': 'Gols_Casa_Médio',
            'Chutes Casa': 'Chutes_Casa_Médio',
            'Faltas em Casa': 'Faltas_Casa_Médio',
            'Cartões Amarelos Casa': 'Amarelos_Casa_Médio',
            'Cartões Vermelhos Casa': 'Vermelhos_Casa_Médio'
        })
            
        df_home.head(5)""",language='python')
    with st.expander("Resultado", expanded=True):
        st.dataframe(df_home.head(5))

    st.write("**📌 Base com a média de desempenho dos times jogando fora de casa**")
    st.code("""
            df_away = df_filtrado.groupby('Time Visitante').agg({
            'Gols Fora': 'mean',  # Gols feitos em casa
            'Chutes Fora': 'mean',    # Chutes em casa
            'Faltas Fora': 'mean',     # Faltas cometidas em casa
            'Cartões Amarelos Fora': 'mean',    # Amarelos em casa
            'Cartões Vermelhos Fora': 'mean'     # Vermelhos em casa
        }).rename(columns={
            'Gols Fora': 'Gols_Fora_Médio',
            'Chutes Fora': 'Chutes_Fora_Médio',
            'Faltas Fora': 'Faltas_Fora_Médio',
            'Cartões Amarelos Fora': 'Amarelos_Fora_Médio',
            'Cartões Vermelhos Fora': 'Vermelhos_Fora_Médio'
        })
        
        df_away.head(5)""",language='python')
    with st.expander("Resultado", expanded=True):
        st.dataframe(df_away.head(5))
    
    st.write("**📌 Consolidando as bases de dados (Média de desempenho em casa + Média de desempenho fora)**")
    st.code("""
            df_merged = pd.merge(df_home, df_away, left_index=True, right_index=True, how='outer')
            df_merged.reset_index(inplace=True)
            df_merged.head(5)""", language='python')
    with st.expander("Resultado", expanded=True):
        st.dataframe(df_merged.head(5))
    
    st.write("**📌 Realizando o agrupamento com K-Means e PCA**")
    st.code("""
            base_cluster = df_merged[
                [
                    'Time da Casa',
                    'Gols_Casa_Médio', 
                    'Gols_Fora_Médio',
                    'Chutes_Casa_Médio', 
                    'Chutes_Fora_Médio',
                ]
            ]

            # 2. Calcular médias por time (agregando casa e fora)
            base_cluster['Time da Casa'] = base_cluster['Time da Casa'].astype(str)
            colunas = ['Gols_Casa_Médio', 'Gols_Fora_Médio', 'Chutes_Casa_Médio', 'Chutes_Fora_Médio']
            df_cluster = base_cluster.groupby('Time da Casa')[colunas].mean().reset_index()

            # 3. Padronizar variáveis
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(df_cluster[colunas])

            # 4. Método do cotovelo para definir k
            inertias = []
            K = range(1, 11)
            for k in K:
                kmeans = KMeans(n_clusters=k, random_state=42)
                kmeans.fit(X_scaled)
                inertias.append(kmeans.inertia_)""", language='python')
    
    st.write("**📊 Gráfico 7: Método Cotovelo**")
    st.code("""
            fig7 = px.line(
                x=K,
                y=inertias,
                title='Método do Cotovelo',
                labels={'x': 'Número de Clusters (k)', 'y': 'Inércia'},
                markers=True
            )
            fig7.update_layout(
                xaxis_title='Número de Clusters (k)',
                yaxis_title='Inércia',
                template='plotly_white'
            )

            # Adicionando a linha do cotovelo
            fig7.add_shape(
                type='line',
                x0=4,
                y0=min(inertias),
                x1=4,
                y1=max(inertias),
                line=dict(color='red', width=2, dash='dash'),
            )


            fig7.show()""", language='python')
    with st.expander("Resultado", expanded=True):
        st.plotly_chart(fig7, use_container_width=True)
    
    st.write("**📌 Aplicando K-Means com k ideal**")
    st.code("""
            kmeans = KMeans(n_clusters=4, random_state=42)
            df_cluster['Cluster'] = kmeans.fit_predict(X_scaled)

            # 6. Redução de dimensionalidade para visualização
            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(X_scaled)
            df_cluster['PCA1'] = pca_result[:, 0]
            df_cluster['PCA2'] = pca_result[:, 1]""", language='python')


    st.write("**📊 Gráfico 8: Times agrupados por desempenho (K-Means + PCA)**")
    st.code("""
            fig8 = px.scatter(
                df_cluster,
                x='PCA1',
                y='PCA2',
                color='Cluster',
                title='Times agrupados por desempenho (K-Means + PCA)',
                labels={'PCA1': 'Componente Principal 1', 'PCA2': 'Componente Principal 2'},
                hover_name='Time da Casa',
                color_continuous_scale=px.colors.qualitative.Set2
            )
            fig8.update_traces(marker=dict(size=10))
            fig8.update_layout(
                xaxis_title='Componente Principal 1',
                yaxis_title='Componente Principal 2',
                template='plotly_white'
            )

            fig8.show()""", language='python')
    with st.expander("Resultado", expanded=True):
        st.plotly_chart(fig8, use_container_width=True)
    
    st.subheader("**FIM**")

elif opcao == "ℹ️ Sobre o Autor":
    st.subheader("**Raphael Balmant**")

    st.write("Profissional com experiência em automação de processos, análise de dados e visualização, especializado no uso de ferramentas como Excel, Power BI, SQL, Python e R.\n"
             "\n Atuação na construção de dashboards interativos e relatórios estratégicos, transformando dados complexos em insights acionáveis para apoiar a tomada de decisões. Experiência na elaboração de apresentações executivas para diretoria e stakeholders, garantindo comunicação clara e objetiva dos dados.\n"
             "\n Habilidades em automação de processos utilizando VBA e Power Automate, otimizando fluxos operacionais e melhorando a eficiência das equipes. Forte conhecimento em controle de pagamentos, análise de indicadores e suporte comercial, aplicando conceitos de Data-Driven Decision Making para aprimorar processos e estratégias de negócio.")

    st.write("🧑‍🎓 Competências:\n"
             "\n * Análise e visualização de dados (Power BI, Excel)\n"
             "\n * Automação de processos (VBA, Power Automate)\n"
             "\n* Banco de dados e manipulação de dados (SQL, Python, R)\n"
             "\n * Elaboração de relatórios e apresentações executivas\n"
             "\n * Controle e otimização de fluxos operacionais")
    
    st.write("🚀 Interesse em oportunidades para aplicar e expandir conhecimentos em análise de dados, BI e automação de processos.")
    st.write("📩 Aberto a conexões e oportunidades na área de dados.")

    st.write("🔗 Linkedin: https://www.linkedin.com/in/raphael-henrique-balmant/")








    
    
    


    
    

    
    
