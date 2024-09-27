import math
import matplotlib.pyplot as plt
import plotly.io as pio
import plotly.express as px
# from plotly.offline import plot
pio.renderers.default='browser'
# from scipy.interpolate import griddata
#from CUSTOS import custos
import time 
from ipywidgets import interact, fixed
import ipywidgets as widgets
from IPython.display import display  # Import display function
import glob
import threading
import shutil
import argparse
import sys

import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.interpolate import griddata
from scipy.interpolate import interp1d
# from dash import Dash, dcc, html
# from dash.dependencies import Input, Output, State

# def criar_app_dash():
#     # Inicialização do aplicativo Dash
#     app = Dash(__name__)

#     # Caminho para a pasta 'Resultados/'
#     pasta_resultados = 'Resultados/'

#     # Verifica se a pasta 'Resultados/' existe
#     if not os.path.exists(pasta_resultados):
#         raise FileNotFoundError(f"A pasta '{pasta_resultados}' não foi encontrada.")

#     # Obtém lista de subpastas dentro de 'Resultados/'
#     subfolders = [f.name for f in os.scandir(pasta_resultados) if f.is_dir()]

#     # Inicializa conjuntos para armazenar valores únicos dos parâmetros
#     tamanho_talhao = set()
#     infestacao = set()
#     tamanho_drone = set()
#     indice_teste = set()
#     tipo_rota = set()

#     # Padrão esperado: field_16ha_100ha_14%_6m_3_TSP
#     for folder in subfolders:
#         parts = folder.split('_')
#         if len(parts) >= 7:
#             tamanho_talhao.add(parts[1])    # Exemplo: '16ha'
#             infestacao.add(parts[3])         # Exemplo: '14%'
#             tamanho_drone.add(parts[4])      # Exemplo: '6m'
#             indice_teste.add(parts[5])       # Exemplo: '3'
#             tipo_rota.add(parts[6])          # Exemplo: 'TSP' ou 'LM'
#         else:
#             print(f"Aviso: Nome de pasta não corresponde ao padrão esperado: {folder}")

#     # Ordena as opções para os dropdowns
#     tamanho_talhao = sorted(tamanho_talhao)
#     infestacao = sorted(infestacao)
#     tamanho_drone = sorted(tamanho_drone)
#     indice_teste = sorted(indice_teste, key=lambda x: int(x))  # Ordena numericamente
#     tipo_rota = sorted(tipo_rota)

#     # Layout do aplicativo
#     app.layout = html.Div([
#         html.H1('Gráfico 2D/3D Interativo', style={'textAlign': 'center', 'padding': '20px'}),

#         # Dropdown para escolher entre 2D e 3D
#         html.Div([
#             html.Label('Escolha o Tipo de Gráfico'),
#             dcc.Dropdown(
#                 id='tipo-grafico',
#                 options=[
#                     {'label': 'Gráfico 3D', 'value': '3D'},
#                     {'label': 'Gráfico 2D', 'value': '2D'}
#                 ],
#                 value='3D',
#                 clearable=False
#             ),
#         ], style={'width': '20%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '20px'}),

#         html.Div([
#             # Dropdowns para seleção dos parâmetros do caso
#             html.Div([
#                 html.Label('Tamanho talhão'),
#                 dcc.Dropdown(
#                     id='tamanho-talhao',
#                     options=[{'label': t, 'value': t} for t in tamanho_talhao],
#                     value=tamanho_talhao[0] if tamanho_talhao else None
#                 ),
#                 html.Label('Infestação', style={'marginTop': '20px'}),
#                 dcc.Dropdown(
#                     id='infestacao',
#                     options=[{'label': i, 'value': i} for i in infestacao],
#                     value=infestacao[0] if infestacao else None
#                 ),
#                 html.Label('Tamanho do drone', style={'marginTop': '20px'}),
#                 dcc.Dropdown(
#                     id='tamanho-drone',
#                     options=[{'label': td, 'value': td} for td in tamanho_drone],
#                     value=tamanho_drone[0] if tamanho_drone else None
#                 ),
#                 html.Label('Índice do Teste', style={'marginTop': '20px'}),
#                 dcc.Dropdown(
#                     id='indice-teste',
#                     options=[{'label': it, 'value': it} for it in indice_teste],
#                     value=indice_teste[0] if indice_teste else None
#                 ),
#                 html.Label('Tipo de rota', style={'marginTop': '20px'}),
#                 dcc.Dropdown(
#                     id='tipo-rota',
#                     options=[{'label': tr, 'value': tr} for tr in tipo_rota],
#                     value=tipo_rota[0] if tipo_rota else None
#                 ),
#                 html.Button('Simula', id='botao-simula', n_clicks=0, style={'marginTop': '30px', 'padding': '10px 20px'})
#             ], style={
#                 'width': '20%',
#                 'display': 'inline-block',
#                 'verticalAlign': 'top',
#                 'padding': '20px'
#             }),

#             # Dropdowns para seleção dos eixos (inclui controle de visibilidade do eixo Z)
#             html.Div([
#                 html.Label('Eixo X'),
#                 dcc.Dropdown(
#                     id='eixo-x',
#                     options=[],  # Será preenchido via callback
#                     value=None
#                 ),
#                 html.Label('Eixo Y', style={'marginTop': '20px'}),
#                 dcc.Dropdown(
#                     id='eixo-y',
#                     options=[],  # Será preenchido via callback
#                     value=None
#                 ),
#                 # Eixo Z só aparece se o gráfico for 3D
#                 html.Div(id='div-eixo-z', children=[
#                     html.Label('Eixo Z', style={'marginTop': '20px'}),
#                     dcc.Dropdown(
#                         id='eixo-z',
#                         options=[],  # Será preenchido via callback
#                         value=None
#                     ),
#                 ], style={'display': 'block'}),  # Inicialmente visível
#                 html.Label('Eixo Slider', style={'marginTop': '20px'}),
#                 dcc.Dropdown(
#                     id='eixo-slider',
#                     options=[],  # Será preenchido via callback
#                     value=None
#                 ),
#             ], style={
#                 'width': '25%',
#                 'display': 'inline-block',
#                 'verticalAlign': 'top',
#                 'padding': '20px'
#             }),

#             # Gráfico (2D ou 3D)
#             html.Div([
#                 dcc.Graph(id='grafico', style={'height': '80vh'})
#             ], style={
#                 'width': '50%',
#                 'display': 'inline-block',
#                 'padding': '20px'
#             }),
#         ], style={'display': 'flex', 'justifyContent': 'space-between'}),

#         # Áreas para mensagens de erro ou informações
#         html.Div(id='message-load', style={
#             'textAlign': 'center',
#             'color': 'red',
#             'padding': '10px',
#             'fontSize': '16px'
#         }),
#         html.Div(id='message-axes', style={
#             'textAlign': 'center',
#             'color': 'red',
#             'padding': '10px',
#             'fontSize': '16px'
#         }),
#         html.Div(id='message-graph', style={
#             'textAlign': 'center',
#             'color': 'red',
#             'padding': '10px',
#             'fontSize': '16px'
#         }),
#         # Componente para armazenar os dados carregados
#         dcc.Store(id='dados-carregados')
#     ], style={'height': '100vh', 'margin': '0', 'padding': '0'})

#     # Callback 1: Carregar Dados ao Clicar em "Simula"
#     @app.callback(
#         [Output('dados-carregados', 'data'),
#          Output('message-load', 'children')],
#         [Input('botao-simula', 'n_clicks')],
#         [State('tamanho-talhao', 'value'),
#          State('infestacao', 'value'),
#          State('tamanho-drone', 'value'),
#          State('indice-teste', 'value'),
#          State('tipo-rota', 'value')]
#     )
#     def carregar_dados(n_clicks, tamanho_talhao_sel, infestacao_sel, tamanho_drone_sel, indice_teste_sel, tipo_rota_sel):
#         if n_clicks == 0:
#             # Ainda não clicou no botão
#             return None, ''

#         # Verifica se todos os parâmetros foram selecionados
#         if not all([tamanho_talhao_sel, infestacao_sel, tamanho_drone_sel, indice_teste_sel, tipo_rota_sel]):
#             return None, 'Por favor, selecione todos os parâmetros antes de simular.'

#         # Monta o nome da pasta com base nos parâmetros selecionados
#         selected_folder = None
#         for folder in subfolders:
#             parts = folder.split('_')
#             if len(parts) >= 7:
#                 if (parts[1] == tamanho_talhao_sel and
#                     parts[3] == infestacao_sel and
#                     parts[4] == tamanho_drone_sel and
#                     parts[5] == indice_teste_sel and
#                     parts[6] == tipo_rota_sel):
#                     selected_folder = folder
#                     break

#         if not selected_folder:
#             # Se não encontrar a pasta correspondente
#             return None, 'Nenhum dado encontrado para a seleção atual. Verifique as combinações escolhidas.'

#         # Caminho para o arquivo Excel selecionado
#         arquivo_excel = os.path.join(pasta_resultados, selected_folder, 'Planilha_Unificada.xlsx')

#         # Imprime o nome da pasta que está sendo usada
#         print(f"Carregando dados da pasta: {selected_folder}")

#         # Verifica se o arquivo Excel existe
#         if not os.path.exists(arquivo_excel):
#             return None, 'Arquivo Excel não encontrado na pasta selecionada.'

#         # Tenta ler o arquivo Excel
#         try:
#             df = pd.read_excel(arquivo_excel, sheet_name='resultados')
#         except Exception as e:
#             return None, f'Erro ao ler o arquivo Excel: {e}'

#         # Remove colunas não numéricas
#         df_numeric = df.select_dtypes(include=[float, int])

#         # Verifica se há colunas numéricas suficientes
#         if len(df_numeric.columns) < 4:
#             return None, 'Não há colunas numéricas suficientes para criar o gráfico.'

#         # Transforma o DataFrame em um dicionário para armazenar no dcc.Store
#         dados = df_numeric.to_dict('records')

#         return dados, ''

#     # Callback 2: Atualizar Opções dos Dropdowns de Eixos com Base nos Dados Carregados
#     @app.callback(
#         [Output('eixo-x', 'options'),
#          Output('eixo-y', 'options'),
#          Output('eixo-z', 'options'),
#          Output('eixo-slider', 'options'),
#          Output('eixo-x', 'value'),
#          Output('eixo-y', 'value'),
#          Output('eixo-z', 'value'),
#          Output('eixo-slider', 'value'),
#          Output('div-eixo-z', 'style')],
#         [Input('dados-carregados', 'data'),
#          Input('tipo-grafico', 'value')]
#     )
#     def atualizar_eixos(dados_carregados, tipo_grafico):
#         if dados_carregados is None:
#             # Não há dados carregados
#             return [], [], [], [], None, None, None, None, {'display': 'block'}

#         # Recria o DataFrame a partir dos dados armazenados
#         df_numeric = pd.DataFrame(dados_carregados)

#         # Obtém a lista de colunas numéricas
#         colunas_numericas = df_numeric.columns.tolist()

#         # Cria as opções para os eixos
#         options = [{'label': col, 'value': col} for col in colunas_numericas]

#         # Define valores padrão para os eixos (primeiras quatro colunas)
#         eixo_x_val = colunas_numericas[0] if len(colunas_numericas) >= 1 else None
#         eixo_y_val = colunas_numericas[1] if len(colunas_numericas) >= 2 else None
#         eixo_z_val = colunas_numericas[2] if len(colunas_numericas) >= 3 else None
#         eixo_slider_val = colunas_numericas[3] if len(colunas_numericas) >= 4 else None

#         # Se o tipo de gráfico for 2D, esconde o eixo Z
#         if tipo_grafico == '2D':
#             return options, options, [], options, eixo_x_val, eixo_y_val, None, eixo_slider_val, {'display': 'none'}
#         else:
#             return options, options, options, options, eixo_x_val, eixo_y_val, eixo_z_val, eixo_slider_val, {'display': 'block'}

#     # Callback 3: Gerar o Gráfico (2D ou 3D)
#     @app.callback(
#         [Output('grafico', 'figure'),
#          Output('message-graph', 'children')],
#         [Input('eixo-x', 'value'),
#          Input('eixo-y', 'value'),
#          Input('eixo-z', 'value'),
#          Input('eixo-slider', 'value'),
#          Input('tipo-grafico', 'value')],
#         [State('dados-carregados', 'data')]
#     )
#     def gerar_grafico(eixo_x, eixo_y, eixo_z, eixo_slider, tipo_grafico, dados_carregados):
#         # Chama a função responsável por plotar o gráfico 2D ou 3D
#         if tipo_grafico == '3D':
#             return plot_3D(eixo_x, eixo_y, eixo_z, eixo_slider, dados_carregados)
#         else:
#             return plot_2D(eixo_x, eixo_y, eixo_slider, dados_carregados)

#     # Inicia o servidor Dash
#     app.run_server(debug=True)


# def plot_3D(eixo_x, eixo_y, eixo_z, eixo_slider, dados_carregados):
#     if dados_carregados is None:
#         # Não há dados carregados
#         fig = go.Figure()
#         fig.update_layout(
#             title='Nenhum dado carregado. Selecione os parâmetros e clique em "Simula".',
#             scene=dict(
#                 xaxis_title='',
#                 yaxis_title='',
#                 zaxis_title=''
#             ),
#             margin=dict(l=0, r=0, t=50, b=0)
#         )
#         return fig, ''

#     # Recria o DataFrame a partir dos dados armazenados
#     df_numeric = pd.DataFrame(dados_carregados)

#     # Verifica se os eixos selecionados existem nas colunas
#     for eixo in [eixo_x, eixo_y, eixo_z, eixo_slider]:
#         if eixo not in df_numeric.columns:
#             fig = go.Figure()
#             fig.update_layout(
#                 title=f'A coluna "{eixo}" não foi encontrada na planilha.',
#                 scene=dict(
#                     xaxis_title='',
#                     yaxis_title='',
#                     zaxis_title=''
#                 ),
#                 margin=dict(l=0, r=0, t=50, b=0)
#             )
#             return fig, f'A coluna "{eixo}" não foi encontrada na planilha.'

#     # Filtra os dados com base nos valores únicos do eixo_slider
#     valores_slider = sorted(df_numeric[eixo_slider].unique())
#     frames = []
#     mensagem = ''

#     for valor in valores_slider:
#         df_filtrado = df_numeric[df_numeric[eixo_slider] == valor]
#         # Verifica se há pelo menos 4 pontos para criar a superfície
#         if len(df_filtrado) >= 4:
#             # Cria uma grade para a interpolação
#             xi = np.linspace(df_filtrado[eixo_x].min(), df_filtrado[eixo_x].max(), 50)
#             yi = np.linspace(df_filtrado[eixo_y].min(), df_filtrado[eixo_y].max(), 50)
#             xi, yi = np.meshgrid(xi, yi)
#             # Interpola os dados
#             try:
#                 zi = griddata(
#                     (df_filtrado[eixo_x], df_filtrado[eixo_y]),
#                     df_filtrado[eixo_z],
#                     (xi, yi),
#                     method='linear'
#                 )
#                 # Trata valores NaN resultantes da interpolação
#                 zi = np.nan_to_num(zi, nan=np.nanmin(df_filtrado[eixo_z]))
#                 # Cria a superfície
#                 surface = go.Surface(
#                     x=xi,
#                     y=yi,
#                     z=zi,
#                     colorscale='Viridis',
#                     cmin=df_numeric[eixo_z].min(),
#                     cmax=df_numeric[eixo_z].max(),
#                     showscale=False,
#                     name=str(valor)
#                 )
#                 frames.append(go.Frame(data=[surface], name=str(valor)))
#             except Exception as e:
#                 mensagem += f"Aviso: Erro ao interpolar para o valor {valor}: {e}. Plotando apenas pontos.<br>"
#                 # Se a interpolação falhar, plota apenas pontos
#                 scatter = go.Scatter3d(
#                     x=df_filtrado[eixo_x],
#                     y=df_filtrado[eixo_y],
#                     z=df_filtrado[eixo_z],
#                     mode='markers',
#                     marker=dict(
#                         size=5,
#                         color=df_filtrado[eixo_z],
#                         colorscale='Viridis',
#                         opacity=0.8
#                     ),
#                     name=str(valor)
#                 )
#                 frames.append(go.Frame(data=[scatter], name=str(valor)))
#         else:
#             # Se não houver pontos suficientes, plota apenas os pontos
#             mensagem += f"Aviso: Não há pontos suficientes para interpolação em {eixo_slider} = {valor}. Plotando apenas pontos.<br>"
#             scatter = go.Scatter3d(
#                 x=df_filtrado[eixo_x],
#                 y=df_filtrado[eixo_y],
#                 z=df_filtrado[eixo_z],
#                 mode='markers',
#                 marker=dict(
#                     size=5,
#                     color=df_filtrado[eixo_z],
#                     colorscale='Viridis',
#                     opacity=0.8
#                 ),
#                 name=str(valor)
#             )
#             frames.append(go.Frame(data=[scatter], name=str(valor)))

#     # Verifica se há frames para evitar erros
#     if not frames:
#         fig = go.Figure()
#         fig.update_layout(
#             title='Nenhum dado disponível para exibir.',
#             scene=dict(
#                 xaxis_title=eixo_x,
#                 yaxis_title=eixo_y,
#                 zaxis_title=eixo_z
#             ),
#             margin=dict(l=0, r=0, t=50, b=0)
#         )
#         return fig, 'Nenhum dado disponível para exibir.'

#     # Cria o slider
#     slider_steps = []
#     for frame in frames:
#         slider_step = dict(
#             method='animate',
#             args=[[frame.name], dict(mode='immediate',
#                                      frame=dict(duration=500, redraw=True),
#                                      transition=dict(duration=0))],
#             label=str(frame.name)
#         )
#         slider_steps.append(slider_step)

#     sliders = [dict(
#         active=0,
#         currentvalue={'prefix': f'{eixo_slider}: '},
#         pad={'t': 50},
#         steps=slider_steps
#     )]

#     # Cria a figura
#     fig = go.Figure(
#         data=frames[0].data,
#         frames=frames,
#         layout=go.Layout(
#             scene=dict(
#                 xaxis_title=eixo_x,
#                 yaxis_title=eixo_y,
#                 zaxis_title=eixo_z
#             ),
#             sliders=sliders,
#             updatemenus=[dict(
#                 type='buttons',
#                 showactive=False,
#                 y=1.15,
#                 x=0.8,
#                 xanchor='left',
#                 yanchor='top',
#                 pad=dict(t=0, r=10),
#                 buttons=[dict(label='Play',
#                               method='animate',
#                               args=[None, dict(frame=dict(duration=500, redraw=True),
#                                                transition=dict(duration=0),
#                                                fromcurrent=True,
#                                                mode='immediate')])]
#             )],
#             margin=dict(l=0, r=0, t=0, b=0)
#         )
#     )

#     return fig, mensagem

# def plot_2D(eixo_x, eixo_y, eixo_slider, dados_carregados):
#     if dados_carregados is None:
#         # Não há dados carregados
#         fig = go.Figure()
#         fig.update_layout(
#             title='Nenhum dado carregado. Selecione os parâmetros e clique em "Simula".',
#             xaxis_title='',
#             yaxis_title='',
#             margin=dict(l=0, r=0, t=50, b=0)
#         )
#         return fig, ''

#     # Recria o DataFrame a partir dos dados armazenados
#     df_numeric = pd.DataFrame(dados_carregados)

#     # Verifica se os eixos selecionados existem nas colunas
#     for eixo in [eixo_x, eixo_y, eixo_slider]:
#         if eixo not in df_numeric.columns:
#             fig = go.Figure()
#             fig.update_layout(
#                 title=f'A coluna "{eixo}" não foi encontrada na planilha.',
#                 xaxis_title='',
#                 yaxis_title='',
#                 margin=dict(l=0, r=0, t=50, b=0)
#             )
#             return fig, f'A coluna "{eixo}" não foi encontrada na planilha.'

#     # Filtra os dados com base nos valores únicos do eixo_slider
#     valores_slider = sorted(df_numeric[eixo_slider].unique())
#     frames = []
#     mensagem = ''

#     for valor in valores_slider:
#         df_filtrado = df_numeric[df_numeric[eixo_slider] == valor]

#         # Realiza a interpolação cúbica spline para suavizar a curva
#         if len(df_filtrado) >= 4:
#             try:
#                 # Ordena os valores de x para garantir que estejam em ordem crescente
#                 df_filtrado = df_filtrado.sort_values(by=eixo_x)
                
#                 # Interpolação cúbica spline
#                 f_interp = interp1d(df_filtrado[eixo_x], df_filtrado[eixo_y], kind='cubic', fill_value="extrapolate")
#                 x_novo = np.linspace(df_filtrado[eixo_x].min(), df_filtrado[eixo_x].max(), 100)
#                 y_novo = f_interp(x_novo)

#                 # Cria a curva suavizada
#                 scatter = go.Scatter(
#                     x=x_novo,
#                     y=y_novo,
#                     mode='lines',
#                     line=dict(width=2),
#                     name=f'{eixo_slider}: {valor}'
#                 )
#             except Exception as e:
#                 mensagem += f"Aviso: Erro ao interpolar os dados para {valor}: {e}.<br>"
#                 scatter = go.Scatter(
#                     x=df_filtrado[eixo_x],
#                     y=df_filtrado[eixo_y],
#                     mode='markers',
#                     marker=dict(
#                         size=5,
#                         color=df_filtrado[eixo_y],
#                         colorscale='Viridis',
#                         opacity=0.8
#                     ),
#                     name=str(valor)
#                 )
#         else:
#             # Se não houver pontos suficientes, plota os pontos diretamente
#             scatter = go.Scatter(
#                 x=df_filtrado[eixo_x],
#                 y=df_filtrado[eixo_y],
#                 mode='markers+lines',
#                 marker=dict(
#                     size=5,
#                     color=df_filtrado[eixo_y],
#                     colorscale='Viridis',
#                     opacity=0.8
#                 ),
#                 name=str(valor)
#             )

#         frames.append(go.Frame(data=[scatter], name=str(valor)))

#     # Verifica se há frames para evitar erros
#     if not frames:
#         fig = go.Figure()
#         fig.update_layout(
#             title='Nenhum dado disponível para exibir.',
#             xaxis_title=eixo_x,
#             yaxis_title=eixo_y,
#             margin=dict(l=0, r=0, t=50, b=0)
#         )
#         return fig, 'Nenhum dado disponível para exibir.'

#     # Cria o slider
#     slider_steps = []
#     for frame in frames:
#         slider_step = dict(
#             method='animate',
#             args=[[frame.name], dict(mode='immediate',
#                                      frame=dict(duration=500, redraw=True),
#                                      transition=dict(duration=0))],
#             label=str(frame.name)
#         )
#         slider_steps.append(slider_step)

#     sliders = [dict(
#         active=0,
#         currentvalue={'prefix': f'{eixo_slider}: '},
#         pad={'t': 50},
#         steps=slider_steps
#     )]

#     # Cria a figura
#     fig = go.Figure(
#         data=frames[0].data,
#         frames=frames,
#         layout=go.Layout(
#             xaxis_title=eixo_x,
#             yaxis_title=eixo_y,
#             sliders=sliders,
#             updatemenus=[dict(
#                 type='buttons',
#                 showactive=False,
#                 y=1.15,
#                 x=0.8,
#                 xanchor='left',
#                 yanchor='top',
#                 pad=dict(t=0, r=10),
#                 buttons=[dict(label='Play',
#                               method='animate',
#                               args=[None, dict(frame=dict(duration=500, redraw=True),
#                                                transition=dict(duration=0),
#                                                fromcurrent=True,
#                                                mode='immediate')])]
#             )],
#             margin=dict(l=0, r=0, t=0, b=0)
#         )
#     )

#     return fig, mensagem


# def plot_3D(eixo_x, eixo_y, eixo_z, eixo_slider, dados_carregados):
#     if dados_carregados is None:
#         # Não há dados carregados
#         fig = go.Figure()
#         fig.update_layout(
#             title='Nenhum dado carregado. Selecione os parâmetros e clique em "Simula".',
#             scene=dict(
#                 xaxis_title='',
#                 yaxis_title='',
#                 zaxis_title=''
#             ),
#             margin=dict(l=0, r=0, t=50, b=0)
#         )
#         return fig, ''

#     # Recria o DataFrame a partir dos dados armazenados
#     df_numeric = pd.DataFrame(dados_carregados)

#     # Verifica se os eixos selecionados existem nas colunas
#     for eixo in [eixo_x, eixo_y, eixo_z, eixo_slider]:
#         if eixo not in df_numeric.columns:
#             fig = go.Figure()
#             fig.update_layout(
#                 title=f'A coluna "{eixo}" não foi encontrada na planilha.',
#                 scene=dict(
#                     xaxis_title='',
#                     yaxis_title='',
#                     zaxis_title=''
#                 ),
#                 margin=dict(l=0, r=0, t=50, b=0)
#             )
#             return fig, f'A coluna "{eixo}" não foi encontrada na planilha.'

#     # Filtra os dados com base nos valores únicos do eixo_slider
#     valores_slider = sorted(df_numeric[eixo_slider].unique())
#     frames = []
#     mensagem = ''

#     for valor in valores_slider:
#         df_filtrado = df_numeric[df_numeric[eixo_slider] == valor]
#         # Verifica se há pelo menos 4 pontos para criar a superfície
#         if len(df_filtrado) >= 4:
#             # Cria uma grade para a interpolação
#             xi = np.linspace(df_filtrado[eixo_x].min(), df_filtrado[eixo_x].max(), 50)
#             yi = np.linspace(df_filtrado[eixo_y].min(), df_filtrado[eixo_y].max(), 50)
#             xi, yi = np.meshgrid(xi, yi)
#             # Interpola os dados
#             try:
#                 zi = griddata(
#                     (df_filtrado[eixo_x], df_filtrado[eixo_y]),
#                     df_filtrado[eixo_z],
#                     (xi, yi),
#                     method='linear'
#                 )
#                 # Trata valores NaN resultantes da interpolação
#                 zi = np.nan_to_num(zi, nan=np.nanmin(df_filtrado[eixo_z]))
#                 # Cria a superfície
#                 surface = go.Surface(
#                     x=xi,
#                     y=yi,
#                     z=zi,
#                     colorscale='Viridis',
#                     cmin=df_numeric[eixo_z].min(),
#                     cmax=df_numeric[eixo_z].max(),
#                     showscale=False,
#                     name=str(valor)
#                 )
#                 frames.append(go.Frame(data=[surface], name=str(valor)))
#             except Exception as e:
#                 mensagem += f"Aviso: Erro ao interpolar para o valor {valor}: {e}. Plotando apenas pontos.<br>"
#                 # Se a interpolação falhar, plota apenas pontos
#                 scatter = go.Scatter3d(
#                     x=df_filtrado[eixo_x],
#                     y=df_filtrado[eixo_y],
#                     z=df_filtrado[eixo_z],
#                     mode='markers',
#                     marker=dict(
#                         size=5,
#                         color=df_filtrado[eixo_z],
#                         colorscale='Viridis',
#                         opacity=0.8
#                     ),
#                     name=str(valor)
#                 )
#                 frames.append(go.Frame(data=[scatter], name=str(valor)))
#         else:
#             # Se não houver pontos suficientes, plota apenas os pontos
#             mensagem += f"Aviso: Não há pontos suficientes para interpolação em {eixo_slider} = {valor}. Plotando apenas pontos.<br>"
#             scatter = go.Scatter3d(
#                 x=df_filtrado[eixo_x],
#                 y=df_filtrado[eixo_y],
#                 z=df_filtrado[eixo_z],
#                 mode='markers',
#                 marker=dict(
#                     size=5,
#                     color=df_filtrado[eixo_z],
#                     colorscale='Viridis',
#                     opacity=0.8
#                 ),
#                 name=str(valor)
#             )
#             frames.append(go.Frame(data=[scatter], name=str(valor)))

#     # Verifica se há frames para evitar erros
#     if not frames:
#         fig = go.Figure()
#         fig.update_layout(
#             title='Nenhum dado disponível para exibir.',
#             scene=dict(
#                 xaxis_title=eixo_x,
#                 yaxis_title=eixo_y,
#                 zaxis_title=eixo_z
#             ),
#             margin=dict(l=0, r=0, t=50, b=0)
#         )
#         return fig, 'Nenhum dado disponível para exibir.'

#     # Cria o slider
#     slider_steps = []
#     for frame in frames:
#         slider_step = dict(
#             method='animate',
#             args=[[frame.name], dict(mode='immediate',
#                                      frame=dict(duration=500, redraw=True),
#                                      transition=dict(duration=0))],
#             label=str(frame.name)
#         )
#         slider_steps.append(slider_step)

#     sliders = [dict(
#         active=0,
#         currentvalue={'prefix': f'{eixo_slider}: '},
#         pad={'t': 50},
#         steps=slider_steps
#     )]

#     # Cria a figura
#     fig = go.Figure(
#         data=frames[0].data,
#         frames=frames,
#         layout=go.Layout(
#             scene=dict(
#                 xaxis_title=eixo_x,
#                 yaxis_title=eixo_y,
#                 zaxis_title=eixo_z
#             ),
#             sliders=sliders,
#             updatemenus=[dict(
#                 type='buttons',
#                 showactive=False,
#                 y=1.15,
#                 x=0.8,
#                 xanchor='left',
#                 yanchor='top',
#                 pad=dict(t=0, r=10),
#                 buttons=[dict(label='Play',
#                               method='animate',
#                               args=[None, dict(frame=dict(duration=500, redraw=True),
#                                                transition=dict(duration=0),
#                                                fromcurrent=True,
#                                                mode='immediate')])]
#             )],
#             margin=dict(l=0, r=0, t=0, b=0)
#         )
#     )

#     return fig, mensagem

# def unificar_planilhas():
#     """
#     Unifica várias planilhas Excel em uma única, com opções interativas e gráficos 2D.
#     Também move os arquivos originais para uma subpasta 'Arquivos_brutos'.
#     """
#     # Caminho para a pasta 'Resultados'
#     pasta_resultados = 'Resultados/'
#     pasta_brutos = os.path.join(pasta_resultados, 'Arquivos_brutos')

#     # Cria a pasta 'Arquivos_brutos' se não existir
#     if not os.path.exists(pasta_brutos):
#         os.makedirs(pasta_brutos)

#     # Lista todos os arquivos xlsx na pasta 'Resultados' com o padrão de nomenclatura
#     arquivos_excel = glob.glob(os.path.join(pasta_resultados, 'RESULTADOS_*.xlsx'))

#     # Dicionário para armazenar DataFrames
#     planilhas = {}

#     # Lê cada arquivo e armazena no dicionário
#     for arquivo in arquivos_excel:
#         nome_arquivo = os.path.basename(arquivo)
#         nome_planilha = os.path.splitext(nome_arquivo)[0]
#         df = pd.read_excel(arquivo)
#         planilhas[nome_planilha] = df

#     # Caminho para a planilha unificada
#     caminho_planilha_unificada = os.path.join(pasta_resultados, 'Planilha_Unificada.xlsx')

#     # Cria um objeto ExcelWriter para escrever em um novo arquivo Excel
#     with pd.ExcelWriter(caminho_planilha_unificada, engine='xlsxwriter') as escritor:
#         # Escreve cada DataFrame em uma aba separada
#         for nome_planilha, df in planilhas.items():
#             df.to_excel(escritor, sheet_name=nome_planilha, index=False)

#         # Concatena todos os DataFrames em um só
#         df_resultados = pd.concat(planilhas.values(), ignore_index=True)
#         # Escreve o DataFrame combinado na aba 'resultados'
#         df_resultados.to_excel(escritor, sheet_name='resultados', index=False)

#         # Cria a aba 'gráficos'
#         workbook = escritor.book
#         worksheet = workbook.add_worksheet('gráficos')
#         escritor.sheets['gráficos'] = worksheet

#         # Lista de colunas disponíveis para plotagem
#         colunas = df_resultados.columns.tolist()

#         # Escreve opções para seleção
#         worksheet.write('A1', 'Selecione o eixo X:')
#         worksheet.write('A2', 'Selecione o eixo Y:')
#         worksheet.write('A3', 'Selecione o eixo Z:')

#         # Cria uma planilha oculta para armazenar as opções das colunas
#         hidden_sheet = workbook.add_worksheet('hidden')
#         hidden_sheet.hide()
#         for idx, coluna in enumerate(colunas):
#             hidden_sheet.write_string(idx, 0, coluna)  # Escreve na coluna A da planilha 'hidden'

#         # Define a validação de dados para os menus suspensos usando as colunas da planilha 'hidden'
#         faixa_colunas = f"'hidden'!$A$1:$A${len(colunas)}"
#         for row in range(1, 4):
#             worksheet.data_validation(f'B{row}', {
#                 'validate': 'list',
#                 'source': faixa_colunas
#             })

#         # Escreve as fórmulas para obter os dados das colunas selecionadas
#         data_start_row = 5  # Linha inicial para os dados
#         data_start_col = 0  # Coluna inicial para os dados

#         # Escreve os cabeçalhos
#         worksheet.write(data_start_row, data_start_col, 'Eixo X')
#         worksheet.write(data_start_row, data_start_col + 1, 'Eixo Y')
#         worksheet.write(data_start_row, data_start_col + 2, 'Eixo Z')

#         num_dados = len(df_resultados)

#         # Escreve as fórmulas para cada linha de dados
#         for i in range(num_dados):
#             row = data_start_row + 1 + i
#             # Fórmula para o Eixo X
#             formula_x = f"=INDEX(resultados!$A$2:$ZZ${num_dados + 1}, {i + 1}, MATCH($B$1, resultados!$A$1:$ZZ$1, 0))"
#             # Fórmula para o Eixo Y
#             formula_y = f"=INDEX(resultados!$A$2:$ZZ${num_dados + 1}, {i + 1}, MATCH($B$2, resultados!$A$1:$ZZ$1, 0))"
#             # Fórmula para o Eixo Z
#             formula_z = f"=INDEX(resultados!$A$2:$ZZ${num_dados + 1}, {i + 1}, MATCH($B$3, resultados!$A$1:$ZZ$1, 0))"

#             worksheet.write_formula(row, data_start_col, formula_x)
#             worksheet.write_formula(row, data_start_col + 1, formula_y)
#             worksheet.write_formula(row, data_start_col + 2, formula_z)

#         # Cria o gráfico 2D usando os dados calculados, representando o terceiro eixo pelo tamanho dos marcadores
#         chart = workbook.add_chart({'type': 'scatter'})

#         # Define os dados para o gráfico usando referências de células
#         chart.add_series({
#             'name': 'Dados Selecionados',
#             'categories': [worksheet.name, data_start_row + 1, data_start_col, data_start_row + num_dados, data_start_col],
#             'values':     [worksheet.name, data_start_row + 1, data_start_col + 1, data_start_row + num_dados, data_start_col + 1],
#             'marker': {
#                 'type': 'circle',
#                 'size': 5,
#                 'border': {'color': 'black'},
#                 'fill':   {'color': '#FF9900'},
#             },
#             'points': [
#                 {'fill': {'color': f'#{int(255 - (i / num_dados) * 255):02X}FF00'}} for i in range(num_dados)
#             ]
#         })
#         chart.set_title({'name': 'Gráfico 2D com 3 Eixos'})
#         chart.set_x_axis({'name': '=gráficos!$B$1'})
#         chart.set_y_axis({'name': '=gráficos!$B$2'})

#         # Inserimos o gráfico na planilha
#         worksheet.insert_chart('F5', chart)

#     # Move os arquivos originais para a pasta 'Arquivos_brutos'
#     for arquivo in arquivos_excel:
#         nome_arquivo = os.path.basename(arquivo)
#         destino = os.path.join(pasta_brutos, nome_arquivo)
#         shutil.move(arquivo, destino)

#     print("Planilhas unificadas com susso!")
#     print(f"Arquivos originais movidos para '{pasta_brutos}'.")

#================================================== Preparação do algoritmo ===================================

# Definir o caminho das pastas
dados_path = os.path.join(os.getcwd(), "Dado0s")
resultados_path = os.path.join(os.getcwd(), "Resultados")

# Verificar se as pastas existem, caso contrário, criar
if not os.path.exists(dados_path):
    os.makedirs(dados_path)

if not os.path.exists(resultados_path):
    os.makedirs(resultados_path)

parser = argparse.ArgumentParser(description='Descrição do seu programa')
parser.add_argument('--plot', action='store_true', help='Executa apenas criar_grafico_interativo_html_com_slider')
parser.add_argument('--uni', action='store_true', help='Executa apenas unificar_planilhas')
args = parser.parse_args()

# if args.plot and args.uni:
#     unificar_planilhas()
#     #criar_grafico_interativo_html_com_slider()
#     criar_app_dash()

#     sys.exit()
# elif args.plot:
#     #criar_grafico_interativo_html_com_slider()
#     criar_app_dash()
#     sys.exit()
# elif args.uni:
#     unificar_planilhas()
#     sys.exit()

#================================================== Inicio do algoritmo do Marcus ===================================

t0 = time.time()
# DADOS DE ENTRADA ALGORITIMO DE ROTA
# Definir o arquivo CSV de entrada
csv_file = os.path.join(dados_path, "field_16ha_100ha_2%_12m_0_LM.csv")

# Carregar o arquivo CSV
df = pd.read_csv(csv_file, delimiter=';')
xp = df['x[m]'].values
yp = df['y[m]'].values
a_tot = df['area.total[m2]'].values
a_mato = df['area.mato[m2]'].values
delta_theta = df['ang.proa[deg]'].values
theta_abs = df['ang.abs[deg]'].values
dist_vector = df['dist[m]'].values
status = df['operacao'].values

n_motores = 8

volume_tanque = np.arange(40,40.1,1)        #[L]
paralelos = np.arange(1,20.1,1)

area_total = 1000               # [ha]
AREA = 16 # [ha]
resultados = []
it = 0

#=========================== Área do Euler , XISPA [    não meche pufavo <:(    ] ================================


def extrair_informacoes_baterias(arquivo_excel):
    """
    Lê o arquivo Excel (.xlsx), extrai até 25 linhas de cada planilha e identifica as informações das células de bateria,
    incluindo valores presentes em colunas B e C ou apenas em C, ignorando valores N/A.

    Parameters:
        arquivo_excel (str): Caminho para o arquivo Excel

    Returns:
        pd.DataFrame: DataFrame contendo as informações das células extraídas.
    """
    # Verificar se o arquivo existe
    if not os.path.exists(arquivo_excel):
        raise FileNotFoundError(f"O arquivo '{arquivo_excel}' não foi encontrado.")

    # Dicionário para armazenar as informações extraídas
    dados_extracao = {
        "Propriedade": [],
        "Valor": [],
        "Planilha": []  # Para armazenar o nome da planilha de origem
    }

    # Ler o arquivo Excel com todas as planilhas
    xls = pd.ExcelFile(arquivo_excel, engine='openpyxl')

    # Iterar sobre cada planilha
    for sheet_name in xls.sheet_names:
        # Ler até 25 linhas da planilha
        df = pd.read_excel(xls, sheet_name=sheet_name, engine='openpyxl', nrows=25)

        # Iterar pelas linhas do DataFrame
        for index, row in df.iterrows():
            propriedade = row.iloc[0]  # Valor da coluna A (Propriedade)
            valor_b = row.iloc[1] if not pd.isna(row.iloc[1]) else None  # Valor na coluna B
            valor_c = row.iloc[2] if not pd.isna(row.iloc[2]) else None  # Valor na coluna C

            # Lógica para pegar o valor relevante entre as colunas B e C
            if valor_b and valor_b != 'N/A':
                valor = valor_b
            elif valor_c and valor_c != 'N/A':
                valor = valor_c
            else:
                valor = "Valor não disponível"

            # Armazenar a propriedade, o valor e o nome da planilha no dicionário
            dados_extracao["Propriedade"].append(propriedade)
            dados_extracao["Valor"].append(valor)
            dados_extracao["Planilha"].append(sheet_name)

    # Converter o dicionário para um DataFrame para facilitar a visualização
    dados_bateria_df = pd.DataFrame(dados_extracao)

    return dados_bateria_df

# Caminho para o arquivo Excel 'Database_de_Baterias.xlsx'
#arquivo_excel = 'Database/Database_de_Baterias.xlsx'

# Chama a função para extrair informações
#informacoes_celulas = extrair_informacoes_baterias(arquivo_excel)

# Configura o pandas para exibir todas as linhas do DataFrame
#pd.set_option('display.max_rows', None)
#pd.set_option('display.max_columns', None)

# Imprime o DataFrame completo
#print(informacoes_celulas)

def plotar_celulas(informacoes_celulas, V_usuais):
    """
    Plota gráficos de Tensão x Corrente Requerida para cada célula baseada nas taxas de descarga.

    Parameters:
        informacoes_celulas (dict): Dicionário com as informações das células extraídas das planilhas.
        V_usuais (list): Lista com valores de tensões usuais para células em série
    """
    for planilha, dados in informacoes_celulas.items():
        # Obter a capacidade e a tensão nominal
        capacidade = dados.loc[dados.iloc[:, 0] == 'Capacidade Nominal [mAh]', dados.columns[1]].values[0]
        tensao_nominal = dados.loc[dados.iloc[:, 0] == 'Tensão na Célula [V]', dados.columns[1]].values[0]
        taxa_descarga_max = dados.loc[dados.iloc[:, 0] == 'Taxa de Descarga [C]', dados.columns[1]].values[0]
        
        # Calcular a corrente máxima
        I_max = (capacidade / 1000) * taxa_descarga_max  # Convertendo mAh para Ah

        # Plotar gráfico baseado na taxa de descarga máxima
        correntes_requeridas = [I_max / V for V in V_usuais]

        plt.figure(figsize=(10, 6))
        plt.plot(V_usuais, correntes_requeridas, marker='o')
        plt.title(f'Tensão x Corrente Requerida para a célula da planilha {planilha}')
        plt.xlabel('Tensão (V)')
        plt.ylabel('Corrente Requerida (A)')
        plt.grid(True)
        plt.show()

#def define_tesao(volume_tanque, A, M_tot_in, informacoes_celulas):
def define_tesao(volume_tanque, A, M_tot_in):
    """
    Calcula a corrente requerida para cada valor de tensão em V_usuais.
    Gera um gráfico de Corrente x Tensão.

    Parameters:
        volume_tanque (float): Tamanho do tanque em Litros
        A (float): Área (calculada pela relação com massa total e número de motores)
        M_tot_in (float): Massa total do sistema
        informacoes_celulas (pd.DataFrame): DataFrame com as informações das células

    """
    # Definindo as variáveis----------------------------------
    rho = 1.225  # densidade do ar em kg/m^3 (exemplo)
    n = 8  # número de rotores (exemplo)
    TOW = 2 * volume_tanque  # Takeoff weight (peso de decolagem) em Newtons (exemplo)
    g = 9.8  # aceleração gravitacional em m/s^2
    V_usuais = [12, 14, 18, 24, 32, 48]  # S de células
    V_nom = 4.2
    I_cel = []  # S de células

    #Características da célula (Atualizar com database depois) --------------------------
    # Welion Solid State
    Capacidade_nom = 22 #Ah. Vc que se lasque Marcus >:(
    C_rate = 6.13
    Ciclos = 800
    Massa = 299 #gramas
    
    # Multiplicação de V_usuais por V_nom
    V_usuais = [v * V_nom for v in V_usuais]  # Corrige a multiplicação para operar sobre a lista

    #print("V_usuais após multiplicação:", V_usuais)

    # Cálculo do termo
    W_G = ((1.25 / 0.75) * (TOW*g / (2 * rho * A * n))**0.5) * (g)  # não converte para kg. Pois o marcus adora tabalhar com Toneladas. KiloAmpere. Então aqui vai ser kg/W
    P = M_tot_in*W_G
    
    I_max = C_rate*Capacidade_nom
    V = P/I_max
        
    if Debug_Fit_Celulas_Bat:
        print(f"g/W = {1000/W_G}")
        print(f"Potência necessária [watts]: {P}")
        print(f"Tensão [V]: {V}")

    
    return 0

# ==========================================================================================================

for bb, M_pulv_max in enumerate(volume_tanque):
    #print("Tanque [L]: ",M_pulv_max,round(bb/(len(volume_tanque)-1)*100,2),"%")
    voo_vector = []
    dias = []
    tempo_manobra = []
    EOC_hr = []
    vol_conb_consumido = [] #[kg]
    tempo_missao = []   #[s]
    
    RESULTADOS = []
    RESULTADOS.append(f'OP\t\tSTATUS\ti\tj\tx\ty\ttheta\tM_pulv\tE_bat\tPreq_tot\tv\tw\tvz')
    RTLS = []
    
    operacao = []
    TANQUES = []
    tempo_idas_vindas = []
    Tempo_rtw = 0
    talhao_maximus = []
    voo_vector = []
    M_pulvs = []
    MTOW = []
    capacidade_vector = []
    vol_comb = []
    dist_percorrida = []
    dist_pulverizando = []
    vazao_bombas = []
    EOC_km = []
    area_por_voo = []
    faixa_vector = []
    tempo_por_voo = []
    tempo_util = []
    capex = []
    drone_e_bateria = []
    abastecimentos = []
    preco_drone_bateria = []
    CALDA_CONS = []
    preco_por_ha_real = []
    tanque = 0
    AUTONOMIA_PROJ = []
    MASSA_ESTRUTURA = []
    COMBUSTIVEL = []
    GERADOR = []
    VPULV = []
    VDESLOC = []
    PGERADOR = []
    CV_CAP = []
    SERIE = []
    PARALELOS = []
    ENERGIA = []
    M_BATERIA = []
    for aa in range(len(paralelos)):
        
        #BATERIA
        cap_max = paralelos[aa]*8              #[Ah]
        serie = 14
        m_celula = 0.2142857                    #[kg]
        M_bat = m_celula*serie*paralelos[aa]       #[kg]
        tensao_nominal = 3.7*serie              #[V]
        E_bat_max = cap_max*tensao_nominal      #[Ah]
        
        P_sensores = 0;
        P_LED = 100
        P_sist = 38.71
        P_bombas = 95.04
        
        
        # MASSAS
        M_estrut = (18.3611161042012*np.log(M_pulv_max) - 30.178770579692)       #[kg]
        #M_bat = 12      #[kg]
        M_tot_in = M_bat + M_pulv_max + M_estrut
        
        
        ## PROPULSAO
        eta_escmotor = 0.848                                                           # Eficiência ESC e Motor
        eta_helice = 0.7719                                                            # Eficiência hélice
        rho = 1.225
        cnst = 1/(eta_escmotor*eta_helice)
        A =  (0.0431*(M_pulv_max + M_estrut) + 0.7815)/n_motores
        Diametro = (4*A/np.pi)**0.5/0.0254   
        
        
        # PULVERIZAÇÃO
        v_pulv = 1.0                              #[m/s]
        v_desloc = 10.0                           #[m/s]
        v_subida = 2.5  
        v_descida = -v_subida
        omega = 40.0                              #[graus/s]
        Tempo_rtl = 0

        t_prep = 300; t_abs_calda = M_pulv_max*60/50+20; t_desloc_pre_op = 2520  
        t_desloc_pos_op = t_desloc_pre_op ;
        t_triplice_lavagem = 6*M_pulv_max/(6*2)*60;
        t_lavagem_limpeza = (2*(0.7*M_pulv_max/(6*2))+5)*60
        
        n_abs = 0
        n = 1
        voo = 1                        
        cons_pulv = []
        M_tot = []
        M_pulv = []; M_pulv.append(M_pulv_max)
        E_bat = []; E_bat.append(E_bat_max)

        
        # TEMPOS
        t = []; t.append(t_prep + t_desloc_pre_op)
        t_pulv = []; t_pulv.append(0.0)
        t_manobra = []; t_manobra.append(0.0)
        t_de_voo = []; t_de_voo.append(0)
        calda_cons = []; calda_cons.append(0)
        dt = 1                  #[s]
    
        OP = []
        STATUS = []
        v = []
        vz = []
        w = []
        dist_percorr = []; dist_percorr.append(0)
        dist_pulv = []; dist_pulv.append(0)
        rtl_acumulado = 0
        
        x = []; x_rtl = 0
        y = []; y_rtl = 0
        z = []; z_rtl = 0
        theta = []; theta_rtl = 0
        alpha = 0
        zi = 5
        
        x.append(0.0)
        y.append(0.0)
        z.append(0.0)
        theta.append(0.0)
        autonomia = [];
        dist_rtl = [];
        dist_rtl.append(0)
        produtiv_por_voo = []
        M_tot.append(M_tot_in)
        

        
        T_hover = []; PWM_hover = []; T_M1 = []; PWM_M1 = []; ef_M1 = []; Preq_M1 = []
        
        T_M2 = []; PWM_M2 = []; ef_M2 = []; Preq_M2 = []
        
        T_M3 = []; PWM_M3 = []; ef_M3 = []; Preq_M3 = []
        
        T_M4 = []; PWM_M4 = []; ef_M4 = []; Preq_M4 = []
        
        T_M5 = []; PWM_M5 = []; ef_M5 = []; Preq_M5 = []
        
        T_M6 = []; PWM_M6 = []; ef_M6 = []; Preq_M6 = []
        
        T_M7 = []; PWM_M7 = []; ef_M7 = []; Preq_M7 = []
        
        T_M8 = []; PWM_M8 = []; ef_M8 = []; Preq_M8 = []
        
        Preq_prop = []; P_gerador = []
        vazao_vetor = []
        
        Preq_tot = [];
        flag = "off"
        i = 0
        OP.append("DESLOCANDO")
        #for j,dist in enumerate(dist_vector):
        j = 0
        xis = 0
        yplis = 0
        vazao_vetor.append(0)
        
        while OP[i] != "FIM\t" and flag =="off":
            tiro_percurso = 0.06*a_mato[j]    #[kg] ou [L] de pulv
            if OP[i] == "DESLOCANDO":
                if z[i] < zi:
                    vz.append(v_subida)
                    v.append(0.0)
                    w.append(0.0)
                    STATUS.append("SUBIDA")
                elif status[j] == "p":
                   vz.append(0.0)
                   v.append(v_pulv)
                   w.append(0.0)
                   STATUS.append("PITCH")
                elif status[j] == "y":
                    if(delta_theta[j] > 0):
                        vz.append(0.0)
                        v.append(0.0)
                        w.append(omega)
                        STATUS.append("YAW+")
                    else:
                        vz.append(0.0)
                        v.append(0.0)
                        w.append(-omega)
                        STATUS.append("YAW-")
                        
                if STATUS[i] == "YAW+":
                    if (theta[i] + w[i] * dt > theta_abs[j+1]):
                        theta.append(theta_abs[j+1])
                    else:
                        theta.append(theta[i] + w[i] * dt)
                elif STATUS[i] == "YAW-":
                    if (theta[i] + w[i] * dt < theta_abs[j+1]):
                        theta.append(theta_abs[j+1])
                    else:
                        theta.append(theta[i] + w[i] * dt)
                else:
                    theta.append(theta[i] + w[i] * dt)
                    
                if (z[i] + vz[i] * dt > zi):
                    z.append(zi)
                else:
                    z.append(z[i] + vz[i] * dt)
                
                if ( 
                    (status[j] == "y" and abs(delta_theta[j]) < 0.1) or
                    (status[j] == "p" and dist_vector[j]== 0)
                ):
                    vazao = 0
                    delta_x = 0
                    delta_y = 0
                  
                    #print("CHEGOU")
                elif status[j] == "y":
                    vazao = tiro_percurso/(abs(delta_theta[j]/omega))           #[kg/s] ou [L/s] durante o intervalo dt
                    delta_x = 0
                    delta_y = 0
                else:
                    delta_x = v[i] * math.cos(math.radians(theta_abs[j])) * dt
                    delta_y = v[i] * math.sin(math.radians(theta_abs[j])) * dt
                    vazao = tiro_percurso/(dist_vector[j]/v_pulv)
                x.append(x[i] + delta_x)
                y.append(y[i] + delta_y)
                            
                
                if vazao > tiro_percurso:
                    vazao = tiro_percurso/dt
                
                if (
                    (delta_x > 0 and x[i+1] > xp[j+1])
                    or (delta_x < 0 and x[i+1] < xp[j+1])
                ):
                    x[i+1] = xp[j+1] 
                    
                if (
                    (delta_y > 0 and y[i+1] > yp[j+1]) 
                    or (delta_y < 0 and y[i+1] < yp[j+1])
                ):
                    y[i+1] = yp[j+1]
                    

                
            if OP[i] == "RTL-CALDA\t" or OP[i] == "RTL-BAT\t" or OP[i] == "RTL-FIM\t":
                vazao = 0
                if theta_rtl > alpha:
                    if theta[i] > alpha:
                        vz.append(0.0)
                        v.append(0.0)
                        w.append(-omega)
                        STATUS.append("YAW-")
                    
                    elif (x[i] > 0 or y[i] > 0):
                        vz.append(0.0)
                        v.append(v_desloc)
                        w.append(0.0)
                        STATUS.append("PITCH")
                    elif z[i] > 0:
                        vz.append(-v_subida)
                        v.append(0.0)
                        w.append(0.0)
                        STATUS.append("DESCIDA")
                
                elif theta_rtl < alpha:
                    if theta[i] < alpha:
                        vz.append(0.0)
                        v.append(0.0)
                        w.append(omega)
                        STATUS.append("YAW+")
                    
                    elif (x[i] > 0 or y[i] > 0):
                        vz.append(0.0)
                        v.append(v_desloc)
                        w.append(0.0)
                        STATUS.append("PITCH")
                    elif z[i] > 0:
                        vz.append(-v_subida)
                        v.append(0.0)
                        w.append(0.0)
                        STATUS.append("DESCIDA")
               
                if STATUS[i] == "YAW+":
                    if (theta[i] + w[i] * dt > alpha):
                        theta.append(alpha)
                    else:
                        theta.append(theta[i] + w[i] * dt)
                elif STATUS[i] == "YAW-":
                    if (theta[i] + w[i] * dt < alpha):
                        theta.append(alpha)
                    else:
                        theta.append(theta[i] + w[i] * dt)
                else:
                    theta.append(theta[i])
                    
                if (z[i] + vz[i] * dt < 0):
                    z.append(0)
                else:
                    z.append(z[i] + vz[i] * dt)
                    
                if (x[i] - abs(v[i]*math.sin(math.radians(theta[i])) * dt) < 0):
                    x.append(0)
                else:
                    x.append(x[i] - abs(v[i]*math.sin(math.radians(theta[i])) * dt))
                    
                if (y[i] - abs(v[i] * math.cos(math.radians(theta[i])) * dt) < 0):
                    y.append(0)
                else:
                    y.append(y[i] - abs(v[i] * math.cos(math.radians(theta[i])) * dt))
                    
            if OP[i] == "RTW\t":
                vazao = 0
                if theta_rtl < 90 - alpha or theta_rtl == 0 : # REVER
                    if z[i] < z_rtl:
                         vz.append(v_subida)
                         v.append(0.0)
                         w.append(0.0)
                         STATUS.append("SUBIDA")
                    elif theta[i] < 90 - alpha and x[i] == 0:
                         vz.append(0.0)
                         v.append(0.0)
                         w.append(omega)
                         STATUS.append("YAW+")
                    elif (x[i] < x_rtl or y[i] < y_rtl):
                         vz.append(0.0)
                         v.append(v_desloc)
                         w.append(0.0)
                         STATUS.append("PITCH")
                    elif theta[i] > theta_rtl and x[i] == x_rtl:
                         vz.append(0.0)
                         v.append(0.0)
                         w.append(-omega)
                         STATUS.append("YAW-2")
                    
                         
                    if STATUS[i] == "YAW+" and theta[i] + w[i] * dt > 90 - alpha:
                        theta.append(90 - alpha)
                    elif STATUS[i] == "YAW-2" and theta[i] + w[i] * dt < theta_rtl:
                        theta.append(theta_rtl)
                    else:
                        theta.append(theta[i] + w[i] * dt)
                        
                    if (z[i] + vz[i] * dt > z_rtl):
                         z.append(z_rtl)
                    else:
                         z.append(z[i] + vz[i] * dt)
                         
                    if (x[i] + v[i] * math.cos(math.radians(theta[i])) * dt > x_rtl):
                         x.append(x_rtl)
                    else:
                         x.append(x[i] + v[i] * math.cos(math.radians(theta[i])) * dt)
                         
                    if ((y[i] + v[i] * math.sin(math.radians(theta[i])) * dt) > y_rtl):
                         y.append(y_rtl)
                    else:
                         y.append(y[i] + v[i] * math.sin(math.radians(theta[i])) * dt)
                         
                elif theta_rtl > 90 - alpha:
                    if z[i] < z_rtl:
                         vz.append(v_subida)
                         v.append(0.0)
                         w.append(0.0)
                         STATUS.append("SUBIDA")
                    elif theta[i] < 90 - alpha and x[i] < x_rtl:
                         vz.append(0.0)
                         v.append(0.0)
                         w.append(omega)
                         STATUS.append("YAW+")
                    elif (x[i] < x_rtl or y[i] < y_rtl):
                         vz.append(0.0)
                         v.append(v_desloc)
                         w.append(0.0)
                         STATUS.append("PITCH")
                    elif theta[i] < theta_rtl:
                         vz.append(0.0)
                         v.append(0.0)
                         w.append(omega)
                         STATUS.append("YAW+2")
                         
                    if STATUS[i] == "YAW+" and theta[i] + w[i] * dt > 90 - alpha:
                        theta.append(90 - alpha)
                    elif STATUS[i] == "YAW+2" and theta[i] + w[i] * dt > theta_rtl:
                        theta.append(theta_rtl)
                    else:
                        theta.append(theta[i] + w[i] * dt)
                         
                    if (z[i] + vz[i] * dt > z_rtl):
                         z.append(z_rtl)
                    else:
                         z.append(z[i] + vz[i] * dt)
                         
                    if (x[i] + v[i] * math.cos(math.radians(theta[i])) * dt > x_rtl):
                         x.append(x_rtl)
                    else:
                         x.append(x[i] + v[i] * math.cos(math.radians(theta[i])) * dt)
                         
                    if ((y[i] + v[i] * math.sin(math.radians(theta[i])) * dt) > y_rtl):
                         y.append(y_rtl)
                    else:
                         y.append(y[i] + v[i] * math.sin(math.radians(theta[i])) * dt)        
            
            
            
#==================== CALCULO POTENCIA =================================-------#
            
            
            T_hover.append(M_tot[i]/n_motores)      # [kg]
            Cnst_PWM_T = 0.02209*Diametro - 0.43406
            COAXIAL_80 = 1
            
            if n_motores == 8:
                
                T_M1.append(T_hover[i] + Cnst_PWM_T * (- 0.8 * v[i] - 0.4035 * w[i] + 3.5 * vz[i]))
                if T_M1[i] < 0.05*T_hover[i]:
                    T_M1[i] = 0.05*T_hover[i]
                ef_M1.append(1000/9.81/(cnst*(np.sqrt(T_M1[i]*9.81/(2*rho*A)))))
                Preq_M1.append(COAXIAL_80*(1000 * T_M1[i]/ef_M1[i]))
                
                T_M2.append(T_hover[i] + Cnst_PWM_T * (0.8 * v[i] - 0.4035 * w[i] + 3.5 * vz[i]))
                if T_M2[i] < 0.05*T_hover[i]:
                    T_M2[i] = 0.05*T_hover[i]
                ef_M2.append(1000/9.81/(cnst*(np.sqrt(T_M2[i]*9.81/(2*rho*A)))))
                Preq_M2.append(COAXIAL_80*(1000 * T_M2[i]/ef_M2[i]))
                
                T_M3.append(T_hover[i] + Cnst_PWM_T * (- 0.23 * v[i] + 0.4035 * w[i] + 3.5 * vz[i]))
                if T_M3[i] < 0.05*T_hover[i]:
                    T_M3[i] = 0.05*T_hover[i]
                ef_M3.append(1000/9.8/(cnst*(np.sqrt(T_M3[i]*9.8/(2*rho*A)))))
                Preq_M3.append(COAXIAL_80*(1000 * T_M3[i]/ef_M3[i]))
                
                T_M4.append(T_hover[i] + Cnst_PWM_T * (0.8 * v[i] + 0.4035 * w[i] + 3.5 * vz[i]))
                if T_M4[i] < 0.05*T_hover[i]:
                    T_M4[i] = 0.05*T_hover[i]
                ef_M4.append(1000/9.81/(cnst*(np.sqrt(T_M4[i]*9.81/(2*rho*A)))))
                Preq_M4.append(COAXIAL_80*(1000 * T_M4[i]/ef_M4[i]))
        
                T_M5.append(T_hover[i] + Cnst_PWM_T * (- 0.8 * v[i] + 0.4035 * w[i] + 3.5 * vz[i]))
                if T_M5[i] < 0.05*T_hover[i]:
                    T_M5[i] = 0.05*T_hover[i]
                ef_M5.append(1000/9.81/(cnst*(np.sqrt(T_M5[i]*9.81/(2*rho*A)))))
                Preq_M5.append(COAXIAL_80*(1000 * T_M5[i]/ef_M5[i]))
                
                T_M6.append(T_hover[i] + Cnst_PWM_T * (0.23 * v[i] + 0.4035 * w[i] + 3.5 * vz[i]))
                if T_M6[i] < 0.05*T_hover[i]:
                    T_M6[i] = 0.05*T_hover[i]
                ef_M6.append(1000/9.8/(cnst*(np.sqrt(T_M6[i]*9.8/(2*rho*A)))))
                Preq_M6.append(COAXIAL_80*(1000 * T_M6[i]/ef_M6[i]))
                
                T_M7.append(T_hover[i] + Cnst_PWM_T * (- 0.23 * v[i] - 0.4035 * w[i] + 3.5 * vz[i]))
                if T_M7[i] < 0.05*T_hover[i]:
                    T_M7[i] = 0.05*T_hover[i]
                ef_M7.append(1000/9.8/(cnst*(np.sqrt(T_M7[i]*9.8/(2*rho*A)))))
                Preq_M7.append(COAXIAL_80*(1000 * T_M7[i]/ef_M7[i]))
                
                T_M8.append(T_hover[i] + Cnst_PWM_T * (0.23 * v[i] - 0.4035 * w[i] + 3.5 * vz[i]))
                if T_M8[i] < 0.05*T_hover[i]:
                    T_M8[i] = 0.05*T_hover[i]
                ef_M8.append(1000/9.8/(cnst*(np.sqrt(T_M8[i]*9.8/(2*rho*A)))))
                Preq_M8.append(COAXIAL_80*(1000 * T_M8[i]/ef_M8[i]))
                
                Preq_prop.append(Preq_M1[i] +  Preq_M2[i] + Preq_M3[i] + Preq_M4[i] + Preq_M5[i] + Preq_M6[i] + Preq_M7[i] + Preq_M8[i])
            if (OP[i] == "DESLOCANDO" and a_mato[j] !=0):
                Preq_tot.append(Preq_prop[i] + P_LED + P_bombas + P_sist)
                cons_pulv.append(vazao*dt)
                t_pulv.append(t_pulv[i] + dt)
                dist_pulv.append(dist_pulv[i] + math.sqrt((x[i+1]-x[i])**2 + (y[i+1]-y[i])**2 + (z[i+1]-z[i])**2))
            else:
                Preq_tot.append(Preq_prop[i] + P_LED + P_sist)
                cons_pulv.append(0.0)
                t_pulv.append(t_pulv[i])
                dist_pulv.append(dist_pulv[i])
                
            if(STATUS[i] == "YAW+" or STATUS[i] == "YAW-" or STATUS[i] == "YAW-2" or STATUS[i] == "YAW+2"):
                t_manobra.append(t_manobra[i] + dt)
            else:
                t_manobra.append(t_manobra[i])
            
            dist_percorr.append(dist_percorr[i] + math.sqrt((x[i+1]-x[i])**2 + (y[i+1]-y[i])**2 + (z[i+1]-z[i])**2))
            
            P_gerador.append(Preq_tot[i])
            
            
            E_bat.append(E_bat[i] - Preq_tot[i]*dt/3600)        #[Wh]
           
            #P_hover = (n_motores*(COAXIAL_80*(1000 * (M_tot[i]/n_motores)/(1000/9.81/(cnst*(np.sqrt((M_tot[i]/n_motores)*9.81/(2*rho*A))))))))
            autonomia.append(3600*E_bat[i]/(Preq_tot[i]))  #(Preq_tot[i]))
            # print(E_bat[i],(Preq_tot[i]*dt/3600))
            
            if M_pulv[i] - cons_pulv[i] < 0:
                M_pulv.append(0)
            else:
                M_pulv.append(M_pulv[i] - cons_pulv[i])
                
            M_tot.append(M_estrut + M_pulv[i+1] + M_bat)
                
            calda_cons.append(calda_cons[i] + cons_pulv[i] )
            
            dist_rtl.append(math.sqrt(x[i+1]**2 + y[i+1]**2 + z[i+1]**2))
            vazao_vetor.append(vazao*60)       #[L/min]
            t.append(t[i] + dt)
            t_de_voo.append(t_de_voo[i] + dt)
            
#=========================== OP[i+1] ==================================================###
            if (OP[i] == "RTL-FIM\t"):
                if (x[i+1] == 0 and y[i+1] == 0 and z[i+1] == 0):
                    OP.append("FIM\t")
                    STATUS.append("CONCLUIDO\t")
                    t[i+1] = t[i+1] + t_lavagem_limpeza + t_triplice_lavagem + t_desloc_pos_op
                else:
                    OP.append("RTL-FIM\t")
            elif(abs(x[i+1] - xp[j+1]) < 1e-2 and abs(y[i+1] - yp[j+1]) < 1e-2 and j == (len(dist_vector) -2)):
                theta_rtl = theta[i+1]
                alpha = math.atan2(x[i+1],y[i+1])*180/math.pi
                x_rtl = x[i+1]
                y_rtl = y[i+1]
                z_rtl = z[i+1]
                OP.append("RTL-FIM\t")
                rtl_acumulado = rtl_acumulado + math.sqrt(x_rtl**2 + y_rtl**2)
            elif(M_pulv[i+1] == 0 and (STATUS[i] != "YAW+" and STATUS[i] != "YAW-" )):
                if (x[i+1] == 0 and y[i+1] == 0 and z[i+1] == 0):
                    OP.append("RTW\t")
                    E_bat[i+1] = E_bat_max
                    M_pulv[i+1] = M_pulv_max
                    t[i+1] = t[i+1] + t_abs_calda
                    theta[i+1] = -alpha - 90
                    voo = voo + 1
                    #n_abs = n_abs + 1
                elif (OP[i] == "DESLOCANDO"):
                     theta_rtl = theta[i+1]
                     alpha = math.atan2(x[i+1],y[i+1])*180/math.pi
                     x_rtl = x[i+1]
                     y_rtl = y[i+1]
                     z_rtl = z[i+1]
                     OP.append("RTL-CALDA\t")
                     rtl_acumulado = rtl_acumulado + math.sqrt(x_rtl**2 + y_rtl**2)
                else:
                    OP.append("RTL-CALDA\t")
                    
            elif(OP[i] == "RTL-BAT\t" or (v_desloc*autonomia[i] <= dist_rtl[i+1] and STATUS[i] != "YAW+" and STATUS[i] !="YAW-")):
                if(OP[i] == "RTW\t"):
                    OP.append("FIM\t")
                    STATUS.append("INCOMLPLETO")
                    print(i,"->>RTL-BAT\t")
                
                if (x[i+1] == 0 and y[i+1] == 0 and z[i+1] == 0):
                    OP.append("RTW\t")
                    #print(round(M_pulv_max,3),round(M_comb_max,3),round(M_comb[i],3),round(M_pulv[i],3))
                    E_bat[i+1] = E_bat_max
                    M_pulv[i+1] = M_pulv_max
                    n_abs = n_abs + 1
                    t[i+1] = t[i+1] + t_abs_calda
                    theta[i+1] = -alpha - 90
                    voo = voo + 1
                    if theta[i+1] < 0:
                        theta[i+1] = theta[i+1] + 180   
                elif (OP[i] != "RTL-BAT\t"):
                    theta_rtl = theta[i+1]
                    alpha = math.atan2(x[i+1],y[i+1])*180/math.pi
                    x_rtl = x[i+1]
                    y_rtl = y[i+1]
                    z_rtl = z[i+1]
                    OP.append("RTL-BAT\t")
                    rtl_acumulado = rtl_acumulado + math.sqrt(x_rtl**2 + y_rtl**2)
                    #print(i,j)
                else:
                    OP.append("RTL-BAT\t")
                    #print(i,j)
            elif(OP[i]=="DESLOCANDO"):
                if status[j] == "p":
                    if (abs(x[i+1] -  xp[j+1]) < 1e-2 and abs(y[i+1] - yp[j+1]) < 1e-2 and z[i+1] == zi):
                        OP.append("DESLOCANDO")
                        j = j + 1
                    else:
                        OP.append("DESLOCANDO")
                elif status[j] == "y":
                    if (theta[i+1] == theta_abs[j+1]):
                        OP.append("DESLOCANDO")
                        j = j + 1
                    else:
                        OP.append("DESLOCANDO") 
            elif(OP[i] == "RTW\t"):
                if (x[i+1] == x_rtl  and y[i+1] ==  y_rtl and z[i+1] == z_rtl and theta[i+1] == theta_rtl):
                    OP.append("DESLOCANDO")
                else:
                    OP.append(OP[i])
                # if j > 1000:
                #     flag = "on"

            else:
                OP.append(OP[i])
            
            if (OP[i] == "RTL-BAT\t" or OP[i] == "RTL-CALDA\t" or OP[i] == "RTL-FIM\t" or OP[i] == "RTW\t"):
                Tempo_rtl = Tempo_rtl + dt
            
            
                
            RESULTADOS.append(f'{OP[i]}\t{STATUS[i]}\t{i:.1f}\t{j:.1f}\t{x[i]:.1f}\t{y[i]:.1f}\t{theta[i]:.1f}\t{M_pulv[i]:.1f}\t{E_bat[i]:.1f}\t{Preq_tot[i]:.1f}\t{v[i]:.1f}\t{w[i]:.1f}\t{vz[i]:.1f}')
            # if i == 41322:
            #     flag = "on"
            # else:
            i = i + 1
        print(f"{time.time()-t0:.2f}s")
        
        # plt.plot(x, y)
        # # Adicionar título e rótulos aos eixos
        # plt.title('Volume de calda [L]:' + str(M_pulv_max))  # Título do gráfico
            
            
#---------------- VETORES PARA RESULTADOS ---------------------------------#
        tempo_missao.append(max(t) / 3600)  # Removido o arredondamento
        # dias.append(math.ceil(area_total/(X**2/10**4)))
        MTOW.append(M_tot_in)  # Removido o arredondamento
        voo_vector.append(voo)
        # area_por_voo.append(X**2/10**4/voo)
        #vol_comb.append(comb_cons[i] / 0.715)  # Removido o arredondamento
        dist_percorrida.append(dist_percorr[i] / 1000)  # Removido o arredondamento
        dist_pulverizando.append(dist_pulv[i] / 1000)  # Removido o arredondamento
        EOC_km.append(dist_pulv[i] / dist_percorr[i])  # Removido o arredondamento
        EOC_hr.append(t_pulv[i] / t_de_voo[i])  # Removido o arredondamento
        abastecimentos.append(n_abs)
        capacidade_vector.append(cap_max)
        SERIE.append(serie)
        PARALELOS.append(paralelos[aa])
        ENERGIA.append(E_bat_max)
        M_BATERIA.append(M_bat)
        TANQUES.append(M_pulv_max)
        RTLS.append(rtl_acumulado)  # Removido o arredondamento
        CALDA_CONS.append(calda_cons[i])  # Removido o arredondamento
        operacao.append(STATUS[i])
        
        # (preco_por_ha_real[tanque],preco_drone_bateria[tanque]) = custos(M_pulv_max,cap_bat,area_total,t_de_voo[i]/3600,comb_cons[i]/0.715,voo,n_abs,math.ceil(area_total/(X**2/10**4)),X**2/10**4)
        # capex[tanque] = produtividade[tanque]/preco_drone_bateria[tanque]
        tempo_util.append(t_de_voo[i] / 3600)  # Removido o arredondamento
        tempo_idas_vindas.append(Tempo_rtl / 3600)  # Removido o arredondamento
        tempo_manobra.append(t_manobra[i] / 3600)  # Removido o arredondamento
        tempo_por_voo.append(t_de_voo[i] / voo / 60)  # Removido o arredondamento
        
       # AUTONOMIA_PROJ.append(M_comb_max / ((((0.009424 * P_gerador_max) * 1 / 60) * (1 - ganho_cons)) / 1000) / 3600)  # Removido o arredondamento
        MASSA_ESTRUTURA.append(M_estrut)  # Removido o arredondamento
        #COMBUSTIVEL.append(M_comb_max)  # Removido o arredondamento
        #GERADOR.append(M_gerador)  # Removido o arredondamento
        #PGERADOR.append(P_gerador_max)  # Removido o arredondamento
        VPULV.append(v_pulv)
        VDESLOC.append(v_desloc)
        
        
        #if  abs(n_abs - min(n_abs))/min(n_abs) >  :
        #    print("t2[h]: ",round(tempo_missao[len(tempo_missao)-1],4) ,"// t1[h]: ",round(tempo_missao[len(tempo_missao)-2],4) )
        #    break
        # else:
        #     M_comb_max = M_comb_max + dcomb

        
        # X0 = X - 10; Y0 = X0
        # print("-->",M_pulv_max,faixa, X, max(t)/3600,capex[tanque],tanque)
        # tanque = tanque + 1
        # M_pulv_max = M_pulv_max + delta_pulv
        # # fig, axs = plt.subplots()
        # # axs.plot(x,y)
        # # axs.set_title("Pulverizante = " + str(M_pulv_max) + " kg")
        
        # for i in range(len(x) - 1):
        #     # Determinar a cor da linha com base no valor de OP[i]
        #     if OP[i] == 'RTL' or OP[i] == 'RTW':
        #         color = 'black'
        #     else:
        #         color = 'blue'  # Cor padrão para outros valores
        # #     # Plotar a linha entre (x[i], y[i]) e (x[i+1], y[i+1])
        # plt.plot([x[i], x[i+1]], [y[i], y[i+1]], color=color)
                
            # Criar o gráfico
        
        
    #======================= PLOTS =======================================#
    produtividade = [AREA/x for x in tempo_missao]
    
    data = {
        # "Faixa [m]": faixa_vector,
        "Volume de calda [L]": [round(x, 2) for x in TANQUES], 
        "STATUS": operacao,
        "Capacidade operacional [ha/h]": [round(x, 2) for x in produtividade],
        "Tempo de missao [h]": [round(x, 2) for x in tempo_missao],
        "Tempo util [h]": [round(x, 2) for x in tempo_util],
        "Tempo por voo [min]": [round(x, 2) for x in tempo_por_voo],
        #"Autonomia Projetada [h]": [round(x, 2) for x in AUTONOMIA_PROJ],
        "N° Voos": [round(x, 2) for x in voo_vector],
        # "Tempo operacional dias": dias,
        "MTOW [kg]": [round(x, 2) for x in MTOW],
        "Massa Estrutura [kg]": [round(x, 2) for x in MASSA_ESTRUTURA],
        #"Combustível [kg]": [round(x, 2) for x in COMBUSTIVEL],
        "Abastecimentos": [round(x, 2) for x in abastecimentos],
        #"Massa gerador": [round(x, 2) for x in GERADOR],
        #"Potência gerador [W]": [round(x, 2) for x in PGERADOR],
        # "Area pulverizada por dia [ha]": talhao_maximus,
        # "Hectare por voo [ha/voo]": area_por_voo,
        "Capacidade bateria [Ah]": capacidade_vector,
        "S": SERIE,
        "P": PARALELOS,
        "Energia [Wh]":ENERGIA,
        "Massa bateria [kg]": M_BATERIA,
        #"Combustível consumido [L]": [round(x, 2) for x in vol_comb],
        "Calda cons [L]": [round(x, 2) for x in CALDA_CONS],
        # "Vazao [L/min]": vazao_bombas,
        "Distância percorrida [km]": [round(x, 2) for x in dist_percorrida],
        "Distância Pulverizando [km]": [round(x, 2) for x in dist_pulverizando],
        "RTL ACUMULADO[m]": [round(x, 2) for x in RTLS],
        "EOC [km/km]": [round(x, 2) for x in EOC_km],
        "EOC [h/h]": [round(x, 2) for x in EOC_hr],
        "Tempo de manobra [h]": [round(x, 2) for x in tempo_manobra],
        "Tempo rtl_rtw [h]": [round(x, 2) for x in tempo_idas_vindas],
        "V pulv [m/s]": [round(x, 2) for x in VPULV],
        "V desloc [m/s]": [round(x, 2) for x in VDESLOC],
    }


 
    #     "CAPEX [ha/h/R$]": capex,
    #     "Preco por ha real": preco_por_ha_real,
    #     "Preco drone, carreg e bat": preco_drone_bateria,

    
    
    # vol = []
    # for k in range(len(MTOW)):
        #vol.append(str(delta_pulv*k + M_pulv_min) + "[L]")
    #print(produtividade)
    
    # produtividade_matriz[it] = produtividade
    # capex_matriz[it] = capex
    # faixa = faixa + 1
    #df1 = pd.DataFrame(data)
    # resultados.append(data)
    # it = it + 1
    
    vol = []
    # for k in range(len(volume_tanque)):
        #vol.append(str(delta_pulv*k + min(volume_tanque)) + "[L]")
    #print(produtividade)
    
    #produtividade_matriz[it] = produtividade
    #capex_matriz[it] = capex
    #faixa = faixa + 1
    df1 = pd.DataFrame(data)
    resultados.append(df1)
    it = it + 1
    
# Criar o gráfico
# plt.figure(figsize=(8, 6))

# # Plotar os vetores como pontos
# plt.scatter(x, y, color='red', label='Pontos')

# def gerar_superficie(x, y, z):
#     # Converte listas para arrays NumPy
#     x_interp, y_interp = np.meshgrid(np.array(x), np.array(y))

#     # Interpolação 2D usando griddata
#     z_interp = griddata((x_interp.flatten(), y_interp.flatten()), np.array(z).flatten(), (x_interp, y_interp), method='cubic')

#     return x_interp, y_interp, z_interp

# x_interp, y_interp, z_interp = gerar_superficie(volume_tanque, faixas, produtividade_matriz)

# def plotar_superficie_3d(x_interp, y_interp, z_interp):
#     fig = plt.figure(figsize=(12, 8))
#     ax = fig.add_subplot(111, projection='3d')
#     surf = ax.plot_surface(x_interp, y_interp, z_interp, cmap='hot')
#     fig.colorbar(surf, ax=ax, shrink=.5, aspect=10, label='Produtividade [ha/h]')
#     ax.set_xlabel('Volume do tanque [L]')
#     ax.set_ylabel('Faixa [m]')
#     ax.set_title('Produtividade [ha/h]')
#     plt.show()

# plotar_superficie_3d(x_interp, y_interp, z_interp)

# fig = go.Figure(data=[go.Surface(z=z_interp, x=x_interp, y=y_interp)])
# fig.update_layout(title='Produtividade [ha/h]')
#               #width=500, height=500,
#               #margin=dict(l=65, r=50, b=65, t=90))
# fig.show()
# fig.write_html("output.html")  



## =======================   GERAR PLANILHA =========================##

# fig1, figure1 = plt.subplots()
# a = np.arange(M_pulv_min,M_pulv_lim + 1,delta_pulv)
# for i in range(len(resultados)):
#       b = np.array(resultados[i].loc[:, "EOC [km/km]"])
#       figure1.plot(a, b, color=intervalos_cores[i][1], label=str(
#           resultados[i].loc[str(M_pulv_min) + "[L]", "Faixa [m]"]))

# figure1.legend(loc='upper left', bbox_to_anchor=(0.0, 1), borderaxespad=0.1)


#for k in range(len(resultados)):
#    resultados[k].to_excel("RESULTADOS " +str(k) + " .xlsx")

intervalos_cores = [
    (0, 'red'),
    (1, 'blue'),
    (2, 'green'),
    (3, 'purple'),
    (4, 'orange'),
    (5, 'yellow'),
    (6, 'brown'),
    (7, 'pink'),
    (8, 'cyan'),
    (9, 'gray'),
    (10, 'red'),
    (11, 'blue'),
    (12, 'green'),
    (13, 'purple'),
    (14, 'orange'),
    (15, 'yellow'),
    (16, 'brown'),
    (17, 'pink'),
    (18, 'black'),
    (19, 'blue'),
    (20, 'green'),
    (21, 'purple'),
    (22, 'orange'),
    (23, 'yellow'),
    (24, 'brown'),
    (25, 'pink'),
    (26, 'cyan'),
    (27, 'gray'),
    (28, 'red'),
    (29, 'blue'),
    (30, 'green'),
    (31, 'purple'),
    (32, 'orange'),
    (33, 'yellow'),
    (34, 'brown'),
    (35, 'pink'),
    (36, 'black'),
    (37, 'red'),
    (38, 'blue'),
    (39, 'green'),
    (40, 'purple'),
    (41, 'orange'),
    (42, 'yellow'),
    (43, 'brown'),
    (44, 'pink'),
    (45, 'cyan'),
    (46, 'gray'),
    (47, 'red'),
    (48, 'blue'),
    (49, 'green'),
    (50, 'purple'),
    (51, 'green'),
    (52, 'purple'),
    (53, 'orange'),
    (54, 'yellow'),
    (55, 'brown'),
    (56, 'pink'),
    (57, 'cyan'),
    (58, 'gray'),
    (59, 'red'),
    (60, 'blue'),
    (61, 'green'),
    (62, 'purple'),
    (63, 'orange'),
    (64, 'yellow'),
    (65, 'brown'),
    (66, 'pink'),
    (67, 'black'),
    (68, 'red'),
    (69, 'blue'),
    (70, 'green'),
    (71, 'purple'),
    (72, 'orange'),
    (73, 'yellow'),
    (74, 'brown'),
    (75, 'pink'),
    (76, 'cyan'),
    (77, 'gray'),
    (78, 'red'),
    (79, 'blue'),
    (80, 'green'),
    (81, 'purple'),
]



t_horas =np.array(t)/3600
# Função para atribuir cor com base no valor em t
def atribuir_cor(valor_t):
    for intervalo, cor in intervalos_cores:
        if valor_t <= intervalo:
            return cor
    return intervalos_cores[-1][1]

# # Iterar sobre os vetores e a lista OP




# Atribuir cores com base no vetor t
cores = [atribuir_cor(valor) for valor in t_horas]

fig_trajeto = go.Figure()
# # Adicionar linhas aos eixos x, y e z

fig_trajeto.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='lines', line=dict(color=cores, colorscale='Viridis', width=2),))
fig_trajeto.update_layout(paper_bgcolor="white", showlegend=False, scene=dict(aspectmode='data'))
fig_trajeto.write_html("saida_eletrico.html")

# Ao final de cada simulação, salvar os resultados na pasta `Resultados`
for k in range(len(resultados)):
    resultado_file = os.path.join(resultados_path, f"RESULTADOS_{k}.xlsx")
    resultados[k].to_excel(resultado_file)

#criar_grafico_interativo_html_com_slider()

# # Salvar o gráfico 3D na pasta `Resultados`
# saida_html = os.path.join(resultados_path, "saida_sem_discreziar.html")
# fig_trajeto.write_html(saida_html)

# numeric_columns = df.columns.tolist()

# # Substitua vírgulas por pontos e converta para float
# for col in numeric_columns:
#     df[col] = df[col].astype(str).str.replace(',', '.').astype(float)

