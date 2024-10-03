import math
import matplotlib.pyplot as plt
import plotly.io as pio
import plotly.express as px
# from plotly.offline import plot
pio.renderers.default='browser'
# from scipy.interpolate import griddata
#from CUSTOS_ELETRICO import custos
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



def unificar_planilhas(output_folder):
    """
    Unifica várias planilhas Excel em uma única, com opções interativas e gráficos 2D.
    Também move os arquivos originais para uma subpasta 'Arquivos_brutos'.
    """
    # Caminho para a pasta 'Resultados'
    pasta_resultados = output_folder
    #pasta_brutos = os.path.join(pasta_resultados, 'Arquivos_brutos')
    pasta_brutos = "resultados-brutos"

    # Cria a pasta 'Arquivos_brutos' se não existir
    if not os.path.exists(pasta_brutos):
        os.makedirs(pasta_brutos)

    # Lista todos os arquivos xlsx na pasta 'Resultados' com o padrão de nomenclatura
    arquivos_excel = glob.glob(os.path.join(pasta_resultados, 'RESULTADOS_*.xlsx'))

    # Dicionário para armazenar DataFrames
    planilhas = {}

    # Lê cada arquivo e armazena no dicionário
    for arquivo in arquivos_excel:
        nome_arquivo = os.path.basename(arquivo)
        nome_planilha = os.path.splitext(nome_arquivo)[0]
        df = pd.read_excel(arquivo)
        planilhas[nome_planilha] = df

    # Caminho para a planilha unificada
    caminho_planilha_unificada = os.path.join(pasta_resultados, 'Planilha_Unificada.xlsx')

    # Cria um objeto ExcelWriter para escrever em um novo arquivo Excel
    with pd.ExcelWriter(caminho_planilha_unificada, engine='xlsxwriter') as escritor:
        # Escreve cada DataFrame em uma aba separada
        for nome_planilha, df in planilhas.items():
            df.to_excel(escritor, sheet_name=nome_planilha, index=False)

        # Concatena todos os DataFrames em um só
        df_resultados = pd.concat(planilhas.values(), ignore_index=True)
        # Escreve o DataFrame combinado na aba 'resultados'
        df_resultados.to_excel(escritor, sheet_name='resultados', index=False)

        # Cria a aba 'gráficos'
        workbook = escritor.book
        worksheet = workbook.add_worksheet('gráficos')
        escritor.sheets['gráficos'] = worksheet

        # Lista de colunas disponíveis para plotagem
        colunas = df_resultados.columns.tolist()

        # Escreve opções para seleção
        worksheet.write('A1', 'Selecione o eixo X:')
        worksheet.write('A2', 'Selecione o eixo Y:')
        worksheet.write('A3', 'Selecione o eixo Z:')

        # Cria uma planilha oculta para armazenar as opções das colunas
        hidden_sheet = workbook.add_worksheet('hidden')
        hidden_sheet.hide()
        for idx, coluna in enumerate(colunas):
            hidden_sheet.write_string(idx, 0, coluna)  # Escreve na coluna A da planilha 'hidden'

        # Define a validação de dados para os menus suspensos usando as colunas da planilha 'hidden'
        faixa_colunas = f"'hidden'!$A$1:$A${len(colunas)}"
        for row in range(1, 4):
            worksheet.data_validation(f'B{row}', {
                'validate': 'list',
                'source': faixa_colunas
            })

        # Escreve as fórmulas para obter os dados das colunas selecionadas
        data_start_row = 5  # Linha inicial para os dados
        data_start_col = 0  # Coluna inicial para os dados

        # Escreve os cabeçalhos
        worksheet.write(data_start_row, data_start_col, 'Eixo X')
        worksheet.write(data_start_row, data_start_col + 1, 'Eixo Y')
        worksheet.write(data_start_row, data_start_col + 2, 'Eixo Z')

        num_dados = len(df_resultados)

        # Escreve as fórmulas para cada linha de dados
        for i in range(num_dados):
            row = data_start_row + 1 + i
            # Fórmula para o Eixo X
            formula_x = f"=INDEX(resultados!$A$2:$ZZ${num_dados + 1}, {i + 1}, MATCH($B$1, resultados!$A$1:$ZZ$1, 0))"
            # Fórmula para o Eixo Y
            formula_y = f"=INDEX(resultados!$A$2:$ZZ${num_dados + 1}, {i + 1}, MATCH($B$2, resultados!$A$1:$ZZ$1, 0))"
            # Fórmula para o Eixo Z
            formula_z = f"=INDEX(resultados!$A$2:$ZZ${num_dados + 1}, {i + 1}, MATCH($B$3, resultados!$A$1:$ZZ$1, 0))"

            worksheet.write_formula(row, data_start_col, formula_x)
            worksheet.write_formula(row, data_start_col + 1, formula_y)
            worksheet.write_formula(row, data_start_col + 2, formula_z)

        # Cria o gráfico 2D usando os dados calculados, representando o terceiro eixo pelo tamanho dos marcadores
        chart = workbook.add_chart({'type': 'scatter'})

        # Define os dados para o gráfico usando referências de células
        chart.add_series({
            'name': 'Dados Selecionados',
            'categories': [worksheet.name, data_start_row + 1, data_start_col, data_start_row + num_dados, data_start_col],
            'values':     [worksheet.name, data_start_row + 1, data_start_col + 1, data_start_row + num_dados, data_start_col + 1],
            'marker': {
                'type': 'circle',
                'size': 5,
                'border': {'color': 'black'},
                'fill':   {'color': '#FF9900'},
            },
            'points': [
                {'fill': {'color': f'#{int(255 - (i / num_dados) * 255):02X}FF00'}} for i in range(num_dados)
            ]
        })
        chart.set_title({'name': 'Gráfico 2D com 3 Eixos'})
        chart.set_x_axis({'name': '=gráficos!$B$1'})
        chart.set_y_axis({'name': '=gráficos!$B$2'})

        # Inserimos o gráfico na planilha
        worksheet.insert_chart('F5', chart)

    # Move os arquivos originais para a pasta 'Arquivos_brutos'
    for arquivo in arquivos_excel:
        nome_arquivo = os.path.basename(arquivo)
        destino = os.path.join(pasta_brutos, nome_arquivo)
        shutil.move(arquivo, destino)


#================================================== Preparação do algoritmo ===================================

def simulador(input_file: str, output_folder: str, output_file: str, nova_area_marcola, pulverization_speed, largura_do_dronezaum, num_motors, volume, paralelos_, M_bat_, cap_max_, E_bat_max_, serie_, A_):

    #================================================== Inicio do algoritmo do Marcus ===================================

    t0 = time.time()
    # DADOS DE ENTRADA ALGORITIMO DE ROTA
    # Definir o arquivo CSV de entrada
    csv_file = input_file

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

    n_motores = num_motors

    #volume_tanque = np.arange(10, 100.1, 5)        #[L]
    volume_tanque = volume
    #paralelos = np.arange(1, 80.1, 1)
    paralelos = paralelos_

    area_total = 1000               # [ha]
    AREA = nova_area_marcola # [ha]                                            # HADLER PUXAR VARIAVEL EXTERNA
    resultados = []
    it = 0

    E_bat_max = E_bat_max_
    M_bat = M_bat_
    cap_max = cap_max_
    serie = serie_

    #=========================== Área do Euler , XISPA [    não meche pufavo <:(   ja era, mexi >:) ] ================================


    # ==========================================================================================================

    for bb, M_pulv_max in enumerate(volume_tanque):
        #print("Tanque [L]: ",M_pulv_max,round(bb/(len(volume_tanque)-1)*100,2),"%")
        voo_vector = []
        dias = []
        tempo_manobra = []
        EOC_hr = []
        vol_conb_consumido = [] #[kg]pulverization_speed
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
        GW = []
        DIAMETRO = []
        BATERIA_MINIMA = []
        CALDA_RESTANTE = []
        
        barra = largura_do_dronezaum
        # print(largura_do_dronezaum)
        
        # BARRA
        m_barra = barra*10/6

        # MASSAS
        M_estrut_inicial = 0.55401*(M_pulv_max) + 5.39343 +  m_barra    #[kg]
        #A_inicial =  (0.0431*(M_pulv_max + M_estrut_inicial) + 0.7815)/n_motores
        
        

        #E_bat_max, M_bat, cap_max, serie = define_tesao(volume_tanque[bb], A_inicial, M_pulv_max + M_estrut_inicial, 1) #Ve a bateria 1P [ALTERAR PARA COLOCAR O P INICIAAAAALLLLLLLLLLLLLLLL]
        
        for aa in range(len(paralelos)):
            print("Volume do tanque: ", volume_tanque[bb])
            print("Paralelos: ", paralelos[aa])
            P_sensores = 0;
            P_LED = 100
            P_sist = 38.71
            P_bombas = 95.04
            # MASSAS
            M_estrut = 0.55401*(M_bat + M_pulv_max) + 5.39343 + m_barra      #[kg]
            #A =  (0.0431*(M_pulv_max + M_estrut) + 0.7815)/n_motores            # incluir massa bat
            M_tot_in = M_bat + M_pulv_max + M_estrut
            
            #bateria
            #E_bat_max, M_bat, cap_max, serie = define_tesao(volume_tanque[bb], A, M_tot_in, paralelos[aa])
            M_estrut = 0.55401*(M_bat + M_pulv_max) + 5.39343 + m_barra      #[kg]
            M_tot_in = M_bat + M_pulv_max + M_estrut
            
            
            ## PROPULSAO
            eta_escmotor = 0.848                                                           # Eficiência ESC e Motor
            eta_helice = 0.7719                                                            # Eficiência hélice
            rho = 1.225
            cnst = 1/(eta_escmotor*eta_helice)
            
            A =  A_
            Diametro = (4*A/np.pi)**0.5/0.0254   
            T_motor = M_tot_in/8
            ef = (1000/9.81/(cnst*(np.sqrt(T_motor*9.81/(2*rho*A)))))
            
            # PULVERIZAÇÃO
            v_pulv = pulverization_speed                              #[m/s]
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
        
            calda_passeio = 0
                        
            while OP[i] != "FIM\t" and flag =="off":
                #print(OP[i])
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
                COAXIAL_80 = 1.397542375147
                
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
                    
                elif n_motores == 4:
                    
                    T_M1.append(T_hover[i] - 0.1548 * v[i] + 0.0423 * w[i] + 0.8315 * vz[i])
                    if T_M1[i] < 0.05*T_hover[i]:
                        T_M1[i] = 0.05*T_hover[i]
                    ef_M1.append((1/(cnst * np.sqrt(T_M1[i]*9.81/(2*rho*A) ) )*1000/9.81))
                    Preq_M1.append(1000 * T_M1[i]/ef_M1[i])
                    
                    T_M2.append(T_hover[i] - 0.1548 * v[i] - 0.0423 * w[i] + 0.8315 * vz[i])
                    if T_M2[i] < 0.05*T_hover[i]:
                        T_M2[i] = 0.05*T_hover[i]
                    ef_M2.append((1/(cnst * np.sqrt(T_M2[i]*9.81/(2*rho*A) ) )*1000/9.81))
                    Preq_M2.append(1000 * T_M2[i]/ef_M2[i])
                    
                    T_M3.append(T_hover[i] + 0.1548 * v[i] + 0.0423 * w[i] + 0.8315 * vz[i])
                    if T_M3[i] < 0.05*T_hover[i]:
                        T_M3[i] = 0.05*T_hover[i]
                    ef_M3.append((1/(cnst * np.sqrt(T_M3[i]*9.81/(2*rho*A) ) )*1000/9.81))
                    Preq_M3.append(1000 * T_M3[i]/ef_M3[i])
                    
                    T_M4.append(T_hover[i] + 0.1548 * v[i] - 0.0423 * w[i] + 0.8315 * vz[i])
                    if T_M4[i] < 0.05*T_hover[i]:
                        T_M4[i] = 0.05*T_hover[i]
                    ef_M4.append((1/(cnst * np.sqrt(T_M4[i]*9.81/(2*rho*A) ) )*1000/9.81))
                    Preq_M4.append(1000 * T_M4[i]/ef_M4[i])
                    
                    Preq_prop.append(Preq_M1[i] +  Preq_M2[i] + Preq_M3[i] + Preq_M4[i])
                    
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
                
                #print(OP[i], Preq_tot[i])
                E_bat.append(E_bat[i] - Preq_tot[i]*dt/3600)        #[Wh]
                #print(OP[i], E_bat[i])
               
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
                #print(f"Operacao = {OP[i]} Status = {STATUS[i]} VDESLOC * AUTONOMIA: {v_desloc*autonomia[i]}  dist_rtl *1.2 = {1.2*dist_rtl[i+1]}")
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
                        calda_passeio = calda_passeio + M_pulv[i+1]                   # CALDA QUE NAO FOI PULVERIZADA NO VOO
                        t[i+1] = t[i+1] +  (M_pulv_max - M_pulv[i+1])*60/50+20
                        M_pulv[i+1] = M_pulv_max
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
                        
                elif(OP[i] == "RTL-BAT\t" or (v_desloc*autonomia[i] <= 1.2*dist_rtl[i+1] and STATUS[i] != "YAW+" and STATUS[i] !="YAW-")):
                    if(OP[i] == "RTW\t"):
                        OP.append("FIM\t")
                        STATUS.append("INCOMLPLETO")
                        print(i,"->>NAO CHEGOOOUUU")
                    
                    if (x[i+1] == 0 and y[i+1] == 0 and z[i+1] == 0):
                        OP.append("RTW\t")
                        #print(round(M_pulv_max,3),round(M_comb_max,3),round(M_comb[i],3),round(M_pulv[i],3))
                        E_bat[i+1] = E_bat_max
                        calda_passeio = calda_passeio + M_pulv[i+1]                     # CALDA QUE NAO FOI PULVERIZADA NO VOO
                        t[i+1] = t[i+1] +  (M_pulv_max - M_pulv[i+1])*60/50+20
                        M_pulv[i+1] = M_pulv_max
                        n_abs = n_abs + 1
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
                    elif v_desloc*autonomia[i] <= 100.00:
                        OP.append("FIM\t")
                        STATUS.append("INCOMLPLETO")
                        print(i,"->>NAO CHEGOOOUUU")
                    else:
                        OP.append(OP[i])
                    # if j > 1000:
                    #     flag = "on"

                else:
                    OP.append(OP[i])
                
                if (OP[i] == "RTL-BAT\t" or OP[i] == "RTL-CALDA\t" or OP[i] == "RTL-FIM\t" or OP[i] == "RTW\t"):
                    Tempo_rtl = Tempo_rtl + dt
                
                
                    
                #RESULTADOS.append(f'{OP[i]}\t{STATUS[i]}\t{i:.1f}\t{j:.1f}\t{x[i]:.1f}\t{y[i]:.1f}\t{theta[i]:.1f}\t{M_pulv[i]:.1f}\t{E_bat[i]:.1f}\t{Preq_tot[i]:.1f}\t{v[i]:.1f}\t{w[i]:.1f}\t{vz[i]:.1f}')
                if i == 1000000:
                    flag = "on"
                    print('SOCORRO AAAAAAAAAAAAAAAA ')
                else:
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
            print(STATUS[i-1], OP[i])
            operacao.append(STATUS[i])
            
            #(preco_por_ha_real,preco_drone_bateria) = custos(M_pulv_max,cap_max,area_total,t_de_voo[i]/3600,voo,math.ceil(area_total/(X**2/10**4)),X**2/10**4)
            #capex = produtividade/preco_drone_bateria
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
            GW.append(ef)
            DIAMETRO.append(Diametro)
            CALDA_RESTANTE.append(calda_passeio)
            BATERIA_MINIMA.append(min(E_bat))
            
            
            if n_abs == 0:
                print("t2[h]: ",round(tempo_missao[len(tempo_missao)-1],4) ,"// t1[h]: ",round(tempo_missao[len(tempo_missao)-2],4) )
                break
            
            
        #======================= PLOTS =======================================#
        produtividade = [AREA/x for x in tempo_missao]
        
        data = {
            # "Faixa [m]": faixa_vector,
            "Volume de calda [L]": [round(x, 2) for x in TANQUES], 
            "STATUS": operacao,
            "Capacidade operacional [ha/h]": [round(x, 4) for x in produtividade],
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
            "g/w": GW,
            "Diametro": DIAMETRO,
            "CALDA PASSEADA [L]": CALDA_RESTANTE,
            "Bat_min [Wh]": BATERIA_MINIMA
        }
        
        vol = []
        df1 = pd.DataFrame(data)
        resultados.append(df1)
        it = it + 1
        

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
        resultado_file = os.path.join(output_folder, f"RESULTADOS_{k}.xlsx")
        resultados[k].to_excel(resultado_file)

    unificar_planilhas(output_folder)
    print(f"Arquivos salvos e unificados em {output_folder}")


target_mask = lambda sba, p, w, i, m: f'field_{sba}ha_100ha_{p}%_{w}m_{i}_{m}'




#   Drone T20P - Bateria T20P          ------------------------------------------------------------------
#num_motor = 4
#volume_tanque = np.arange(20,20.1,1)        #[L]
#paralelos = np.arange(1,1.1,1)
#M_bat = 6
#cap_max = 13*0.7
#E_bat_max = cap_max*3.7*2
#serie = 1
#K = 1.477559
#A =  K/num_motor
#
#subareas = [16, 36, 64, 100]
#percentages = [2, 6, 10, 14, 18]
#width = [0, 6, 12]
#indexes = [0]
#modes = ['TSP', 'LM']
#speeds = [0.5, 1.0, 2.0, 4.0]
#
#input_file = f'rotas-condicionadas/field_16ha_100ha_2%_0m_0_TSP/mission.csv'
#
#output_folder = f'resultados-eletrico/vel{int(0.5*10):02}/TESTE/'
#os.makedirs(output_folder, exist_ok=True)
#
#simulador(input_file, output_folder, 'mission.csv', 16, 0.5, 0, num_motor, volume_tanque, paralelos, M_bat, cap_max, E_bat_max, serie, A)