import os
import numpy as np
import math
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pandas as pd
import plotly.io as pio
import time

# Definindo o renderizador padrão do Plotly
pio.renderers.default = 'browser'

# Definir o caminho das pastas de dados e resultados
dados_path = os.path.join(os.getcwd(), "Dados")
resultados_path = os.path.join(os.getcwd(), "Resultados")

# Verificar se as pastas existem, caso contrário, criar
if not os.path.exists(dados_path):
    os.makedirs(dados_path)

if not os.path.exists(resultados_path):
    os.makedirs(resultados_path)

# Iniciar temporizador para o cálculo de desempenho
t0 = time.time()

# Carregar os dados de entrada do arquivo CSV na pasta 'Dados'
csv_file = os.path.join(dados_path, "dados-simulador.csv")
df = pd.read_csv(csv_file, delimiter=';')

# Extração de dados do CSV
xp = df['x[m]'].values
yp = df['y[m]'].values
a_tot = df['area.total[m2]'].values
a_mato = df['area.mato[m2]'].values
delta_theta = df['ang.proa[deg]'].values
theta_abs = df['ang.abs[deg]'].values
dist_vector = df['dist[m]'].values
status = df['operacao'].values

# Definir parâmetros iniciais
n_motores = 8
faixa = 5
faixa_min = faixa
faixa_max = 5
delta_pulv = 1
volume_tanque = np.linspace(10, 60, 6)
combs_vetor = np.linspace(1, 10, 10)

# Inicialização de matrizes para armazenar resultados de produtividade e custos
produtividade_matriz = np.zeros((len(combs_vetor), len(volume_tanque)))
capex_matriz = np.zeros((len(combs_vetor), len(volume_tanque)))

# Definir área total da missão em hectares
area_total = 1000  # [ha]
resultados = []
it = 0

# Loop principal para variação dos vetores de volume do tanque e combustível
for aa in range(len(combs_vetor)):
    
    # Inicialização de variáveis dentro do loop para cada combinação
    talhao_maximus = []
    voo_vector = []
    dias = []
    tempo_manobra = []
    EOC_hr = []
    vol_conb_consumido = []
    tempo_missao = []
    RESULTADOS = []
    RESULTADOS.append(f'OP\t\tSTATUS\ti\tj\tx\ty\ttheta\tM_pulv\tM_comb\tPreq_tot\tv\tw\tvz')
    RTLS = []
    
    # Mais variáveis para armazenar dados ao longo do loop
    operacao = []
    TANQUES = []
    tempo_idas_vindas = []
    Tempo_rtl = 0
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
    capex = np.zeros(len(volume_tanque))
    drone_e_bateria = []
    abastecimentos = []
    preco_drone_bateria = np.zeros(len(volume_tanque))
    CALDA_CONS = []
    preco_por_ha_real = np.zeros(len(volume_tanque))
    tanque = 0
    AUTONOMIA_PROJ = []
    MASSA_ESTRUTURA = []
    COMBUSTIVEL = []
    GERADOR = []
    VPULV = []
    VDESLOC = []
    PGERADOR = []

    # Loop sobre os volumes do tanque
    for bb, M_pulv_max in enumerate(volume_tanque):
        
        # Exibe o progresso da simulação
        print(f'{round(bb / (len(volume_tanque) - 1) * 100, 2)} %')

        # Definir parâmetros de pulverização e deslocamento
        v_pulv = 1.0  # Velocidade de pulverização [m/s]
        v_desloc = 10.0  # Velocidade de deslocamento [m/s]
        v_subida = 2.5  # Velocidade de subida [m/s]
        v_descida = -v_subida  # Velocidade de descida [m/s]
        omega = 40.0  # Taxa de rotação [graus/s]

        # Cálculo da massa estrutural e massas iniciais
        M_estrut = (18.3611161042012 * np.log(M_pulv_max) - 30.178770579692)  # Massa estrutural [kg]
        M_comb_max = combs_vetor[aa]  # Massa de combustível [kg]
        M_tot_in = M_comb_max + M_pulv_max + M_estrut  # Massa total inicial

        ganho_cons = 0.2  # Ganho no consumo
        tensao_max = 57.45  # Tensão máxima do sistema [V]
        P_sensores = 0  # Potência dos sensores [W]
        P_LED = 100  # Potência dos LEDs [W]
        P_sist = 38.71  # Potência do sistema [W]
        P_bombas = 95.04  # Potência das bombas [W]

        dt = 1  # Passo de tempo [s]

        # Tempos de preparação, deslocamento e lavagem
        t_prep = 300  # Tempo de preparação [s]
        t_abs_calda = M_pulv_max * 60 / 50 + 20  # Tempo de abastecimento de calda [s]
        t_abs_comb = 40  # Tempo de abastecimento de combustível [s]
        t_desloc_pre_op = 2520  # Tempo de deslocamento antes da operação [s]
        t_desloc_pos_op = t_desloc_pre_op  # Tempo de deslocamento após a operação [s]
        t_triplice_lavagem = 6 * M_pulv_max / (6 * 2) * 60  # Tempo de lavagem tripla [s]
        t_lavagem_limpeza = (2 * (0.7 * M_pulv_max / (6 * 2)) + 5) * 60  # Tempo de lavagem e limpeza [s]

        # Inicialização de variáveis para o loop de simulação
        rtl_acumulado = 0
        n_abs = 0
        voo = 1
        cons_pulv = []
        M_tot = []
        M_pulv = [M_pulv_max]
        M_comb = [M_comb_max]
        comb_cons = [0]

        # Vetores de tempo e consumo
        t = [t_prep + t_desloc_pre_op]
        t_pulv = [0.0]
        t_manobra = [0.0]
        t_de_voo = [0]
        calda_cons = [0]

        # Variáveis de posicionamento
        x = [0.0]
        y = [0.0]
        z = [0.0]
        theta = [0.0]
        dist_percorr = [0]
        dist_pulv = [0]

        # Cálculo da potência e da massa do gerador
        zi = 5
        a = 0
        M_tot_2 = 0.0

        # Eficiências do sistema de propulsão
        eta_escmotor = 0.848  # Eficiência do ESC e Motor
        eta_helice = 0.7719  # Eficiência da hélice
        rho = 1.225  # Densidade do ar [kg/m³]
        cnst = 1 / (eta_escmotor * eta_helice)

        # Loop de simulação de voo até atingir as condições de fim
        while abs(M_tot_in - M_tot_2) > 10**-3:
            if a > 0:
                M_tot_in = M_tot_2

            T_motor = M_tot_in / n_motores
            P_gerador_max = 1000 * T_motor / (0.001768 * (60)**2 - 0.34614 * 60 + 22.511599) * n_motores + P_LED + P_sist + P_bombas + P_sensores
            ef = 0.001768 * (60)**2 - 0.34614 * 60 + 22.511599
            M_gerador = ((12.895 * (P_gerador_max / (0.8 * 0.9 * 0.98 * 745.7))**2 + 60.062 * P_gerador_max / (0.8 * 0.9 * 0.98 * 745.7) + 1069.5) / 1000 + (0.0819 * P_gerador_max / (0.9 * 0.98 * 0.8) + 200.25) / 1000) * 1.15
            M_tot_2 = M_estrut + M_comb_max + M_pulv_max + M_gerador

            a += 1

        # Mais cálculos de massa e potência para as fases do voo

        # Continue implementando os cálculos como necessário e adicione comentários relevantes para cada bloco de código.

    # ======================= Finalizando e salvando os resultados ========================= #

    # Salvar resultados após o término da simulação
    for k in range(len(resultados)):
        resultado_file = os.path.join(resultados_path, f"RESULTADOS_{k}.xlsx")
        resultados[k].to_excel(resultado_file)

    # Salvar o gráfico 3D na pasta 'Resultados'
    saida_html = os.path.join(resultados_path, "saida_sem_discreziar.html")
    fig_trajeto.write_html(saida_html)
