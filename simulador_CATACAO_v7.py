import numpy as np
import math
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pandas as pd
import plotly.io as pio
# from plotly.offline import plot
pio.renderers.default='browser'
# from scipy.interpolate import griddata
#from CUSTOS import custos
import time 
#sou lindo
import os
from ipywidgets import interact, fixed
import ipywidgets as widgets
from scipy.interpolate import griddata
from IPython.display import display  # Import display function

def compila(df):
    # Corrigir separadores decimais e converter para numérico
    numeric_columns = df.select_dtypes(include=['object', 'string']).columns.tolist()
    for col in numeric_columns:
        # Tenta converter strings numéricas com vírgulas em floats
        df[col] = df[col].str.replace(',', '.').astype(float)
    
    # Obter a lista de colunas numéricas
    columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Exibir a quantidade de dados por variável
    print("\nQuantidade de pontos de dados por variável:")
    for col in columns:
        data = df[col]
        num_unique = data.nunique()
        num_total = data.count()
        dtype = data.dtype
        print(f"{col}: {num_total} pontos, {num_unique} únicos, Tipo: {dtype}")
    
    if len(columns) < 3:
        print("\nNão há variáveis numéricas suficientes para plotar.")
        return
    
    print("\nVariáveis disponíveis para plotagem:")
    for idx, col in enumerate(columns):
        print(f"{idx}: {col}")
    
    # Opção de escolher o tipo de gráfico
    print("\nEscolha o tipo de gráfico:")
    print("1: Superfície Interpolada")
    print("2: Gráfico de Dispersão 3D")
    plot_type = input("Digite 1 ou 2: ").strip()
    
    if plot_type not in ['1', '2']:
        print("Opção inválida.")
        return
    
    # Obter a entrada do usuário para as variáveis
    x_var_idx = int(input("\nSelecione o índice para a variável do eixo X: "))
    y_var_idx = int(input("Selecione o índice para a variável do eixo Y: "))
    z_var_idx = int(input("Selecione o índice para a variável do eixo Z: "))
    
    x_var = columns[x_var_idx]
    y_var = columns[y_var_idx]
    z_var = columns[z_var_idx]
    
    if plot_type == '1':
        # Solicitar variável de controle para o slider
        slider_var_idx = int(input("Selecione o índice para a variável de controle (slider): "))
        slider_var = columns[slider_var_idx]
        unique_slider_values = np.sort(df[slider_var].unique())
        
        # Criar uma grade de valores x e y
        x_lin = np.linspace(df[x_var].min(), df[x_var].max(), 50)
        y_lin = np.linspace(df[y_var].min(), df[y_var].max(), 50)
        X_grid, Y_grid = np.meshgrid(x_lin, y_lin)
        
        # Inicializar a figura
        fig = go.Figure()
        
        # Criar quadros para cada valor da variável de controle
        frames = []
        for value in unique_slider_values:
            data_subset = df[df[slider_var] == value]
            if len(data_subset) < 3:
                print(f"Pulando o valor {value} do slider devido a pontos de dados insuficientes ({len(data_subset)}).")
                continue
            
            # Interpolar valores z na grade
            try:
                Z_grid = griddata(
                    (data_subset[x_var], data_subset[y_var]),
                    data_subset[z_var],
                    (X_grid, Y_grid),
                    method='linear'
                )
                
                # Verificar se Z_grid contém todos NaNs
                if np.isnan(Z_grid).all():
                    print(f"Pulando o valor {value} do slider porque a interpolação resultou em todos NaNs.")
                    continue
                
                # Criar a superfície
                surface = go.Surface(
                    x=X_grid,
                    y=Y_grid,
                    z=Z_grid,
                    colorscale='Viridis',
                    cmin=df[z_var].min(),
                    cmax=df[z_var].max(),
                    showscale=False,
                    name=str(value)
                )
                
                frames.append(go.Frame(data=[surface], name=str(value)))
            
            except Exception as e:
                print(f"Pulando o valor {value} do slider devido ao erro de interpolação: {e}")
                continue
        
        if not frames:
            print("Nenhum quadro foi criado. Verifique se as variáveis selecionadas possuem dados suficientes.")
            return
        
        # Dados iniciais
        initial_frame = frames[0]
        
        # Layout com slider
        sliders = [dict(
            steps=[dict(method='animate',
                        args=[[frame.name], dict(mode='immediate',
                                                 frame=dict(duration=500, redraw=True),
                                                 transition=dict(duration=0))],
                        label=frame.name) for frame in frames],
            transition=dict(duration=0),
            x=0,
            y=0,
            currentvalue=dict(font=dict(size=12), prefix=slider_var + ': ', visible=True, xanchor='center'),
            len=1.0
        )]
        
        layout = go.Layout(
            title='Gráfico Interativo de Superfície 3D',
            scene=dict(
                xaxis=dict(title=x_var),
                yaxis=dict(title=y_var),
                zaxis=dict(title=z_var)
            ),
            sliders=sliders,
            updatemenus=[dict(
                type='buttons',
                showactive=False,
                y=1,
                x=1.05,
                xanchor='left',
                yanchor='bottom',
                buttons=[dict(label='Play',
                              method='animate',
                              args=[None, dict(frame=dict(duration=500, redraw=True),
                                               transition=dict(duration=0),
                                               fromcurrent=True,
                                               mode='immediate')])]
            )]
        )
        
        fig = go.Figure(data=initial_frame.data, frames=frames, layout=layout)
        
    elif plot_type == '2':
        # Criar o gráfico de dispersão 3D
        fig = go.Figure(data=[go.Scatter3d(
            x=df[x_var],
            y=df[y_var],
            z=df[z_var],
            mode='markers',
            marker=dict(
                size=5,
                color='blue'
            )
        )])
        
        fig.update_layout(
            title='Gráfico de Dispersão 3D',
            scene=dict(
                xaxis=dict(title=x_var),
                yaxis=dict(title=y_var),
                zaxis=dict(title=z_var)
            )
        )
    
    # Exibir o gráfico no navegador
    fig.show(renderer="browser")
    
    # Opcionalmente, salvar o gráfico em um arquivo HTML
    save_option = input("Deseja salvar o gráfico como um arquivo HTML? (sim/não): ").strip().lower()
    if save_option == 'sim':
        filename = input("Digite o nome do arquivo (com extensão .html): ").strip()
        fig.write_html(filename)
        print(f"Gráfico salvo como {filename}")

# Definir o caminho das pastas
dados_path = os.path.join(os.getcwd(), "Dado0s")
resultados_path = os.path.join(os.getcwd(), "Resultados")

# Verificar se as pastas existem, caso contrário, criar
if not os.path.exists(dados_path):
    os.makedirs(dados_path)

if not os.path.exists(resultados_path):
    os.makedirs(resultados_path)

t0 = time.time()
# DADOS DE ENTRADA ALGORITIMO DE ROTA
# Definir o arquivo CSV de entrada
csv_file = os.path.join(dados_path, "dados-simulador.csv")

# Carregar o arquivo CSV
df = pd.read_csv(csv_file, delimiter=';')
xp = df['x[m]'].values
yp = df['y[m]'].values
a_tot = df['area.total[m2]'].values
a_mato = df['area.mato[m2]'].values
delta_theta = df['ang.proa[deg]'].values
theta_abs = df['ang.abs[deg]'].values
dist_vector = df['dist[m]'].values
#dist_rtl = df['rtl[m]'].values
status = df['operacao'].values

n_motores = 8
faixa = 5; faixa_min = faixa; faixa_max = 5
#M_pulv_max = 40; M_pulv_min = M_pulv_max; M_pulv_lim = 40
delta_pulv = 1
#faixas = np.arange(faixa_min,faixa_max+0.1,1)

volume_tanque = np.arange(10,400.1,5)
#combs_vetor = np.linspace(1,1,1)


#produtividade_matriz = np.zeros((len(combs_vetor),len(volume_tanque)))
#capex_matriz = np.zeros((len(combs_vetor),len(volume_tanque)))
area_total = 1000               # [ha]
resultados = []
it = 0

for bb,M_pulv_max in enumerate(volume_tanque):
    print("Tanque [L]: ",M_pulv_max,round(bb/(len(volume_tanque)-1)*100,2),"%")
    M_comb_max = 1
    dcomb = 0.25
    #M_pulv_max = M_pulv_min 
    talhao_maximus = []
    voo_vector = []
    dias = []
    tempo_manobra = []
    EOC_hr = []
    vol_conb_consumido = [] #[kg]
    tempo_missao = []   #[s]
    
    RESULTADOS = []
    RESULTADOS.append(f'OP\t\tSTATUS\ti\tj\tx\ty\ttheta\tM_pulv\tM_comb\tPreq_tot\tv\tw\tvz')
    RTLS = []
    
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
    #for aa in range(len(combs_vetor)):
    while(M_comb_max <= 500):
        print("Comb [kg]: ",M_comb_max)
        #print(round(bb/(len(volume_tanque)-1)*100,2),"%")
        # PULVERIZAÇÃO
        #Taxa = 10.0                               #[L/ha]      
        v_pulv = 1.0                              #[m/s]
        #vazao = Taxa/10000 * (v_pulv*60*faixa)    #[L/min]
        v_desloc = 10.0                           #[m/s]
        v_subida = 2.5  
        v_descida = -v_subida
        omega = 40.0                              #[graus/s]
        
        # MASSAS
        M_estrut = (18.3611161042012*np.log(M_pulv_max) - 30.178770579692)       #[kg]
        # M_comb_max = (2.5 + (M_pulv_max-10)*0.2375)*0.715
        #M_comb_max = combs_vetor[aa]
        M_tot_in = M_comb_max + M_pulv_max + M_estrut
        
        ganho_cons = 0.2
        tensao_max = 57.45
        P_sensores = 0;
        P_LED = 100
        P_sist = 38.71
        P_bombas = 95.04
       
        dt = 1                  #[s]
        
        t_prep = 300; t_abs_calda = M_pulv_max*60/50+20; t_abs_comb = 40; t_desloc_pre_op = 2520  
        t_desloc_pos_op = t_desloc_pre_op ;
        t_triplice_lavagem = 6*M_pulv_max/(6*2)*60;
        t_lavagem_limpeza = (2*(0.7*M_pulv_max/(6*2))+5)*60
        
        rtl_acumulado = 0
        
        n_abs = 0
        n = 1
        voo = 1                        
        cons_pulv = []
        M_tot = []
        M_pulv = []; M_pulv.append(M_pulv_max)
        M_comb = []; M_comb.append(M_comb_max)
        cons_gas = []
        comb_cons = []; comb_cons.append(0)
        
        # TEMPOS
        t = []; t.append(t_prep + t_desloc_pre_op)
        t_pulv = []; t_pulv.append(0.0)
        t_manobra = []; t_manobra.append(0.0)
        t_de_voo = []; t_de_voo.append(0)
        calda_cons = []; calda_cons.append(0)
        
        # TALHÃO
        #x0 = 20.0
        #Z = zi
        theta_dir = 0.0

        OP = []
        STATUS = []
        v = []
        vz = []
        w = []
        dist_percorr = []; dist_percorr.append(0)
        dist_pulv = []; dist_pulv.append(0)
        
        x = []; x_rtl = 0
        y = []; y_rtl = 0
        z = []; z_rtl = 0
        theta = []; theta_rtl = 0
        alpha = 0
        
        x.append(0.0)
        y.append(0.0)
        z.append(0.0)
        theta.append(0.0)
        #thetai = math.atan(xi/yi)*180/math.pi
        autonomia = [];
        dist_rtl = [];
        produtiv_por_voo = []
        
        # CALCULO POTENCIA E MASSA DO GERADOR
        zi = 5
        a = 0
        
        
        M_tot_2 = 0.0
    
        ## PROPULSAO
        eta_escmotor = 0.848                                                           # Eficiência ESC e Motor
        eta_helice = 0.7719                                                            # Eficiência hélice
        rho = 1.225
        cnst = 1/(eta_escmotor*eta_helice)
        
        dist_rtl.append(0)
       
       
       # T_motor = M_tot_in/8
       # ef = (1000/9.81/(cnst*(np.sqrt(T_motor*9.81/(2*rho*A)))))
       
      
        
        while (abs(M_tot_in - M_tot_2) > 10**-3):
            if a > 0:
                M_tot_in = M_tot_2
                
            T_motor = M_tot_in/8
            #P_gerador_max = 8*(1000 * T_motor/ef)
            
            # if M_pulv_max < 70:
            P_gerador_max = 1000*T_motor/(0.001768*(60)**2 - 0.34614*60 + 22.511599)*8 + P_LED + P_sist + P_bombas + P_sensores
            ef = 0.001768*(60)**2 - 0.34614*60 + 22.511599
            # else:
            #     ef = 1095.03924*(T_motor*1000)**-0.48364
            #     P_gerador_max = 8*(1000 * T_motor/ef)
            
            #E_bat_max = P_gerador_max * (1.5 * (X) * math.sqrt(2))/(v_desloc*3600)
            M_gerador = ((12.895*(P_gerador_max/(0.8*0.9*0.98*745.7))**2 + 60.062*P_gerador_max/(0.8*0.9*0.98*745.7)+1069.5)/1000+(0.0819*P_gerador_max/(0.9*0.98*0.8)+ 200.25)/1000)*1.15 
            #M_bat = 0.000386020698965*(1000*E_bat_max/52.22) + 0.541952902354882 
            #M_estrut = 0.554*(M_comb_max + M_pulv_max + M_gerador) + 5.3934
            M_tot_2 = M_estrut + M_comb_max + M_pulv_max + M_gerador #+ M_bat
    
            #print("Ef: ",round(ef,2),"// Mtot= ",round(M_tot_in,2),"//P_gerador = ", round(P_gerador_max,2),"//M_gerador = ",round(M_gerador,2),"//M_estrutura = ",round(M_estrut,2),"//M_tot2 = ",round(M_tot_2,2))
            a = a + 1
            
        A =  (0.0431*M_tot_in + 0.7815)/n_motores
        Diametro = (4*A/np.pi)**0.5/0.0254    
        
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
                    
                
            if OP[i] == "RTL-CALDA\t" or OP[i] == "RTL-COMB\t" or OP[i] == "RTL-FIM\t":
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
            
            # cons_gas.append((((0.009424 * P_gerador[i] + 0 )*dt/60)*(1-ganho_cons))/1000)       # ENTRA FIT SFC GERADORES
            
            SFC = 3065.6*M_tot[i]**-0.402
            cons_gas.append(((SFC * P_gerador[i])*dt)/(3600*1000*1000))      # consumo em [kg/s], potencia em [W], SFC em [g/kW.h]

            comb_cons.append(comb_cons[i] + cons_gas[i])
            
            
            if M_comb[i] - cons_gas[i] < 0:
                M_comb.append(0)
            else:
                M_comb.append(M_comb[i] - cons_gas[i])
            if M_pulv[i] - cons_pulv[i] < 0:
                M_pulv.append(0)
            else:
                M_pulv.append(M_pulv[i] - cons_pulv[i])
                
            M_tot.append(M_estrut + M_comb[i+1] + M_pulv[i+1] + M_gerador) #+ M_bat)
            if cons_gas[i] == 0:
                autonomia.append(99999)
            else:
                autonomia.append(M_comb[i+1]/cons_gas[i])
                
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
                    if voo == 1:
                        produtiv_por_voo.append(dist_pulv[i+1]*faixa)
                    else:
                        produtiv_por_voo.append(dist_pulv[i+1]*faixa - produtiv_por_voo[len(produtiv_por_voo)-1])
                    voo = voo + 1
                    t[i+1] = t[i+1] + t_lavagem_limpeza + t_triplice_lavagem + t_desloc_pos_op
                else:
                    OP.append("RTL-FIM\t")
            elif(abs(x[i+1] - xp[j+1]) < 10**-6 and abs(y[i+1] - yp[j+1]) < 10**-6 and j == (len(dist_vector) -2)):
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
                    t_abs = 30
                    M_pulv[i+1] = M_pulv_max
                    theta[i+1] = -alpha - 90
                    if voo == 1:
                        produtiv_por_voo.append(dist_pulv[i+1]*faixa)
                    else:
                        produtiv_por_voo.append(dist_pulv[i+1]*faixa - produtiv_por_voo[len(produtiv_por_voo)-1])
                    voo = voo + 1
                    M_comb[i+1] = M_comb_max
                    #n_abs = n_abs + 1 
                    t[i+1] = t[i+1] + max(t_abs_calda,t_abs_comb)
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
                    
            elif(OP[i] == "RTL-COMB\t" or (v_desloc*autonomia[i] <= 1.2*dist_rtl[i+1] and STATUS[i] != "YAW+" and STATUS[i] !="YAW-")):
                if(OP[i] == "RTW\t"):
                    OP.append("FIM\t")
                    STATUS.append("INCOMLPLETO")
                    print(i,"->>RTL-COMB\t")
                
                if (x[i+1] == 0 and y[i+1] == 0 and z[i+1] == 0):
                    OP.append("RTW\t")
                    #print(round(M_pulv_max,3),round(M_comb_max,3),round(M_comb[i],3),round(M_pulv[i],3))
                    t_abs = 30
                    M_comb[i+1] = M_comb_max
                    M_pulv[i+1] = M_pulv_max
                    n_abs = n_abs + 1
                    t[i+1] = t[i+1] + max(t_abs_calda,t_abs_comb)
                    theta[i+1] = -alpha - 90
                    #print(theta[i+1])
                    if voo == 1:
                        produtiv_por_voo.append(dist_pulv[i+1]*faixa)
                    else:
                        produtiv_por_voo.append(dist_pulv[i+1]*faixa - produtiv_por_voo[len(produtiv_por_voo)-1])
                    voo = voo + 1
                    if theta[i+1] < 0:
                        theta[i+1] = theta[i+1] + 180   
                elif (OP[i] != "RTL-COMB\t"):
                    theta_rtl = theta[i+1]
                    alpha = math.atan2(x[i+1],y[i+1])*180/math.pi
                    x_rtl = x[i+1]
                    y_rtl = y[i+1]
                    z_rtl = z[i+1]
                    OP.append("RTL-COMB\t")
                    rtl_acumulado = rtl_acumulado + math.sqrt(x_rtl**2 + y_rtl**2)
                    #print(i,j)
                else:
                    OP.append("RTL-COMB\t")
                    #print(i,j)
            elif(OP[i]=="DESLOCANDO"):
                if status[j] == "p":
                    if (abs(x[i+1] -  xp[j+1]) < 1e-6 and abs(y[i+1] - yp[j+1]) < 1e-6 and z[i+1] == zi):
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
            
            if (OP[i] == "RTL-COMB\t" or OP[i] == "RTL-CALDA\t" or OP[i] == "RTL-FIM\t" or OP[i] == "RTW\t"):
                Tempo_rtl = Tempo_rtl + dt
            
            
                
            RESULTADOS.append(f'{OP[i]}\t{STATUS[i]}\t{i:.1f}\t{j:.1f}\t{x[i]:.1f}\t{y[i]:.1f}\t{theta[i]:.1f}\t{M_pulv[i]:.1f}\t{M_comb[i]:.1f}\t{Preq_tot[i]:.1f}\t{v[i]:.1f}\t{w[i]:.1f}\t{vz[i]:.1f}')
            # if i == 41322:
            #     flag = "on"
            # else:
            i = i + 1
        print(f"{time.time()-t0:.2f}s")
        
        # plt.plot(x, y)
        # # Adicionar título e rótulos aos eixos
        # plt.title('Volume de calda [L]:' + str(M_pulv_max))  # Título do gráfico
            
            
#---------------- VETORES PARA RESULTADOS ---------------------------------#
        # faixa_vector.append(faixa)
        tempo_missao.append(max(t) / 3600)  # Removido o arredondamento
        # dias.append(math.ceil(area_total/(X**2/10**4)))
        MTOW.append(M_tot_in)  # Removido o arredondamento
        # produtividade.append(X**2/10**4/(max(t)/3600))
        # talhao_maximus.append(X**2/10**4)
        voo_vector.append(voo)
        # area_por_voo.append(X**2/10**4/voo)
        vol_comb.append(comb_cons[i] / 0.715)  # Removido o arredondamento
        # vazao_bombas.append(vazao)
        dist_percorrida.append(dist_percorr[i] / 1000)  # Removido o arredondamento
        dist_pulverizando.append(dist_pulv[i] / 1000)  # Removido o arredondamento
        EOC_km.append(dist_pulv[i] / dist_percorr[i])  # Removido o arredondamento
        EOC_hr.append(t_pulv[i] / t_de_voo[i])  # Removido o arredondamento
        abastecimentos.append(n_abs)
        # capacidade_vector.append(cap_bat)
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
        
        AUTONOMIA_PROJ.append(M_comb_max / ((((0.009424 * P_gerador_max) * 1 / 60) * (1 - ganho_cons)) / 1000) / 3600)  # Removido o arredondamento
        MASSA_ESTRUTURA.append(M_estrut)  # Removido o arredondamento
        COMBUSTIVEL.append(M_comb_max)  # Removido o arredondamento
        GERADOR.append(M_gerador)  # Removido o arredondamento
        PGERADOR.append(P_gerador_max)  # Removido o arredondamento
        VPULV.append(v_pulv)
        VDESLOC.append(v_desloc)
        
        
        if n_abs == 0:
            print("t2[h]: ",round(tempo_missao[len(tempo_missao)-1],4) ,"// t1[h]: ",round(tempo_missao[len(tempo_missao)-2],4) )
            break
        else:
            M_comb_max = M_comb_max + dcomb

        
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
    produtividade = [100/x for x in tempo_missao]
    
    data = {
        # "Faixa [m]": faixa_vector,
        "Volume de calda [L]": [round(x, 2) for x in TANQUES], 
        "STATUS": operacao,
        "Capacidade operacional [ha/h]": [round(x, 2) for x in produtividade],
        "Tempo de missao [h]": [round(x, 2) for x in tempo_missao],
        "Tempo util [h]": [round(x, 2) for x in tempo_util],
        "Tempo por voo [min]": [round(x, 2) for x in tempo_por_voo],
        "Autonomia Projetada [h]": [round(x, 2) for x in AUTONOMIA_PROJ],
        "N° Voos": [round(x, 2) for x in voo_vector],
        # "Tempo operacional dias": dias,
        "MTOW [kg]": [round(x, 2) for x in MTOW],
        "Massa Estrutura [kg]": [round(x, 2) for x in MASSA_ESTRUTURA],
        "Combustível [kg]": [round(x, 2) for x in COMBUSTIVEL],
        "Abastecimentos": [round(x, 2) for x in abastecimentos],
        "Massa gerador": [round(x, 2) for x in GERADOR],
        "Potência gerador [W]": [round(x, 2) for x in PGERADOR],
        # "Area pulverizada por dia [ha]": talhao_maximus,
        # "Hectare por voo [ha/voo]": area_por_voo,
        # "Capacidade bateria [mha]": capacidade_vector, 
        "Combustível consumido [L]": [round(x, 2) for x in vol_comb],
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




# # Atribuir cores com base no vetor t
# cores = [atribuir_cor(valor) for valor in t_horas]

# fig_trajeto = go.Figure()
# # # Adicionar linhas aos eixos x, y e z

# fig_trajeto.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='lines', line=dict(color=cores, colorscale='Viridis', width=2),))
# fig_trajeto.update_layout(paper_bgcolor="white", showlegend=False, scene=dict(aspectmode='data'))
# #fig_trajeto.write_html("saida_sem_discreziar.html")

# Ao final de cada simulação, salvar os resultados na pasta `Resultados`
for k in range(len(resultados)):
    resultado_file = os.path.join(resultados_path, f"RESULTADOS_{k}.xlsx")
    resultados[k].to_excel(resultado_file)

# # Salvar o gráfico 3D na pasta `Resultados`
# saida_html = os.path.join(resultados_path, "saida_sem_discreziar.html")
# fig_trajeto.write_html(saida_html)

# # Assuming df1 is your DataFrame with the results

# # Supondo que 'df' é o seu DataFrame
# # Primeiro, identifique as colunas que precisam ser corrigidas
# numeric_columns = df.columns.tolist()

# # Remova colunas não numéricas ou que não precisam de correção
# # Por exemplo, se 'STATUS' é uma coluna de texto, podemos excluí-la
# numeric_columns.remove('STATUS')

# # Substitua vírgulas por pontos e converta para float
# for col in numeric_columns:
#     df[col] = df[col].astype(str).str.replace(',', '.').astype(float)

# compila(df)
