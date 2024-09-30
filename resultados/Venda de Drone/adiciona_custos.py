import os
import re
import pandas as pd
from CUSTOS_NOVO import custos
import openpyxl
import math

# Lista de pastas principais (SPAD e n�mero de drones)
pastas_principais = ['Sem SPAD 1 Drones', 'Com SPAD 2 Drones', 'Com SPAD 3 Drones', 'Com SPAD 4 Drones']

# Lista de pastas de velocidades a serem processadas dentro das pastas principais
pastas_velocidade = ['vel05', 'vel10', 'vel20', 'vel40']

# Definir as varia��es dos componentes do nome da pasta
field_values = [100, 64, 36, 16]
ha_values = [100]
percent_values = [2, 6, 10, 14, 18]
m_values = [0, 6, 12]
suffixes = ["TSP", "LM"]

# Fun��o para determinar os valores de SPAD e n_drones com base no nome da pasta principal
def associar_variaveis(nome_pasta):
    if "Sem SPAD" in nome_pasta:
        SPAD = 0
        n_drones = 1
    elif "Com SPAD" in nome_pasta:
        SPAD = 1
        if "2 Drones" in nome_pasta:
            n_drones = 2
        elif "3 Drones" in nome_pasta:
            n_drones = 3
        elif "4 Drones" in nome_pasta:
            n_drones = 4
    else:
        SPAD = None
        n_drones = None
    return SPAD, n_drones

# Iterar sobre cada pasta principal (SPAD e drones)
for pasta_principal in pastas_principais:
    print(f"Iniciando processamento na pasta: {pasta_principal}")
    
    # Associa as vari�veis SPAD e n_drones com base no nome da pasta principal
    SPAD, n_drones = associar_variaveis(pasta_principal)
    
    # Verifica se as vari�veis foram atribu�das corretamente
    if SPAD is None or n_drones is None:
        print(f"Nome de pasta inv�lido: {pasta_principal}. Pulando esta pasta.")
        continue
    
    # Muda para o diret�rio da pasta correspondente
    os.chdir(pasta_principal)
    
    # Iterar sobre cada pasta de velocidade dentro da pasta principal
    for pasta_vel in pastas_velocidade:
        print(f"Iniciando processamento na pasta de velocidade: {pasta_vel}")
        
        # Muda para o diret�rio da pasta de velocidade correspondente
        os.chdir(pasta_vel)
        
        # Iterar sobre todas as combina��es de pastas dentro de cada pasta de velocidade
        for field_value in field_values:
            for ha_value in ha_values:
                for percent_value in percent_values:
                    for m_value in m_values:
                        for suffix in suffixes:
                            # Nome da pasta
                            folder_name = f'field_{field_value}ha_{ha_value}ha_{percent_value}%_{m_value}m_0_{suffix}'
                            print(f"Processando pasta: {folder_name}")
                            
                            # Caminho para a planilha dentro dessa pasta
                            file_path = os.path.join(folder_name, 'Planilha_Unificada.xlsx')

                            # Verifica se o arquivo existe antes de prosseguir
                            if not os.path.exists(file_path):
                                print(f"Arquivo n�o encontrado: {file_path}")
                                continue
                            
                            # Abrir a planilha para obter os nomes das abas
                            try:
                                with pd.ExcelFile(file_path) as xls:
                                    all_sheets = xls.sheet_names  # Lista com o nome de todas as abas
                            except Exception as e:
                                print(f"Erro ao abrir o arquivo {file_path}: {e}")
                                continue
                            
                            # Filtrar as abas que come�am com "RESULTADOS_"
                            sheets_to_process = [sheet for sheet in all_sheets if sheet.startswith('RESULTADOS_')]

                            # Limitar para no m�ximo 30 abas
                            sheets_to_process = sheets_to_process[:30]

                            # Fun��o para processar cada aba
                            def processar_aba(sheet_name, file_path, field_value):
                                # Inicializar as listas para armazenar os resultados
                                pre�o_total_por_ha_simples = []
                                pre�o_total_por_ha_presumido = []
                                pre�o_total_por_ha_real = []
                                pre�o_total_simples = []
                                pre�o_total_presumido = []
                                pre�o_total_real = []
                                Prod_capex = []
                                Dias = []
                                
                                # Ler a planilha
                                df = pd.read_excel(file_path, sheet_name=sheet_name)
                                
                                # Fazer os c�lculos e preencher os vetores
                                for i in range(len(df)):
                                    M_pulv_max = df.loc[i, 'Volume de calda [L]']
                                    area_tot = 1000
                                    tempo_util = df.loc[i, 'Tempo util [h]']
                                    comb_cons = df.loc[i, 'Combust�vel consumido [L]']
                                    voo = df.loc[i, 'N� Voos']
                                    n_abs = df.loc[i, 'Abastecimentos']
                                    dias = math.ceil(area_tot/field_value * df.loc[i, 'Tempo de missao [h]'] / (24*n_drones))
                                    Produtividade = df.loc[i, 'Capacidade operacional [ha/h]']
                                    
                                    # Multiplicar Produtividade pelo valor extra�do da pasta
                                    Produtividade *= field_value / 100

                                    area_dia = Produtividade * 24 * n_drones
                                    
                                    # Chamar a fun��o custos e obter os valores
                                    (a, b, c, d, e, f, g) = custos(M_pulv_max, area_tot, tempo_util, comb_cons, voo, n_abs, dias, area_dia, Produtividade, SPAD, n_drones)
                                    
                                    # Armazenar os resultados nos vetores
                                    pre�o_total_por_ha_simples.append(a)
                                    pre�o_total_por_ha_presumido.append(b)
                                    pre�o_total_por_ha_real.append(c)
                                    pre�o_total_simples.append(d)
                                    pre�o_total_presumido.append(e)
                                    pre�o_total_real.append(f)
                                    Prod_capex.append(g)
                                    Dias.append(dias)

                                # Adicionar os vetores como novas colunas no DataFrame
                                df['Pre�o Total por ha Simples [R$/ha]'] = pre�o_total_por_ha_simples
                                df['Pre�o Total por ha Presumido [R$/ha]'] = pre�o_total_por_ha_presumido
                                df['Pre�o Total por ha Real [R$/ha]'] = pre�o_total_por_ha_real
                                df['Pre�o Total Simples [R$]'] = pre�o_total_simples
                                df['Pre�o Total Presumido [R$]'] = pre�o_total_presumido
                                df['Prod CAPEX [ha/h/R$]'] = Prod_capex
                                df['Dias Trabalhados'] = Dias
                                df['Pre�o Total Real [R$]'] = pre�o_total_real
                                
                                return df

                            # Abrir o arquivo Excel para escrita
                            try:
                                with pd.ExcelWriter(file_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
                                    # Iterar sobre as abas filtradas
                                    for sheet in sheets_to_process:
                                        df_resultado = processar_aba(sheet, file_path, field_value)
                                        # Escrever o DataFrame de volta na aba processada
                                        df_resultado.to_excel(writer, sheet_name=sheet, index=False)
                                print(f"Processamento conclu�do para a pasta: {folder_name}")
                            except Exception as e:
                                print(f"Erro ao processar a pasta {folder_name}: {e}")
        
        # Voltar para o diret�rio da pasta principal antes de processar a pr�xima pasta de velocidade
        os.chdir('..')
    
    # Voltar para o diret�rio raiz antes de processar a pr�xima pasta principal
    os.chdir('..')

print("Processamento finalizado para todas as pastas.")
