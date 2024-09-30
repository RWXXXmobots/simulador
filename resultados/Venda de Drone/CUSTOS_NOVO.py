# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 10:40:00 2024

ALTERAÇÃO 1 CICLO ->> 1 ABS

@author: MarcusMoreira esse otario
"""
import numpy as np
import math

def custos(M_pulv_max,area_tot,tempo_util,comb_cons,voo,n_abs,dias,area_dia,Produtividade,SPAD,n_drones):

    #valor_drone_carregador = (2140.9*M_pulv_max**(0.4034)) * 5 

    valor_drone_carregador = 41427.2565404429*M_pulv_max**(0.4033680105) #-13.59*M_pulv_max**2 + 3159.4*M_pulv_max + 77592
    valor_bateria = 0#14578.05*np.log(capac_bat)-120979.66
    valor_eq_suporte = 35850
    valor_gerador = 25850
    valor_carro = 120000

    if SPAD == 1:
        valor_eq_suporte = 18273
        valor_carro = 310000
        valor_SPAD = 420000
        valor_gerador = 0
        aux_spad = 1
        Tempo_operacao_ger = 12
        consumo_ger = 3.9
        preco_combus = 5.91
    else:
        valor_SPAD = 0
        aux_spad = 0
        consumo_ger = 5.5
        Tempo_operacao_ger = 12
        preco_combus = 5.91

    
    dist_cidade_area = 150      #[km]
    dist_hotel_area = 35        #[km]
    n_periodos = math.ceil(dias*2)
    
    # FISCAL -------------------#
    
    margem_lucro = 0.3
    ISS_simpes = 0.1209; ISS_presumido = 0.05; ISS_real = 0.05
    PIS = 0.0065; confins = 0.03; CSLL = 0.09; IRPJ = 0.15; base_lucro = 0.32
    contr_lucro = 0.5; outros_cpp = 0.05
    comissao_comercial = 0
    
    dias_uteis_ano = 5*52
    funcionarios = 6
    salario_piloto = 5500
    salario_auxiliar = 4500
    encargos = 0.63
    insalubridade = 1412 * 0.2
    hotel_diaria = 120
    alimentacao = 80
    telefone_mensal = 60
    EPI = 900
    vida_EPI = 6        #[meses]
    lavanderia_epi = 150; recorrencia = 14 #dias
    depreciacao_drone = 5   #[anos]
    depreciacao_eq = 7  #[anos]
    depreciacao_ger = 10  #[anos]
    depreciacao_Carro = 5 #[anos]
    limp_sistema = 300; recorr_limp = 14 #dias
    limp_carro = 150; recorr_limp_carro = 14 #dias
    seguro_anual_drone = 550
    taxa_outros = 0.15 ; ocorrencia = 3 # manuntenção por ano
    seguro_anual_carro = 4000; impostos_carro = 1300
    outros_anual = 4500
    adicional_noturno = 1 + 0.3/3

    depreciacao_spad = 10 #[anos]
    
#================# PREÇO POR DIA # =============================#

    custo_folha_dia_piloto = (salario_piloto*(1+encargos) + insalubridade)/(dias_uteis_ano/12)*3*adicional_noturno
    custo_folha_dia_auxilar = (salario_auxiliar*(1+encargos) + insalubridade)/(dias_uteis_ano/12)*(1-aux_spad)*3*adicional_noturno
    custo_hotel_dia = hotel_diaria * funcionarios/ (1+aux_spad)
    custo_alimentacao_dia = alimentacao*funcionarios/ (1+aux_spad)
    custo_telefone_dia = telefone_mensal/(dias_uteis_ano/12)
    custo_EPI_dia = (EPI/vida_EPI)*funcionarios/(dias_uteis_ano/12)/ (1+aux_spad)
    custo_lavan_EPI_dia = (lavanderia_epi/recorrencia)*funcionarios/ (1+aux_spad)
    custo_depr_drone_dia = valor_drone_carregador/depreciacao_drone/dias_uteis_ano*n_drones
    custo_depr_eq_dia = valor_eq_suporte/(depreciacao_eq*dias_uteis_ano)
    custo_depr_ger_dia = valor_gerador/(depreciacao_ger*dias_uteis_ano)
    custo_limp_sistema_dia = limp_sistema/recorr_limp
    custo_seguro_drone_dia = seguro_anual_drone/dias_uteis_ano*n_drones
    custos_outros_dia = ocorrencia * taxa_outros * valor_drone_carregador/dias_uteis_ano*n_drones
    custo_depre_carro_dia = valor_carro/depreciacao_Carro/dias_uteis_ano
    custo_limp_carro_dia = limp_carro/recorr_limp_carro
    custo_seguro_carro_dia = seguro_anual_carro/dias_uteis_ano
    custos_imposto_carro_dia = impostos_carro/dias_uteis_ano
    cutos_outros_dia = outros_anual/dias_uteis_ano
    custos_depr_spad = (valor_SPAD+aux_spad*(1360*2+3410*2))/(depreciacao_spad*dias_uteis_ano) #Custo bombas + starlink inclusos
    custos_combustivel_ger = Tempo_operacao_ger*consumo_ger*preco_combus*n_drones
    
    custos_dia = custo_folha_dia_piloto + custo_folha_dia_auxilar + custo_hotel_dia + custo_alimentacao_dia + custo_telefone_dia + custo_EPI_dia + custo_lavan_EPI_dia + custo_depr_drone_dia + custo_depr_eq_dia + custo_limp_sistema_dia + custo_seguro_drone_dia + custos_outros_dia + custo_depre_carro_dia + custo_limp_carro_dia + custo_seguro_carro_dia + custos_imposto_carro_dia + cutos_outros_dia + custo_depr_ger_dia + custos_depr_spad + custos_combustivel_ger

    preco_servico_dia_simples = custos_dia/(1 - margem_lucro - ISS_simpes)
    preco_servico_dia_presumido = custos_dia/(1-(2*ISS_presumido + PIS + confins + base_lucro*(CSLL+IRPJ) + outros_cpp + margem_lucro))
    preco_servico_dia_real = custos_dia/(1-(2*ISS_real + PIS + confins + contr_lucro*margem_lucro*(CSLL+IRPJ) + outros_cpp + margem_lucro))


#================# PREÇO POR km # =============================#

    vida_carro = 250000                         #[km]
    custo_vida_km = valor_carro/vida_carro
    combustivel = 6             #[R$/L]
    consumo_carro = 9           #[L/km]
    custo_comb_km = combustivel/consumo_carro       #[R$/km]
    custo_pedagio_km = 0.1          #[R$/km]
    custo_revisao1 = 500
    intervalo_revisao1 = 5000       #[km]
    custo_revisa2 = 1500
    intervalo_revisao2 = 20000      #[km]
    custo_revisao3 = 4500
    intervalo_revisao3 = 40000
    vida_util_pneu_spad = 40000
    Tempo_entre_revisoes = 20000
    Custo_pneus = 4*530/vida_util_pneu_spad*aux_spad
    custo_rev_spad = 4634/Tempo_entre_revisoes*aux_spad
    custo_revisao_km = custo_revisao1/intervalo_revisao1 + custo_revisa2/intervalo_revisao2 + custo_revisao3/intervalo_revisao3 + Custo_pneus + custo_rev_spad
    custo_km_rodado = custo_vida_km + custo_comb_km + custo_pedagio_km + custo_revisao_km


    preco_servico_km_simples = custo_km_rodado/(1 - margem_lucro - ISS_simpes)
    preco_servico_km_presumido = custo_km_rodado/(1-(2*ISS_presumido + PIS + confins + base_lucro*(CSLL+IRPJ) + outros_cpp + margem_lucro))
    preco_servico_km_real = custo_km_rodado/(1-(2*ISS_real + PIS + confins + contr_lucro*margem_lucro*(CSLL+IRPJ) + outros_cpp + margem_lucro))

#=================# PREÇO POR ha #===============================#

    vida_drone_carregador = 6000           #[horas de voo]
    custo_vida_drone_carregador_ha = valor_drone_carregador*tempo_util/(vida_drone_carregador*area_dia)*n_drones
    vida_bateria = 1500     #[ciclos]
    custo_bateria =  valor_bateria*n_abs/(vida_bateria*area_dia)*n_drones         #[R$/ha]
    cons_gasol_drone_ha = comb_cons/area_dia     #[L/ha]
    comb_drone = 7.69                               #[R$/L]
    custo_gasol_drone_ha = cons_gasol_drone_ha*comb_drone*n_drones       #[R$/ha]
    cons_oleo_drone_ha = cons_gasol_drone_ha/30             #[L/ha]
    oleo_drone = 101                                #[R$/L]
    custo_oleo_drone_ha = oleo_drone*cons_oleo_drone_ha*n_drones     #[R$/ha]
    media_revisao1 = 0.04
    revisao_drone1_ha = 800*area_dia/tempo_util*n_drones    #[ha]
    media_revisao2 = 0.08
    revisao_drone2_ha = 1080*area_dia/tempo_util*n_drones   #[ha]
    custo_revisao_ha = valor_drone_carregador*(media_revisao1/revisao_drone1_ha + media_revisao2/revisao_drone2_ha)*n_drones
    
    comissao_piloto = 0.05
    custo_comissao_piloto_ha_simples = 0.0
    custo_comissao_piloto_ha_presumido = 0.0 
    custo_comissao_piloto_ha_real = 0.0
    flag1 = 0.05 * 20; flag2 = 0.05 * 20; flag3  = 0.05 * 20
    ERRO = 1
    
    while (ERRO > 10**-10):
        custo_comissao_piloto_ha_simples = flag1
        custo_comissao_piloto_ha_presumido = flag2
        custo_comissao_piloto_ha_real = flag3
        
        custo_por_ha_simples = custo_vida_drone_carregador_ha + custo_bateria + custo_gasol_drone_ha + custo_oleo_drone_ha + custo_revisao_ha + custo_comissao_piloto_ha_simples
        custo_por_ha_presumido = custo_vida_drone_carregador_ha + custo_bateria + custo_gasol_drone_ha + custo_oleo_drone_ha + custo_revisao_ha + custo_comissao_piloto_ha_presumido
        custo_por_ha_real = custo_vida_drone_carregador_ha + custo_bateria + custo_gasol_drone_ha + custo_oleo_drone_ha + custo_revisao_ha + custo_comissao_piloto_ha_real
        
        preco_servico_ha_simples = custo_por_ha_simples/(1 - margem_lucro - ISS_simpes)
        preco_servico_ha_presumido = custo_por_ha_presumido/(1-(2*ISS_presumido + PIS + confins + base_lucro*(CSLL+IRPJ) + outros_cpp + margem_lucro))
        preco_servico_ha_real = custo_por_ha_real/(1-(2*ISS_real + PIS + confins + contr_lucro*margem_lucro*(CSLL+IRPJ) + outros_cpp + margem_lucro))
        
        preço_total_simples = (dias*preco_servico_dia_simples*(1 + comissao_comercial)) + preco_servico_km_simples*(2*dist_cidade_area + n_periodos*dist_hotel_area) + preco_servico_ha_simples*area_tot*(1 + comissao_comercial)
        preço_total_presumido = (dias*preco_servico_dia_presumido*(1 + comissao_comercial)) + preco_servico_km_presumido*(2*dist_cidade_area + n_periodos*dist_hotel_area) + preco_servico_ha_presumido*area_tot*(1 + comissao_comercial)
        preço_total_real = (dias*preco_servico_dia_real*(1 + comissao_comercial)) + preco_servico_km_real*(2*dist_cidade_area + n_periodos*dist_hotel_area) + preco_servico_ha_real*area_tot*(1 + comissao_comercial)
        
        preço_total_por_ha_simples = preço_total_simples/area_tot
        preço_total_por_ha_presumido = preço_total_presumido/area_tot
        preço_total_por_ha_real = preço_total_real/area_tot
        
        flag1 = comissao_piloto*preço_total_por_ha_simples
        flag2 = comissao_piloto*preço_total_por_ha_presumido
        flag3 = comissao_piloto*preço_total_por_ha_real
        ERRO = abs(custo_comissao_piloto_ha_simples-flag1 + custo_comissao_piloto_ha_presumido - flag2 + custo_comissao_piloto_ha_real -flag3)
        
        #print(custo_bateria)
        
        #print(preço_total_por_ha_simples,preço_total_por_ha_presumido,preço_total_por_ha_real)
        #return(custo_revisao_ha*1000,(custo_gasol_drone_ha+custo_oleo_drone_ha)*1000,custo_comissao_piloto_ha_real*1000,custo_bateria*1000,vida_drone_carregador*1000,(2*dist_cidade_area + n_periodos*dist_hotel_area)*custo_revisao_km,(2*dist_cidade_area + n_periodos*dist_hotel_area)*custo_pedagio_km,(2*dist_cidade_area + n_periodos*dist_hotel_area)*custo_comb_km,(2*dist_cidade_area + n_periodos*dist_hotel_area)*custo_vida_km,cutos_outros_dia*dias,custos_imposto_carro_dia*dias,custo_seguro_carro_dia*dias,custo_limp_carro_dia*dias,custo_depre_carro_dia*dias,custos_outros_dia*dias,custo_seguro_drone_dia*dias,custo_limp_sistema_dia*dias,custo_depr_eq_dia*dias,custo_depr_drone_dia*dias,custo_lavan_EPI_dia*dias,custo_EPI_dia*dias,custo_telefone_dia*dias,custo_alimentacao_dia*dias,custo_hotel_dia*dias,custo_folha_dia_auxilar*dias,custo_folha_dia_piloto*dias,preço_total_simples,preço_total_presumido,preço_total_real,preço_total_por_ha_simples,preço_total_por_ha_presumido,preço_total_por_ha_real)
    
    Prod_capex = Produtividade/(valor_drone_carregador+valor_SPAD)

    return(preço_total_por_ha_simples,preço_total_por_ha_presumido,preço_total_por_ha_real,preço_total_simples,preço_total_presumido,preço_total_real,Prod_capex)

# (a,b,c,d,e,f,g)=custos(20,1000,24,50.97,55,18,10,100,3)

# print(a)
# print(b)
# print(c)
# print(d)
# print(e)
# print(f)
# print(g)
