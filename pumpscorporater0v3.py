import streamlit as st
import pandas as pd
import math
import time
import numpy as np
from scipy.optimize import root
import graphviz
import matplotlib.pyplot as plt
import io
import yaml
from yaml.loader import SafeLoader
import streamlit_authenticator as stauth

# Importando as funções do banco de dados e do gerador de relatórios
from database import (
    setup_database, save_scenario, load_scenario, get_user_projects,
    get_scenarios_for_project, delete_scenario, add_user_fluid, get_user_fluids,
    delete_user_fluid, add_user_material, get_user_materials, delete_user_material
)
from report_generator import generate_report

# --- CONFIGURAÇÕES E CONSTANTES ---
st.set_page_config(layout="wide", page_title="Análise de Redes Hidráulicas")
plt.style.use('seaborn-v0_8-whitegrid')

# BIBLIOTECAS PADRÃO
MATERIAIS_PADRAO = {
    "Aço Carbono (novo)": 0.046, "Aço Carbono (pouco uso)": 0.1, "Aço Carbono (enferrujado)": 0.2,
    "Aço Inox": 0.002, "Ferro Fundido": 0.26, "PVC / Plástico": 0.0015, "Concreto": 0.5
}
FLUIDOS_PADRAO = {
    "Água a 20°C": {"rho": 998.2, "nu": 1.004e-6, "pv_kpa": 2.34},
    "Etanol a 20°C": {"rho": 789.0, "nu": 1.51e-6, "pv_kpa": 5.8}
}
K_FACTORS = {
    "Entrada de Borda Viva": 0.5, "Entrada Levemente Arredondada": 0.2, "Entrada Bem Arredondada": 0.04,
    "Saída de Tubulação": 1.0, "Válvula Gaveta (Totalmente Aberta)": 0.2, "Válvula Gaveta (1/2 Aberta)": 5.6,
    "Válvula Globo (Totalmente Aberta)": 10.0, "Válvula de Retenção (Tipo Portinhola)": 2.5,
    "Cotovelo 90° (Raio Longo)": 0.6, "Cotovelo 90° (Raio Curto)": 0.9, "Cotovelo 45°": 0.4,
    "Curva de Retorno 180°": 2.2, "Tê (Fluxo Direto)": 0.6, "Tê (Fluxo Lateral)": 1.8,
}

# --- FUNÇÕES DE CÁLCULO ---
def calcular_perda_serie(lista_trechos, vazao_m3h, fluido_selecionado, materiais_combinados, fluidos_combinados):
    perda_total = 0
    if not lista_trechos:
        return 0.0
    for trecho in lista_trechos:
        perdas = calcular_perdas_trecho(trecho, vazao_m3h, fluido_selecionado, materiais_combinados, fluidos_combinados)
        perda_total += perdas["principal"] + perdas["localizada"]
        perda_total += trecho.get('perda_equipamento_m', 0.0)
    return perda_total

def calcular_perdas_trecho(trecho, vazao_m3h, fluido_selecionado, materiais_combinados, fluidos_combinados):
    if vazao_m3h < 0: vazao_m3h = 0
    rugosidade_mm = materiais_combinados[trecho["material"]]
    vazao_m3s, diametro_m = vazao_m3h / 3600, trecho["diametro"] / 1000
    nu = fluidos_combinados[fluido_selecionado]["nu"]
    if diametro_m <= 0: return {"principal": 1e12, "localizada": 0, "velocidade": 0}
    area = (math.pi * diametro_m**2) / 4
    velocidade = vazao_m3s / area if area > 0 else 0
    reynolds = (velocidade * diametro_m) / nu if nu > 0 else 0
    fator_atrito = 0
    if reynolds > 4000:
        rugosidade_m = rugosidade_mm / 1000
        if diametro_m <= 0: return {"principal": 1e12, "localizada": 0, "velocidade": 0}
        log_term = math.log10((rugosidade_m / (3.7 * diametro_m)) + (5.74 / reynolds**0.9))
        fator_atrito = 0.25 / (log_term**2)
    elif reynolds > 0:
        fator_atrito = 64 / reynolds
    perda_principal = fator_atrito * (trecho["comprimento"] / diametro_m) * (velocidade**2 / (2 * 9.81))
    k_total_trecho = sum(ac["k"] * ac["quantidade"] for ac in trecho["acessorios"])
    perda_localizada = k_total_trecho * (velocidade**2 / (2 * 9.81))
    return {"principal": perda_principal, "localizada": perda_localizada, "velocidade": velocidade}

def calcular_perdas_paralelo(ramais, vazao_total_m3h, fluido_selecionado, materiais_combinados, fluidos_combinados):
    num_ramais = len(ramais)
    if num_ramais < 2: return 0, {}
    lista_ramais = list(ramais.values())
    def equacoes_perda(vazoes_parciais_m3h):
        vazao_ultimo_ramal = vazao_total_m3h - sum(vazoes_parciais_m3h)
        if vazao_ultimo_ramal < -0.01: return [1e12] * (num_ramais - 1)
        todas_vazoes = np.append(vazoes_parciais_m3h, vazao_ultimo_ramal)
        perdas = [calcular_perda_serie(ramal, vazao, fluido_selecionado, materiais_combinados, fluidos_combinados) for ramal, vazao in zip(lista_ramais, todas_vazoes)]
        erros = [perdas[i] - perdas[-1] for i in range(num_ramais - 1)]
        return erros
    chute_inicial = np.full(num_ramais - 1, vazao_total_m3h / num_ramais)
    solucao = root(equacoes_perda, chute_inicial, method='hybr', options={'xtol': 1e-8})
    if not solucao.success: return -1, {}
    vazoes_finais = np.append(solucao.x, vazao_total_m3h - sum(solucao.x))
    perda_final_paralelo = calcular_perda_serie(lista_ramais[0], vazoes_finais[0], fluido_selecionado, materiais_combinados, fluidos_combinados)
    distribuicao_vazao = {nome_ramal: vazao for nome_ramal, vazao in zip(ramais.keys(), vazoes_finais)}
    return perda_final_paralelo, distribuicao_vazao

def calcular_analise_energetica(vazao_m3h, h_man, eficiencia_bomba_percent, eficiencia_motor_percent, horas_dia, custo_kwh, fluido_selecionado, fluidos_combinados):
    rho = fluidos_combinados[fluido_selecionado]["rho"]
    ef_bomba = eficiencia_bomba_percent / 100
    ef_motor = eficiencia_motor_percent / 100
    potencia_eletrica_kW = (vazao_m3h / 3600 * rho * 9.81 * h_man) / (ef_bomba * ef_motor) / 1000 if ef_bomba * ef_motor > 0 else 0
    custo_anual = potencia_eletrica_kW * horas_dia * 30 * 12 * custo_kwh
    return {"potencia_eletrica_kW": potencia_eletrica_kW, "custo_anual": custo_anual}

def criar_funcao_curva(df_curva, col_x, col_y, grau=2):
    df_curva[col_x] = pd.to_numeric(df_curva[col_x], errors='coerce')
    df_curva[col_y] = pd.to_numeric(df_curva[col_y], errors='coerce')
    df_curva = df_curva.dropna(subset=[col_x, col_y])
    if len(df_curva) < grau + 1: return None
    coeficientes = np.polyfit(df_curva[col_x], df_curva[col_y], grau)
    return np.poly1d(coeficientes)

def converter_pressao_para_mca(pressao, unidade_origem, rho_fluido):
    if rho_fluido <= 0: return 0.0
    # Converte a pressão para Pascal (Pa)
    if unidade_origem == 'kgf/cm2':
        pressao_pa = pressao * 98066.5
    elif unidade_origem == 'kpa':
        pressao_pa = pressao * 1000
    else:
        return 0.0
    # Converte de Pa para metros de coluna de fluido
    altura_m = pressao_pa / (rho_fluido * 9.81)
    return altura_m

def calcular_pressao_atm_mca(altitude_m, rho_fluido):
    if rho_fluido <= 0: return 0.0
    # Fórmula Barométrica (aproximação para a troposfera)
    pressao_pa = 101325 * (1 - 2.25577e-5 * altitude_m)**5.25588
    return pressao_pa / (rho_fluido * 9.81)

def encontrar_ponto_operacao(sistema_succao, sistema_recalque, h_estatica_total, fluido, func_curva_bomba, materiais_combinados, fluidos_combinados):
    def curva_sistema(vazao_m3h):
        if vazao_m3h < 0: return h_estatica_total
        perda_total_dinamica = 0
        # Perda de carga na sucção
        perda_total_dinamica += calcular_perda_serie(sistema_succao, vazao_m3h, fluido, materiais_combinados, fluidos_combinados)
        # Perda de carga no recalque
        perda_total_dinamica += calcular_perda_serie(sistema_recalque['antes'], vazao_m3h, fluido, materiais_combinados, fluidos_combinados)
        perda_par, _ = calcular_perdas_paralelo(sistema_recalque['paralelo'], vazao_m3h, fluido, materiais_combinados, fluidos_combinados)
        if perda_par == -1: return 1e12
        perda_total_dinamica += perda_par
        perda_total_dinamica += calcular_perda_serie(sistema_recalque['depois'], vazao_m3h, fluido, materiais_combinados, fluidos_combinados)
        return h_estatica_total + perda_total_dinamica
    def erro(vazao_m3h):
        if vazao_m3h < 0: return 1e12
        return func_curva_bomba(vazao_m3h) - curva_sistema(vazao_m3h)
    solucao = root(erro, 50.0, method='hybr', options={'xtol': 1e-8})
    if solucao.success and solucao.x[0] > 1e-3:
        vazao_op = solucao.x[0]
        altura_op = func_curva_bomba(vazao_op)
        return vazao_op, altura_op, curva_sistema
    else:
        return None, None, curva_sistema

def gerar_diagrama_rede(sistema_succao, sistema_recalque, vazao_total, distribuicao_vazao, fluido, materiais_combinados, fluidos_combinados):
    dot = graphviz.Digraph(comment='Rede de Tubulação'); dot.attr('graph', rankdir='LR', splines='ortho'); dot.attr('node', shape='point')
    dot.node('start', 'Reservatório\nSucção', shape='cylinder', style='filled', fillcolor='lightblue')
    ultimo_no = 'start'

    # Desenhar Linha de Sucção
    for i, trecho in enumerate(sistema_succao):
        proximo_no = f"no_succao_{i+1}"
        perdas_info = calcular_perdas_trecho(trecho, vazao_total, fluido, materiais_combinados, fluidos_combinados)
        velocidade = perdas_info['velocidade']
        perda_trecho_hidraulica = perdas_info['principal'] + perdas_info['localizada'] + trecho.get('perda_equipamento_m', 0)
        label = f"{trecho.get('nome', f'Trecho Sucção {i+1}')}\\n{vazao_total:.1f} m³/h\\n{velocidade:.2f} m/s\\nPerda: {perda_trecho_hidraulica:.2f} m"
        dot.edge(ultimo_no, proximo_no, label=label)
        ultimo_no = proximo_no

    dot.node('pump', 'Bomba', shape='circle', style='filled', fillcolor='orange'); dot.edge(ultimo_no, 'pump')
    ultimo_no = 'pump'

    # Desenhar Linha de Recalque
    for i, trecho in enumerate(sistema_recalque['antes']):
        proximo_no = f"no_antes_{i+1}"
        perdas_info = calcular_perdas_trecho(trecho, vazao_total, fluido, materiais_combinados, fluidos_combinados)
        velocidade = perdas_info['velocidade']
        perda_trecho_hidraulica = perdas_info['principal'] + perdas_info['localizada'] + trecho.get('perda_equipamento_m', 0)
        label = f"{trecho.get('nome', f'Trecho Antes {i+1}')}\\n{vazao_total:.1f} m³/h\\n{velocidade:.2f} m/s\\nPerda: {perda_trecho_hidraulica:.2f} m"
        dot.edge(ultimo_no, proximo_no, label=label)
        ultimo_no = proximo_no

    if len(sistema_recalque['paralelo']) >= 2 and distribuicao_vazao:
        no_divisao = ultimo_no; no_juncao = 'no_juncao'; dot.node(no_juncao)
        for nome_ramal, trechos_ramal in sistema_recalque['paralelo'].items():
            vazao_ramal = distribuicao_vazao.get(nome_ramal, 0); ultimo_no_ramal = no_divisao
            for i, trecho in enumerate(trechos_ramal):
                perdas_info_ramal = calcular_perdas_trecho(trecho, vazao_ramal, fluido, materiais_combinados, fluidos_combinados)
                velocidade = perdas_info_ramal['velocidade']
                perda_trecho_ramal_hidraulica = perdas_info_ramal['principal'] + perdas_info_ramal['localizada'] + trecho.get('perda_equipamento_m', 0)
                label_ramal = f"{trecho.get('nome', f'{nome_ramal} (T{i+1})')}\\n{vazao_ramal:.1f} m³/h\\n{velocidade:.2f} m/s\\nPerda: {perda_trecho_ramal_hidraulica:.2f} m"
                if i == len(trechos_ramal) - 1:
                    dot.edge(ultimo_no_ramal, no_juncao, label=label_ramal)
                else:
                    proximo_no_ramal = f"no_{nome_ramal}_{i+1}".replace(" ", "_")
                    dot.edge(ultimo_no_ramal, proximo_no_ramal, label=label_ramal)
                    ultimo_no_ramal = proximo_no_ramal
        ultimo_no = no_juncao

    for i, trecho in enumerate(sistema_recalque['depois']):
        proximo_no = f"no_depois_{i+1}"
        perdas_info = calcular_perdas_trecho(trecho, vazao_total, fluido, materiais_combinados, fluidos_combinados)
        velocidade = perdas_info['velocidade']
        perda_trecho_hidraulica = perdas_info['principal'] + perdas_info['localizada'] + trecho.get('perda_equipamento_m', 0)
        label = f"{trecho.get('nome', f'Trecho Depois {i+1}')}\\n{vazao_total:.1f} m³/h\\n{velocidade:.2f} m/s\\nPerda: {perda_trecho_hidraulica:.2f} m"
        dot.edge(ultimo_no, proximo_no, label=label)
        ultimo_no = proximo_no

    dot.node('end', 'Fim', shape='circle', style='filled', fillcolor='lightgray'); dot.edge(ultimo_no, 'end')
    return dot

def gerar_grafico_sensibilidade_diametro(sistema_succao_base, sistema_recalque_base, fator_escala_range, **params_fixos):
    custos, fatores = [], np.arange(fator_escala_range[0], fator_escala_range[1] + 5, 5)
    materiais_combinados = params_fixos['materiais_combinados']
    fluidos_combinados = params_fixos['fluidos_combinados']
    for fator in fatores:
        escala = fator / 100.0
        # Escalar diâmetros para sucção
        sistema_succao_escalado = [t.copy() for t in sistema_succao_base]
        for t in sistema_succao_escalado:
             t['diametro'] *= escala
        # Escalar diâmetros para recalque
        sistema_recalque_escalado = {'antes': [t.copy() for t in sistema_recalque_base['antes']], 'paralelo': {k: [t.copy() for t in v] for k, v in sistema_recalque_base['paralelo'].items()}, 'depois': [t.copy() for t in sistema_recalque_base['depois']]}
        for t_list in sistema_recalque_escalado.values():
            if isinstance(t_list, list):
                for t in t_list: t['diametro'] *= escala
            elif isinstance(t_list, dict):
                for _, ramal in t_list.items():
                    for t in ramal: t['diametro'] *= escala

        vazao_ref = params_fixos['vazao_op']
        perda_succao = calcular_perda_serie(sistema_succao_escalado, vazao_ref, params_fixos['fluido'], materiais_combinados, fluidos_combinados)
        perda_antes = calcular_perda_serie(sistema_recalque_escalado['antes'], vazao_ref, params_fixos['fluido'], materiais_combinados, fluidos_combinados)
        perda_par, _ = calcular_perdas_paralelo(sistema_recalque_escalado['paralelo'], vazao_ref, params_fixos['fluido'], materiais_combinados, fluidos_combinados)
        perda_depois = calcular_perda_serie(sistema_recalque_escalado['depois'], vazao_ref, params_fixos['fluido'], materiais_combinados, fluidos_combinados)
        
        if perda_par == -1: custos.append(np.nan); continue

        perda_total_dinamica = perda_succao + perda_antes + perda_par + perda_depois
        h_man = params_fixos['h_estatica_total'] + perda_total_dinamica
        
        resultado_energia = calcular_analise_energetica(vazao_ref, h_man, fluidos_combinados=fluidos_combinados, **params_fixos['equipamentos'])
        custos.append(resultado_energia['custo_anual'])
        
    return pd.DataFrame({'Fator de Escala nos Diâmetros (%)': fatores, 'Custo Anual de Energia (R$)': custos})

def render_trecho_ui(trecho, prefixo, lista_trechos, materiais_combinados):
    trecho['nome'] = st.text_input("Nome do Trecho", value=trecho.get('nome'), key=f"nome_{prefixo}_{trecho['id']}")
    c1, c2, c3, c4 = st.columns(4)
    trecho['comprimento'] = c1.number_input("L (m)", min_value=0.1, value=trecho['comprimento'], key=f"comp_{prefixo}_{trecho['id']}")
    trecho['diametro'] = c2.number_input("Ø (mm)", min_value=1.0, value=trecho['diametro'], key=f"diam_{prefixo}_{trecho['id']}")
    lista_materiais = list(materiais_combinados.keys())
    try:
        idx_material = lista_materiais.index(trecho.get('material', 'Aço Carbono (novo)'))
    except ValueError:
        idx_material = 0
    trecho['material'] = c3.selectbox("Material", options=lista_materiais, index=idx_material, key=f"mat_{prefixo}_{trecho['id']}")
    trecho['perda_equipamento_m'] = c4.number_input("Perda Equip. (m)", min_value=0.0, value=trecho.get('perda_equipamento_m', 0.0), key=f"equip_{prefixo}_{trecho['id']}", format="%.2f")

    st.markdown("**Acessórios (Fittings)**")
    for idx, acessorio in enumerate(trecho['acessorios']):
        col1, col2 = st.columns([0.8, 0.2])
        col1.info(f"{acessorio['quantidade']}x {acessorio['nome']} (K = {acessorio['k']})")
        if col2.button("X", key=f"rem_acc_{trecho['id']}_{idx}", help="Remover acessório"):
            trecho['acessorios'].pop(idx); st.rerun()
    c1, c2 = st.columns([3, 1]); c1.selectbox("Selecionar Acessório", options=list(K_FACTORS.keys()), key=f"selectbox_acessorio_{trecho['id']}"); c2.number_input("Qtd", min_value=1, value=1, step=1, key=f"quantidade_acessorio_{trecho['id']}")
    st.button("Adicionar Acessório", on_click=adicionar_acessorio, args=(trecho['id'], lista_trechos), key=f"btn_add_acessorio_{trecho['id']}", use_container_width=True)

def adicionar_item(tipo_lista):
    novo_id = time.time()
    st.session_state[tipo_lista].append({"id": novo_id, "nome": "", "comprimento": 10.0, "diametro": 100.0, "material": "Aço Carbono (novo)", "acessorios": [], "perda_equipamento_m": 0.0})

def remover_ultimo_item(tipo_lista):
    if len(st.session_state[tipo_lista]) > 0: st.session_state[tipo_lista].pop()

def adicionar_ramal_paralelo():
    novo_nome_ramal = f"Ramal {len(st.session_state.ramais_paralelos) + 1}"
    novo_id = time.time()
    st.session_state.ramais_paralelos[novo_nome_ramal] = [{"id": novo_id, "nome": "", "comprimento": 50.0, "diametro": 80.0, "material": "Aço Carbono (novo)", "acessorios": [], "perda_equipamento_m": 0.0}]

def remover_ultimo_ramal():
    if len(st.session_state.ramais_paralelos) > 1: st.session_state.ramais_paralelos.popitem()

def adicionar_acessorio(id_trecho, lista_trechos):
    nome_acessorio = st.session_state[f"selectbox_acessorio_{id_trecho}"]
    quantidade = st.session_state[f"quantidade_acessorio_{id_trecho}"]
    for trecho in lista_trechos:
        if trecho["id"] == id_trecho:
            trecho["acessorios"].append({"nome": nome_acessorio, "k": K_FACTORS[nome_acessorio], "quantidade": int(quantidade)})
            break

# --- INICIALIZAÇÃO E AUTENTICAÇÃO ---
setup_database()
with open('config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)
authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days']
)

authenticator.login()

# --- LÓGICA PRINCIPAL DA APLICAÇÃO ---
if st.session_state.get("authentication_status"):
    name = st.session_state['name']
    username = st.session_state['username']
    
    # Inicialização do st.session_state
    if 'trechos_succao' not in st.session_state: st.session_state.trechos_succao = []
    if 'trechos_antes' not in st.session_state: st.session_state.trechos_antes = []
    if 'trechos_depois' not in st.session_state: st.session_state.trechos_depois = []
    if 'ramais_paralelos' not in st.session_state: st.session_state.ramais_paralelos = {}
    if 'curva_altura_df' not in st.session_state:
        st.session_state.curva_altura_df = pd.DataFrame([{"Vazão (m³/h)": 0, "Altura (m)": 40}, {"Vazão (m³/h)": 50, "Altura (m)": 35}, {"Vazão (m³/h)": 100, "Altura (m)": 25}])
    if 'curva_eficiencia_df' not in st.session_state:
        st.session_state.curva_eficiencia_df = pd.DataFrame([{"Vazão (m³/h)": 0, "Eficiência (%)": 0}, {"Vazão (m³/h)": 50, "Eficiência (%)": 70}, {"Vazão (m³/h)": 100, "Eficiência (%)": 65}])
    if 'curva_npshr_df' not in st.session_state:
        st.session_state.curva_npshr_df = pd.DataFrame([{"Vazão (m³/h)": 0, "NPSHr (m)": 2}, {"Vazão (m³/h)": 50, "NPSHr (m)": 3}, {"Vazão (m³/h)": 100, "NPSHr (m)": 5}])
    if 'fluido_selecionado' not in st.session_state: st.session_state.fluido_selecionado = "Água a 20°C"
    if 'h_geometrica' not in st.session_state: st.session_state.h_geometrica = 15.0
    if 'endpoint_type' not in st.session_state: st.session_state.endpoint_type = "Atmosférico"
    if 'final_pressure' not in st.session_state: st.session_state.final_pressure = 0.0
    if 'altitude' not in st.session_state: st.session_state.altitude = 0.0
    if 'h_estatica_succao' not in st.session_state: st.session_state.h_estatica_succao = 2.0
    if 'suction_tank_type' not in st.session_state: st.session_state.suction_tank_type = "Atmosférico"
    if 'suction_tank_pressure' not in st.session_state: st.session_state.suction_tank_pressure = 0.0

    user_fluids = get_user_fluids(username)
    fluidos_combinados = {**FLUIDOS_PADRAO, **user_fluids}
    user_materials = get_user_materials(username)
    materiais_combinados = {**MATERIAIS_PADRAO, **user_materials}
    
    with st.sidebar:
        st.header(f"Bem-vindo(a), {name}!")
        st.divider()
        st.header("🚀 Gestão de Projetos e Cenários")
        
        user_projects = get_user_projects(username)
        project_idx = 0
        if st.session_state.get('project_to_select') in user_projects:
            project_idx = user_projects.index(st.session_state.get('project_to_select'))
            del st.session_state['project_to_select']
        elif st.session_state.get('selected_project') in user_projects:
            project_idx = user_projects.index(st.session_state.get('selected_project'))
        
        st.selectbox("Selecione o Projeto", user_projects, index=project_idx, key="selected_project", placeholder="Nenhum projeto encontrado")

        scenarios = []
        scenario_idx = 0
        if st.session_state.get("selected_project"):
            scenarios = get_scenarios_for_project(username, st.session_state.selected_project)
            if st.session_state.get('scenario_to_select') in scenarios:
                scenario_idx = scenarios.index(st.session_state.get('scenario_to_select'))
                del st.session_state['scenario_to_select']
            elif st.session_state.get('selected_scenario') in scenarios:
                scenario_idx = scenarios.index(st.session_state.get('selected_scenario'))

        st.selectbox("Selecione o Cenário", scenarios, index=scenario_idx, key="selected_scenario", placeholder="Nenhum cenário encontrado")
        
        col1, col2 = st.columns(2)
        if col1.button("Carregar Cenário", use_container_width=True, disabled=not st.session_state.get("selected_scenario")):
            data = load_scenario(username, st.session_state.selected_project, st.session_state.selected_scenario)
            if data:
                # Carregar todos os dados do cenário, incluindo os novos campos de NPSH
                st.session_state.h_geometrica = data.get('h_geometrica', 15.0)
                st.session_state.fluido_selecionado = data.get('fluido_selecionado', "Água a 20°C")
                st.session_state.endpoint_type = data.get('endpoint_type', 'Atmosférico')
                st.session_state.final_pressure = data.get('final_pressure', 0.0)
                st.session_state.curva_altura_df = pd.DataFrame(data['curva_altura'])
                st.session_state.curva_eficiencia_df = pd.DataFrame(data['curva_eficiencia'])
                st.session_state.trechos_succao = data.get('trechos_succao', [])
                st.session_state.trechos_antes = data.get('trechos_antes', [])
                st.session_state.trechos_depois = data.get('trechos_depois', [])
                st.session_state.ramais_paralelos = data.get('ramais_paralelos', {})
                # Carregar novos campos de NPSH
                st.session_state.curva_npshr_df = pd.DataFrame(data.get('curva_npshr', st.session_state.curva_npshr_df.to_dict('records')))
                st.session_state.altitude = data.get('altitude', 0.0)
                st.session_state.h_estatica_succao = data.get('h_estatica_succao', 2.0)
                st.session_state.suction_tank_type = data.get('suction_tank_type', "Atmosférico")
                st.session_state.suction_tank_pressure = data.get('suction_tank_pressure', 0.0)
                st.success(f"Cenário '{st.session_state.selected_scenario}' carregado.")
                st.rerun()
        if col2.button("Deletar Cenário", use_container_width=True, disabled=not st.session_state.get("selected_scenario")):
            delete_scenario(username, st.session_state.selected_project, st.session_state.selected_scenario)
            st.success(f"Cenário '{st.session_state.selected_scenario}' deletado.")
            st.rerun()

        st.divider()
        st.subheader("Salvar Cenário")
        project_name_input = st.text_input("Nome do Projeto", value=st.session_state.get("selected_project", ""))
        scenario_name_input = st.text_input("Nome do Cenário", value=st.session_state.get("selected_scenario", ""))
        if st.button("Salvar", use_container_width=True):
            if project_name_input and scenario_name_input:
                scenario_data = {
                    'h_geometrica': st.session_state.h_geometrica,
                    'endpoint_type': st.session_state.endpoint_type,
                    'final_pressure': st.session_state.final_pressure,
                    'fluido_selecionado': st.session_state.fluido_selecionado,
                    'curva_altura': st.session_state.curva_altura_df.to_dict('records'),
                    'curva_eficiencia': st.session_state.curva_eficiencia_df.to_dict('records'),
                    'trechos_succao': st.session_state.trechos_succao,
                    'trechos_antes': st.session_state.trechos_antes,
                    'trechos_depois': st.session_state.trechos_depois,
                    'ramais_paralelos': st.session_state.ramais_paralelos,
                    # Salvar novos campos de NPSH
                    'curva_npshr': st.session_state.curva_npshr_df.to_dict('records'),
                    'altitude': st.session_state.altitude,
                    'h_estatica_succao': st.session_state.h_estatica_succao,
                    'suction_tank_type': st.session_state.suction_tank_type,
                    'suction_tank_pressure': st.session_state.suction_tank_pressure,
                }
                save_scenario(username, project_name_input, scenario_name_input, scenario_data)
                st.success(f"Cenário '{scenario_name_input}' salvo.")
                st.session_state.project_to_select = project_name_input
                st.session_state.scenario_to_select = scenario_name_input
                st.rerun()
            else:
                st.warning("É necessário um nome para o Projeto e para o Cenário.")
        
        st.divider()
        authenticator.logout('Logout', 'sidebar')
        st.divider()

        with st.expander("📚 Gerenciador da Biblioteca"):
            st.subheader("Fluidos Customizados")
            with st.form("add_fluid_form", clear_on_submit=True):
                st.write("Adicionar novo fluido")
                new_fluid_name = st.text_input("Nome do Fluido")
                c1f, c2f, c3f = st.columns(3)
                new_fluid_density = c1f.number_input("Densidade (ρ) [kg/m³]", format="%.2f", min_value=0.0)
                new_fluid_viscosity = c2f.number_input("Viscosidade (ν) [m²/s]", format="%.4e", min_value=0.0)
                new_fluid_vapor_pressure = c3f.number_input("Pressão Vapor (kPa)", format="%.2f", min_value=0.0)
                submitted_fluid = st.form_submit_button("Adicionar Fluido")
                if submitted_fluid:
                    if new_fluid_name and new_fluid_density > 0 and new_fluid_viscosity > 0:
                        if add_user_fluid(username, new_fluid_name, new_fluid_density, new_fluid_viscosity, new_fluid_vapor_pressure):
                            st.success(f"Fluido '{new_fluid_name}' adicionado!")
                            st.rerun()
                        else:
                            st.error(f"Fluido '{new_fluid_name}' já existe.")
                    else:
                        st.warning("Preencha todos os campos do fluido com valores válidos.")
            if user_fluids:
                st.write("Fluidos Salvos:")
                fluids_df = pd.DataFrame.from_dict(user_fluids, orient='index').reset_index()
                fluids_df.columns = ['Nome', 'Densidade (ρ)', 'Viscosidade (ν)', 'Pressão Vapor (kPa)']
                st.dataframe(fluids_df, use_container_width=True, hide_index=True)
                fluid_to_delete = st.selectbox("Selecione um fluido para deletar", options=[""] + list(user_fluids.keys()))
                if st.button("Deletar Fluido", key="del_fluid"):
                    if fluid_to_delete:
                        delete_user_fluid(username, fluid_to_delete)
                        st.rerun()

            st.subheader("Materiais Customizados")
            with st.form("add_material_form", clear_on_submit=True):
                st.write("Adicionar novo material")
                new_material_name = st.text_input("Nome do Material")
                new_material_roughness = st.number_input("Rugosidade (ε) [mm]", format="%.4f", min_value=0.0)
                submitted_material = st.form_submit_button("Adicionar Material")
                if submitted_material:
                    if new_material_name and new_material_roughness >= 0:
                        if add_user_material(username, new_material_name, new_material_roughness):
                            st.success(f"Material '{new_material_name}' adicionado!")
                            st.rerun()
                        else:
                            st.error(f"Material '{new_material_name}' já existe.")
                    else:
                        st.warning("Preencha todos os campos do material com valores válidos.")
            if user_materials:
                st.write("Materiais Salvos:")
                materials_df = pd.DataFrame.from_dict(user_materials, orient='index', columns=['Rugosidade (ε)']).reset_index()
                materials_df.columns = ['Nome', 'Rugosidade (ε)']
                st.dataframe(materials_df, use_container_width=True, hide_index=True)
                material_to_delete = st.selectbox("Selecione um material para deletar", options=[""] + list(user_materials.keys()))
                if st.button("Deletar Material", key="del_mat"):
                    if material_to_delete:
                        delete_user_material(username, material_to_delete)
                        st.rerun()
        st.divider()

        st.header("⚙️ Parâmetros da Simulação")
        lista_fluidos = list(fluidos_combinados.keys())
        idx_fluido = 0
        if st.session_state.fluido_selecionado in lista_fluidos:
            idx_fluido = lista_fluidos.index(st.session_state.fluido_selecionado)
        st.session_state.fluido_selecionado = st.selectbox("Selecione o Fluido", lista_fluidos, index=idx_fluido)
        
        st.subheader("Parâmetros de Sucção (NPSH)")
        st.session_state.suction_tank_type = st.radio("Condição do Reservatório de Sucção", ["Atmosférico", "Pressurizado"], key="suction_tank_selector", index=["Atmosférico", "Pressurizado"].index(st.session_state.suction_tank_type))
        if st.session_state.suction_tank_type == "Atmosférico":
            st.session_state.altitude = st.number_input("Altitude Local (m)", value=st.session_state.altitude, min_value=0.0, format="%.1f")
        else:
            st.session_state.suction_tank_pressure = st.number_input("Pressão no Tanque de Sucção (kgf/cm²)", value=st.session_state.suction_tank_pressure, min_value=0.0, format="%.3f")
        st.session_state.h_estatica_succao = st.number_input("Altura Estática de Sucção (m)", value=st.session_state.h_estatica_succao, format="%.2f", help="Nível do fluido MENOS nível do eixo da bomba. Negativo se for sucção negativa.")

        st.subheader("Parâmetros de Recalque")
        st.session_state.h_geometrica = st.number_input("Altura Geométrica de Recalque (m)", 0.0, value=st.session_state.h_geometrica, help="Diferença de elevação entre o eixo da bomba e o ponto final.")
        st.session_state.endpoint_type = st.radio("Condição do Ponto Final", ["Atmosférico", "Pressurizado"], index=["Atmosférico", "Pressurizado"].index(st.session_state.endpoint_type), key="endpoint_type_selector")
        if st.session_state.endpoint_type == "Pressurizado":
            st.session_state.final_pressure = st.number_input("Pressão Final (kgf/cm²)", min_value=0.0, value=st.session_state.final_pressure, format="%.3f")

        st.divider()
        with st.expander("📈 Curva da Bomba", expanded=True):
            st.info("Insira pelo menos 3 pontos da curva de performance.")
            st.subheader("Curva de Altura"); st.session_state.curva_altura_df = st.data_editor(st.session_state.curva_altura_df, num_rows="dynamic", key="editor_altura")
            st.subheader("Curva de Eficiência"); st.session_state.curva_eficiencia_df = st.data_editor(st.session_state.curva_eficiencia_df, num_rows="dynamic", key="editor_eficiencia")
            st.subheader("Curva de NPSH Requerido"); st.session_state.curva_npshr_df = st.data_editor(st.session_state.curva_npshr_df, num_rows="dynamic", key="editor_npshr")

        st.divider(); st.header("🔧 Rede de Tubulação")
        with st.expander("1. Linha de Sucção (Trechos antes da Bomba)"):
            for i, trecho in enumerate(st.session_state.trechos_succao):
                if 'nome' not in trecho or not trecho.get('nome'):
                    trecho['nome'] = f"Trecho de Sucção {i+1}"
                with st.container(border=True): render_trecho_ui(trecho, f"succao_{i}", st.session_state.trechos_succao, materiais_combinados)
            c1, c2 = st.columns(2); c1.button("Adicionar Trecho (Sucção)", on_click=adicionar_item, args=("trechos_succao",), use_container_width=True); c2.button("Remover Trecho (Sucção)", on_click=remover_ultimo_item, args=("trechos_succao",), use_container_width=True)

        with st.expander("2. Linha de Recalque (Trechos após a Bomba)"):
            st.subheader("2.1. Trechos em Série (Antes da Divisão)")
            for i, trecho in enumerate(st.session_state.trechos_antes):
                if 'nome' not in trecho or not trecho.get('nome'):
                    trecho['nome'] = f"Recalque Primário {i+1}"
                with st.container(border=True): render_trecho_ui(trecho, f"antes_{i}", st.session_state.trechos_antes, materiais_combinados)
            c1, c2 = st.columns(2); c1.button("Adicionar Trecho (Antes)", on_click=adicionar_item, args=("trechos_antes",), use_container_width=True); c2.button("Remover Trecho (Antes)", on_click=remover_ultimo_item, args=("trechos_antes",), use_container_width=True)
            
            st.subheader("2.2. Ramais em Paralelo")
            for nome_ramal, trechos_ramal in st.session_state.ramais_paralelos.items():
                with st.container(border=True):
                    st.write(f"**{nome_ramal}**")
                    for i, trecho in enumerate(trechos_ramal):
                        if 'nome' not in trecho or not trecho.get('nome'):
                            trecho['nome'] = f"{nome_ramal} (T{i+1})"
                        render_trecho_ui(trecho, f"par_{nome_ramal}_{i}", trechos_ramal, materiais_combinados)
            c1, c2 = st.columns(2); c1.button("Adicionar Ramal Paralelo", on_click=adicionar_ramal_paralelo, use_container_width=True); c2.button("Remover Último Ramal", on_click=remover_ultimo_ramal, use_container_width=True, disabled=len(st.session_state.ramais_paralelos) < 2)
            
            st.subheader("2.3. Trechos em Série (Depois da Junção)")
            for i, trecho in enumerate(st.session_state.trechos_depois):
                if 'nome' not in trecho or not trecho.get('nome'):
                    trecho['nome'] = f"Recalque Final {i+1}"
                with st.container(border=True): render_trecho_ui(trecho, f"depois_{i}", st.session_state.trechos_depois, materiais_combinados)
            c1, c2 = st.columns(2); c1.button("Adicionar Trecho (Depois)", on_click=adicionar_item, args=("trechos_depois",), use_container_width=True); c2.button("Remover Trecho (Depois)", on_click=remover_ultimo_item, args=("trechos_depois",), use_container_width=True)
        
        st.divider(); st.header("🔌 Equipamentos e Custo"); rend_motor = st.slider("Eficiência do Motor (%)", 1, 100, 90); horas_por_dia = st.number_input("Horas por Dia", 1.0, 24.0, 8.0, 0.5); tarifa_energia = st.number_input("Custo da Energia (R$/kWh)", 0.10, 5.00, 0.75, 0.01, format="%.2f")

    # --- CORPO PRINCIPAL DA APLICAÇÃO ---
    st.title("💧 Análise de Redes de Bombeamento com Curva de Bomba")
    
    try:
        sistema_succao_atual = st.session_state.trechos_succao
        sistema_recalque_atual = {'antes': st.session_state.trechos_antes, 'paralelo': st.session_state.ramais_paralelos, 'depois': st.session_state.trechos_depois}
        
        func_curva_bomba = criar_funcao_curva(st.session_state.curva_altura_df, "Vazão (m³/h)", "Altura (m)")
        func_curva_eficiencia = criar_funcao_curva(st.session_state.curva_eficiencia_df, "Vazão (m³/h)", "Eficiência (%)")
        func_curva_npshr = criar_funcao_curva(st.session_state.curva_npshr_df, "Vazão (m³/h)", "NPSHr (m)")

        if func_curva_bomba is None or func_curva_eficiencia is None or func_curva_npshr is None:
            st.warning("Forneça pontos de dados suficientes para todas as curvas da bomba (Altura, Eficiência e NPSHr).")
            st.stop()
        
        rho_selecionado = fluidos_combinados[st.session_state.fluido_selecionado]['rho']
        h_pressao_final_m = 0
        if st.session_state.endpoint_type == "Pressurizado":
            h_pressao_final_m = converter_pressao_para_mca(st.session_state.final_pressure, 'kgf/cm2', rho_selecionado)
        
        h_estatica_total = st.session_state.h_geometrica + st.session_state.h_estatica_succao + h_pressao_final_m

        shutoff_head = func_curva_bomba(0)
        if shutoff_head < h_estatica_total:
            st.error(f"**Bomba Incompatível:** A altura máxima da bomba ({shutoff_head:.2f} m) é menor que a Altura Estática Total ({h_estatica_total:.2f} m).")
            st.stop()

        is_rede_vazia = not (sistema_succao_atual or any(sistema_recalque_atual.values()))
        if is_rede_vazia:
            st.warning("Adicione pelo menos um trecho à rede (sucção ou recalque) para realizar o cálculo.")
            st.stop()

        vazao_op, altura_op, func_curva_sistema = encontrar_ponto_operacao(
            sistema_succao_atual,
            sistema_recalque_atual,
            h_estatica_total,
            st.session_state.fluido_selecionado,
            func_curva_bomba,
            materiais_combinados,
            fluidos_combinados
        )
        
        if vazao_op is not None and altura_op is not None:
            eficiencia_op = func_curva_eficiencia(vazao_op)
            if eficiencia_op > 100: eficiencia_op = 100
            if eficiencia_op < 0: eficiencia_op = 0
            
            # --- CÁLCULO DE NPSH ---
            # 1. Pressão na superfície do líquido
            h_superficie_m = 0
            if st.session_state.suction_tank_type == "Atmosférico":
                h_superficie_m = calcular_pressao_atm_mca(st.session_state.altitude, rho_selecionado)
            else:
                h_superficie_m = converter_pressao_para_mca(st.session_state.suction_tank_pressure, 'kgf/cm2', rho_selecionado)
            # 2. Pressão de vapor
            pv_kpa = fluidos_combinados[st.session_state.fluido_selecionado]['pv_kpa']
            h_vapor_m = converter_pressao_para_mca(pv_kpa, 'kpa', rho_selecionado)
            # 3. Perda de carga na sucção
            perda_succao_op = calcular_perda_serie(sistema_succao_atual, vazao_op, st.session_state.fluido_selecionado, materiais_combinados, fluidos_combinados)
            # 4. NPSH Disponível
            npsha_op = h_superficie_m + st.session_state.h_estatica_succao - perda_succao_op - h_vapor_m
            # 5. NPSH Requerido
            npshr_op = func_curva_npshr(vazao_op)
            margem_npsh = npsha_op - npshr_op
            
            # --- RESULTADOS ---
            resultados_energia = calcular_analise_energetica(vazao_op, altura_op, eficiencia_op, rend_motor, horas_por_dia, tarifa_energia, st.session_state.fluido_selecionado, fluidos_combinados)
            
            st.header("📊 Resultados no Ponto de Operação")
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("Vazão de Operação", f"{vazao_op:.2f} m³/h")
            c2.metric("Altura de Operação", f"{altura_op:.2f} m")
            c3.metric("Eficiência da Bomba", f"{eficiencia_op:.1f} %")
            c4.metric("Margem NPSH", f"{margem_npsh:.2f} m")

            if margem_npsh <= 0:
                st.error(f"**ALERTA DE CAVITAÇÃO!** NPSH Disponível ({npsha_op:.2f} m) é menor ou igual ao Requerido ({npshr_op:.2f} m). Risco iminente de danos à bomba.")
            elif margem_npsh < 1.5:
                 st.warning(f"**Atenção:** Margem de NPSH ({margem_npsh:.2f} m) é baixa. Recomenda-se uma margem de segurança maior para evitar cavitação.")

            st.metric("Custo Anual", f"R$ {resultados_energia['custo_anual']:.2f}")
            st.divider()

            fig_curvas, ax_curvas = plt.subplots(figsize=(8.5, 5.5))
            
            label_ponto_op = f'Ponto de Operação ({vazao_op:.1f} m³/h, {altura_op:.1f} m)'
            
            max_vazao_curva = st.session_state.curva_altura_df['Vazão (m³/h)'].max()
            max_plot_vazao = max(vazao_op * 1.2, max_vazao_curva * 1.2) if vazao_op else max_vazao_curva * 1.2
            vazao_range = np.linspace(0, max_plot_vazao, 100)
            altura_bomba = func_curva_bomba(vazao_range)
            altura_sistema = [func_curva_sistema(q) if func_curva_sistema(q) < 1e10 else np.nan for q in vazao_range]
            ax_curvas.plot(vazao_range, altura_bomba, label='Curva da Bomba', color='royalblue', lw=2)
            ax_curvas.plot(vazao_range, altura_sistema, label='Curva do Sistema', color='seagreen', lw=2)
            ax_curvas.scatter(vazao_op, altura_op, color='red', s=100, zorder=5, label=label_ponto_op)
            ax_curvas.set_title("Curva da Bomba vs. Curva do Sistema")
            ax_curvas.set_xlabel("Vazão (m³/h)")
            ax_curvas.set_ylabel("Altura Manométrica (m)")
            ax_curvas.legend()
            ax_curvas.grid(True)
            
            st.header("📄 Exportar Relatório")
            params_data = {
                "Fluido Selecionado": st.session_state.fluido_selecionado,
                "Altura Geométrica Total (m)": f"{st.session_state.h_geometrica + st.session_state.h_estatica_succao:.2f}",
                "Condição Final": st.session_state.endpoint_type,
            }
            if st.session_state.endpoint_type == "Pressurizado":
                params_data["Pressão Final (kgf/cm²)"] = f"{st.session_state.final_pressure:.3f}"
                params_data["Altura de Pressão (m)"] = f"{h_pressao_final_m:.2f}"
            
            params_data.update({
                "Altura Estática Total (m)": f"{h_estatica_total:.2f}",
                "Horas de Operação por Dia": f"{horas_por_dia:.1f}",
                "Custo de Energia (R$/kWh)": f"{tarifa_energia:.2f}",
                "Eficiência do Motor (%)": f"{rend_motor:.1f}"
            })

            results_data = {
                "Potência Elétrica Consumida (kW)": f"{resultados_energia['potencia_eletrica_kW']:.2f}",
                "Custo Anual de Energia (R$)": f"{resultados_energia['custo_anual']:.2f}",
                "NPSH Disponível (m)": f"{npsha_op:.2f}",
                "NPSH Requerido (m)": f"{npshr_op:.2f}",
                "Margem de Segurança NPSH (m)": f"{margem_npsh:.2f}",
            }
            metrics_data = [
                ("Vazão (m³/h)", f"{vazao_op:.2f}"),
                ("Altura (m)", f"{altura_op:.2f}"),
                ("Eficiência Bomba (%)", f"{eficiencia_op:.1f}")
            ]
            
            _, distribuicao_vazao_op = calcular_perdas_paralelo(sistema_recalque_atual['paralelo'], vazao_op, st.session_state.fluido_selecionado, materiais_combinados, fluidos_combinados)
            diagrama_obj = gerar_diagrama_rede(sistema_succao_atual, sistema_recalque_atual, vazao_op, distribuicao_vazao_op if len(sistema_recalque_atual['paralelo']) >= 2 else {}, st.session_state.fluido_selecionado, materiais_combinados, fluidos_combinados)
            diagrama_bytes = diagrama_obj.pipe(format='png')

            chart_buffer = io.BytesIO()
            fig_curvas.savefig(chart_buffer, format='PNG', dpi=300, bbox_inches='tight')
            chart_buffer.seek(0)

            # Criar um novo dict para a rede completa para o relatório
            network_data_completa = {
                'succao': sistema_succao_atual,
                'recalque': sistema_recalque_atual
            }
            pdf_bytes = generate_report(
                project_name=st.session_state.get("selected_project", "N/A"),
                scenario_name=st.session_state.get("selected_scenario", "N/A"),
                params_data=params_data,
                results_data=results_data,
                metrics_data=metrics_data,
                network_data=network_data_completa, # Enviando a rede completa
                diagram_image_bytes=diagrama_bytes,
                chart_figure_bytes=chart_buffer.getvalue()
            )
            st.download_button(
                label="📥 Baixar Relatório em PDF",
                data=pdf_bytes,
                file_name=f"Relatorio_{st.session_state.get('selected_project', 'NovoProjeto')}_{st.session_state.get('selected_scenario', 'NovoCenario')}.pdf",
                mime="application/pdf"
            )
            
            st.divider()
            st.header("🗺️ Diagrama da Rede")
            st.graphviz_chart(diagrama_obj)
            st.divider()
            st.header("📈 Gráfico de Curvas: Bomba vs. Sistema")
            st.pyplot(fig_curvas)
            plt.close(fig_curvas)
            st.divider()
            st.header("📈 Análise de Sensibilidade de Custo por Diâmetro")
            escala_range = st.slider("Fator de Escala para Diâmetros (%)", 50, 200, (80, 120), key="sensibilidade_slider")
            params_equipamentos_sens = {'eficiencia_bomba_percent': eficiencia_op, 'eficiencia_motor_percent': rend_motor, 'horas_dia': horas_por_dia, 'custo_kwh': tarifa_energia, 'fluido_selecionado': st.session_state.fluido_selecionado}
            params_fixos_sens = {
                'vazao_op': vazao_op,
                'h_estatica_total': h_estatica_total,
                'fluido': st.session_state.fluido_selecionado,
                'equipamentos': params_equipamentos_sens,
                'materiais_combinados': materiais_combinados,
                'fluidos_combinados': fluidos_combinados
            }
            chart_data_sensibilidade = gerar_grafico_sensibilidade_diametro(sistema_succao_atual, sistema_recalque_atual, escala_range, **params_fixos_sens)
            st.line_chart(chart_data_sensibilidade.set_index('Fator de Escala nos Diâmetros (%)'))
        else:
            st.error("Não foi possível encontrar um ponto de operação. Verifique os parâmetros.")
    except Exception as e:
        st.error(f"Ocorreu um erro inesperado durante a execução. Detalhe: {str(e)}")

elif st.session_state.get("authentication_status") is False:
    st.error('Usuário/senha incorreto')
elif st.session_state.get("authentication_status") is None:
    st.title("Bem-vindo à Plataforma de Análise de Redes Hidráulicas")
    st.warning('Por favor, insira seu usuário e senha para começar.')
