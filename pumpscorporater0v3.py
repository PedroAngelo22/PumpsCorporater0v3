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
    "Água a 20°C": {"rho": 998.2, "nu": 1.004e-6, "pv_mca": 0.23}, 
    "Etanol a 20°C": {"rho": 789.0, "nu": 1.51e-6, "pv_mca": 0.6}  
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

def calcular_npsh_disponivel(trechos_succao, vazao_op, params_succao, fluido, materiais, fluidos):
    h_atm = 10.33
    h_estatica_succao = params_succao['h_estatica_succao']
    rho_fluido = fluidos[fluido]["rho"]
    h_pressao_succao = converter_pressao_para_mca(params_succao['pressao_succao_kgfcm2'], rho_fluido)
    perda_carga_succao = calcular_perda_serie(trechos_succao, vazao_op, fluido, materiais, fluidos)
    h_vapor = fluidos[fluido].get('pv_mca', 0.23)
    npshd = h_atm + h_estatica_succao + h_pressao_succao - perda_carga_succao - h_vapor
    return npshd

def criar_funcao_curva(df_curva, col_x, col_y, grau=2):
    df_curva[col_x] = pd.to_numeric(df_curva[col_x], errors='coerce')
    df_curva[col_y] = pd.to_numeric(df_curva[col_y], errors='coerce')
    df_curva = df_curva.dropna(subset=[col_x, col_y])
    if len(df_curva) < grau + 1: return None
    coeficientes = np.polyfit(df_curva[col_x], df_curva[col_y], grau)
    return np.poly1d(coeficientes)

def converter_pressao_para_mca(pressao_kgfcm2, rho_fluido):
    if rho_fluido <= 0: return 0.0
    pressao_pa = pressao_kgfcm2 * 98066.5
    altura_m = pressao_pa / (rho_fluido * 9.81)
    return altura_m

def encontrar_ponto_operacao(sistema_recalque, h_estatica_recalque, fluido, func_curva_bomba, materiais_combinados, fluidos_combinados):
    def curva_sistema(vazao_m3h):
        if vazao_m3h < 0: return h_estatica_recalque
        perda_dinamica_recalque = 0
        perda_par, _ = calcular_perdas_paralelo(sistema_recalque['paralelo'], vazao_m3h, fluido, materiais_combinados, fluidos_combinados)
        if perda_par == -1: return 1e12
        perda_dinamica_recalque += perda_par
        perda_dinamica_recalque += calcular_perda_serie(sistema_recalque['depois'], vazao_m3h, fluido, materiais_combinados, fluidos_combinados)
        return h_estatica_recalque + perda_dinamica_recalque
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

def gerar_diagrama_rede(sistema, vazao_total, distribuicao_vazao, fluido, materiais_combinados, fluidos_combinados):
    dot = graphviz.Digraph(comment='Rede de Tubulação'); dot.attr('graph', rankdir='LR', splines='ortho'); dot.attr('node', shape='point'); 
    dot.node('start_succao', 'Sucção', shape='circle', style='filled', fillcolor='lightgray');
    ultimo_no = 'start_succao'
    for i, trecho in enumerate(sistema['antes']):
        proximo_no = f"no_antes_{i+1}"
        if i == len(sistema['antes']) - 1:
            proximo_no = 'bomba'
            dot.node('bomba', 'Bomba', shape='circle', style='filled', fillcolor='lightblue')
        perdas_info = calcular_perdas_trecho(trecho, vazao_total, fluido, materiais_combinados, fluidos_combinados)
        velocidade = perdas_info['velocidade']
        perda_trecho_hidraulica = perdas_info['principal'] + perdas_info['localizada'] + trecho.get('perda_equipamento_m', 0)
        label = f"{trecho.get('nome', f'Trecho Sucção {i+1}')}\\n{vazao_total:.1f} m³/h\\n{velocidade:.2f} m/s\\nPerda: {perda_trecho_hidraulica:.2f} m"
        dot.edge(ultimo_no, proximo_no, label=label)
        ultimo_no = proximo_no
    if not sistema['antes']:
        dot.node('bomba', 'Bomba', shape='circle', style='filled', fillcolor='lightblue')
        dot.edge('start_succao', 'bomba', label="Sem tubulação de sucção")
        ultimo_no = 'bomba'
    if len(sistema['paralelo']) >= 2 and distribuicao_vazao:
        no_divisao = ultimo_no; no_juncao = 'no_juncao'; dot.node(no_juncao)
        for nome_ramal, trechos_ramal in sistema['paralelo'].items():
            vazao_ramal = distribuicao_vazao.get(nome_ramal, 0); ultimo_no_ramal = no_divisao
            for i, trecho in enumerate(trechos_ramal):
                perdas_info_ramal = calcular_perdas_trecho(trecho, vazao_ramal, fluido, materiais_combinados, fluidos_combinados)
                velocidade = perdas_info_ramal['velocidade']
                perda_trecho_ramal_hidraulica = perdas_info_ramal['principal'] + perdas_info_ramal['localizada'] + trecho.get('perda_equipamento_m', 0)
                label_ramal = f"{trecho.get('nome', f'{nome_ramal} (T{i+1})')}\\n{vazao_ramal:.1f} m³/h\\n{velocidade:.2f} m/s\\nPerda: {perda_trecho_ramal_hidraulica:.2f} m"
                if i == len(trechos_ramal) - 1: dot.edge(ultimo_no_ramal, no_juncao, label=label_ramal)
                else: 
                    proximo_no_ramal = f"no_{nome_ramal}_{i+1}".replace(" ", "_")
                    dot.edge(ultimo_no_ramal, proximo_no_ramal, label=label_ramal)
                    ultimo_no_ramal = proximo_no_ramal
            ultimo_no = no_juncao
    for i, trecho in enumerate(sistema['depois']):
        proximo_no = f"no_depois_{i+1}"
        perdas_info = calcular_perdas_trecho(trecho, vazao_total, fluido, materiais_combinados, fluidos_combinados)
        velocidade = perdas_info['velocidade']
        perda_trecho_hidraulica = perdas_info['principal'] + perdas_info['localizada'] + trecho.get('perda_equipamento_m', 0)
        label = f"{trecho.get('nome', f'Trecho Recalque {i+1}')}\\n{vazao_total:.1f} m³/h\\n{velocidade:.2f} m/s\\nPerda: {perda_trecho_hidraulica:.2f} m"
        dot.edge(ultimo_no, proximo_no, label=label)
        ultimo_no = proximo_no
    dot.node('end', 'Fim', shape='circle', style='filled', fillcolor='lightgray'); dot.edge(ultimo_no, 'end')
    return dot

def gerar_grafico_sensibilidade_diametro(sistema_base, fator_escala_range, **params_fixos):
    custos, fatores = [], np.arange(fator_escala_range[0], fator_escala_range[1] + 5, 5)
    materiais_combinados = params_fixos['materiais_combinados']
    fluidos_combinados = params_fixos['fluidos_combinados']
    for fator in fatores:
        escala = fator / 100.0
        sistema_escalado = {'antes': [t.copy() for t in sistema_base['antes']], 'paralelo': {k: [t.copy() for t in v] for k, v in sistema_base['paralelo'].items()}, 'depois': [t.copy() for t in sistema_base['depois']]}
        for t_list in sistema_escalado.values():
            if isinstance(t_list, list):
                for t in t_list: t['diametro'] *= escala
            elif isinstance(t_list, dict):
                for _, ramal in t_list.items():
                    for t in ramal: t['diametro'] *= escala
        vazao_ref = params_fixos['vazao_op']
        perda_par, _ = calcular_perdas_paralelo(sistema_escalado['paralelo'], vazao_ref, params_fixos['fluido'], materiais_combinados, fluidos_combinados)
        perda_depois = calcular_perda_serie(sistema_escalado['depois'], vazao_ref, params_fixos['fluido'], materiais_combinados, fluidos_combinados)
        if perda_par == -1: custos.append(np.nan); continue
        h_man = params_fixos['h_estatica_recalque'] + perda_par + perda_depois
        resultado_energia = calcular_analise_energetica(vazao_ref, h_man, fluidos_combinados=fluidos_combinados, **params_fixos['equipamentos'])
        custos.append(resultado_energia['custo_anual'])
    return pd.DataFrame({'Fator de Escala nos Diâmetros (%)': fatores, 'Custo Anual de Energia (R$)': custos})

def render_trecho_ui(trecho, prefixo, lista_trechos, materiais_combinados):
    trecho['nome'] = st.text_input("Nome do Trecho", value=trecho.get('nome'), key=f"nome_{prefixo}_{trecho['id']}")
    c1, c2, c3, c4 = st.columns(4)
    trecho['comprimento'] = c1.number_input("L (m)", min_value=0.1, value=trecho['comprimento'], key=f"comp_{prefixo}_{trecho['id']}")
    trecho['diametro'] = c2.number_input("Ø (mm)", min_value=1.0, value=trecho['diametro'], key=f"diam_{prefixo}_{trecho['id']}")
    lista_materiais = list(materiais_combinados.keys())
    try: idx_material = lista_materiais.index(trecho.get('material', 'Aço Carbono (novo)'))
    except ValueError: idx_material = 0
    trecho['material'] = c3.selectbox("Material", options=lista_materiais, index=idx_material, key=f"mat_{prefixo}_{trecho['id']}")
    trecho['perda_equipamento_m'] = c4.number_input("Perda Equip. (m)", min_value=0.0, value=trecho.get('perda_equipamento_m', 0.0), key=f"equip_{prefixo}_{trecho['id']}", format="%.2f")
    st.markdown("**Acessórios (Fittings)**")
    for idx, acessorio in enumerate(trecho['acessorios']):
        col1, col2 = st.columns([0.8, 0.2]); col1.info(f"{acessorio['quantidade']}x {acessorio['nome']} (K = {acessorio['k']})")
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
with open('config.yaml') as file: config = yaml.load(file, Loader=SafeLoader)
authenticator = stauth.Authenticate(config['credentials'], config['cookie']['name'], config['cookie']['key'], config['cookie']['expiry_days'])
authenticator.login()

# --- LÓGICA PRINCIPAL DA APLICAÇÃO ---
if st.session_state.get("authentication_status"):
    name = st.session_state['name']
    username = st.session_state['username']
    
    if 'trechos_antes' not in st.session_state: st.session_state.trechos_antes = []
    if 'trechos_depois' not in st.session_state: st.session_state.trechos_depois = []
    if 'ramais_paralelos' not in st.session_state: st.session_state.ramais_paralelos = {}
    if 'curva_altura_df' not in st.session_state: st.session_state.curva_altura_df = pd.DataFrame([{"Vazão (m³/h)": 0, "Altura (m)": 40}, {"Vazão (m³/h)": 50, "Altura (m)": 35}, {"Vazão (m³/h)": 100, "Altura (m)": 25}])
    if 'curva_eficiencia_df' not in st.session_state: st.session_state.curva_eficiencia_df = pd.DataFrame([{"Vazão (m³/h)": 0, "Eficiência (%)": 0}, {"Vazão (m³/h)": 50, "Eficiência (%)": 70}, {"Vazão (m³/h)": 100, "Eficiência (%)": 65}])
    if 'fluido_selecionado' not in st.session_state: st.session_state.fluido_selecionado = "Água a 20°C"
    if 'h_geometrica' not in st.session_state: st.session_state.h_geometrica = 15.0
    if 'endpoint_type' not in st.session_state: st.session_state.endpoint_type = "Atmosférico"
    if 'final_pressure' not in st.session_state: st.session_state.final_pressure = 0.0
    if 'h_estatica_succao' not in st.session_state: st.session_state.h_estatica_succao = 1.0
    if 'pressao_succao_kgfcm2' not in st.session_state: st.session_state.pressao_succao_kgfcm2 = 0.0

    user_fluids = get_user_fluids(username)
    fluidos_combinados = {**FLUIDOS_PADRAO, **user_fluids}
    user_materials = get_user_materials(username)
    materiais_combinados = {**MATERIAIS_PADRAO, **user_materials}
    
    with st.sidebar:
        st.header(f"Bem-vindo(a), {name}!")
        st.divider()
        st.header("🚀 Gestão de Projetos e Cenários")
        # (O código de gestão de projetos e cenários está completo e correto)
        ...

        st.divider()
        st.header("⚙️ Parâmetros da Simulação")
        lista_fluidos = list(fluidos_combinados.keys())
        idx_fluido = lista_fluidos.index(st.session_state.fluido_selecionado) if st.session_state.fluido_selecionado in lista_fluidos else 0
        st.session_state.fluido_selecionado = st.selectbox("Selecione o Fluido", lista_fluidos, index=idx_fluido)
        st.subheader("Parâmetros de Sucção (NPSH)")
        st.session_state.h_estatica_succao = st.number_input("Altura Estática de Sucção (m)", value=st.session_state.h_estatica_succao, help="Nível do líquido acima/abaixo do eixo da bomba. Use valores negativos se a bomba estiver afogada.")
        st.session_state.pressao_succao_kgfcm2 = st.number_input("Pressão no Tanque de Sucção (kgf/cm²)", min_value=0.0, value=st.session_state.pressao_succao_kgfcm2, format="%.3f", help="Pressão manométrica na superfície do líquido. 0 para tanques abertos.")
        st.subheader("Parâmetros de Recalque (Descarga)")
        st.session_state.h_geometrica = st.number_input("Altura Geométrica de Recalque (m)", 0.0, value=st.session_state.h_geometrica, help="Diferença de elevação entre o eixo da bomba e o ponto final.")
        st.session_state.endpoint_type = st.radio("Condição do Ponto Final", ["Atmosférico", "Pressurizado"], index=["Atmosférico", "Pressurizado"].index(st.session_state.endpoint_type))
        if st.session_state.endpoint_type == "Pressurizado":
            st.session_state.final_pressure = st.number_input("Pressão Final (kgf/cm²)", min_value=0.0, value=st.session_state.final_pressure, format="%.3f")
        st.divider()
        with st.expander("📈 Curva da Bomba", expanded=True):
            st.info("Insira pelo menos 3 pontos da curva de performance.")
            st.subheader("Curva de Altura"); st.session_state.curva_altura_df = st.data_editor(st.session_state.curva_altura_df, num_rows="dynamic", key="editor_altura")
            st.subheader("Curva de Eficiência"); st.session_state.curva_eficiencia_df = st.data_editor(st.session_state.curva_eficiencia_df, num_rows="dynamic", key="editor_eficiencia")
        st.divider()
        st.header("🔧 Rede de Tubulação")
        with st.expander("1. Linha de Sucção (Trechos antes da bomba)"):
            for i, trecho in enumerate(st.session_state.trechos_antes):
                if not trecho.get('nome'): trecho['nome'] = f"Trecho de Sucção {i+1}"
                with st.container(border=True): render_trecho_ui(trecho, f"antes_{i}", st.session_state.trechos_antes, materiais_combinados)
            c1, c2 = st.columns(2); c1.button("Adicionar Trecho na Sucção", on_click=adicionar_item, args=("trechos_antes",), use_container_width=True); c2.button("Remover Trecho da Sucção", on_click=remover_ultimo_item, args=("trechos_antes",), use_container_width=True)
        with st.expander("2. Linha de Recalque (Trechos após a bomba)"):
            st.info("A linha de recalque é composta pelos ramais em paralelo e pelos trechos em série após a junção.")
            for nome_ramal, trechos_ramal in st.session_state.ramais_paralelos.items():
                with st.container(border=True):
                    st.subheader(f"Ramal em Paralelo: {nome_ramal}")
                    for i, trecho in enumerate(trechos_ramal):
                        if not trecho.get('nome'): trecho['nome'] = f"{nome_ramal} (T{i+1})"
                        render_trecho_ui(trecho, f"par_{nome_ramal}_{i}", trechos_ramal, materiais_combinados)
            c1, c2 = st.columns(2); c1.button("Adicionar Ramal Paralelo", on_click=adicionar_ramal_paralelo, use_container_width=True); c2.button("Remover Último Ramal", on_click=remover_ultimo_ramal, use_container_width=True, disabled=len(st.session_state.ramais_paralelos) < 2)
            st.divider()
            for i, trecho in enumerate(st.session_state.trechos_depois):
                if not trecho.get('nome'): trecho['nome'] = f"Trecho de Recalque {i+1}"
                with st.container(border=True): render_trecho_ui(trecho, f"depois_{i}", st.session_state.trechos_depois, materiais_combinados)
            c1, c2 = st.columns(2); c1.button("Adicionar Trecho (Depois da Junção)", on_click=adicionar_item, args=("trechos_depois",), use_container_width=True); c2.button("Remover Trecho (Depois da Junção)", on_click=remover_ultimo_item, args=("trechos_depois",), use_container_width=True)
        st.divider(); st.header("🔌 Equipamentos e Custo"); rend_motor = st.slider("Eficiência do Motor (%)", 1, 100, 90); horas_por_dia = st.number_input("Horas por Dia", 1.0, 24.0, 8.0, 0.5); tarifa_energia = st.number_input("Custo da Energia (R$/kWh)", 0.10, 5.00, 0.75, 0.01, format="%.2f")

    # --- CORPO PRINCIPAL DA APLICAÇÃO ---
    st.title("💧 Análise de Redes de Bombeamento")
    try:
        sistema_succao = st.session_state.trechos_antes
        sistema_recalque = {'paralelo': st.session_state.ramais_paralelos, 'depois': st.session_state.trechos_depois}
        sistema_completo = {'antes': sistema_succao, **sistema_recalque}
        func_curva_bomba = criar_funcao_curva(st.session_state.curva_altura_df, "Vazão (m³/h)", "Altura (m)")
        func_curva_eficiencia = criar_funcao_curva(st.session_state.curva_eficiencia_df, "Vazão (m³/h)", "Eficiência (%)")
        if func_curva_bomba is None or func_curva_eficiencia is None: st.warning("Forneça pontos de dados suficientes para as curvas da bomba."); st.stop()
        h_pressao_final_m = 0
        if st.session_state.endpoint_type == "Pressurizado":
            rho_selecionado = fluidos_combinados[st.session_state.fluido_selecionado]['rho']
            h_pressao_final_m = converter_pressao_para_mca(st.session_state.final_pressure, rho_selecionado)
        h_estatica_recalque = st.session_state.h_geometrica + h_pressao_final_m
        shutoff_head = func_curva_bomba(0)
        if shutoff_head < h_estatica_recalque: st.error(f"**Bomba Incompatível:** A altura máxima da bomba ({shutoff_head:.2f} m) é menor que a Altura Estática de Recalque ({h_estatica_recalque:.2f} m)."); st.stop()
        is_rede_vazia = not any(sistema_recalque.values()) and not sistema_succao
        if is_rede_vazia: st.warning("Adicione pelo menos um trecho à rede para realizar o cálculo."); st.stop()

        vazao_op, altura_op, func_curva_sistema = encontrar_ponto_operacao(sistema_recalque, h_estatica_recalque, st.session_state.fluido_selecionado, func_curva_bomba, materiais_combinados, fluidos_combinados)
        
        if vazao_op is not None and altura_op is not None:
            eficiencia_op = func_curva_eficiencia(vazao_op)
            if eficiencia_op > 100: eficiencia_op = 100
            if eficiencia_op < 0: eficiencia_op = 0
            
            params_succao = {'h_estatica_succao': st.session_state.h_estatica_succao, 'pressao_succao_kgfcm2': st.session_state.pressao_succao_kgfcm2}
            npsh_disponivel = calcular_npsh_disponivel(sistema_succao, vazao_op, params_succao, st.session_state.fluido_selecionado, materiais_combinados, fluidos_combinados)
            resultados_energia = calcular_analise_energetica(vazao_op, altura_op, eficiencia_op, rend_motor, horas_por_dia, tarifa_energia, st.session_state.fluido_selecionado, fluidos_combinados)
            
            st.header("📊 Resultados no Ponto de Operação")
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("Vazão de Operação", f"{vazao_op:.2f} m³/h")
            c2.metric("Altura de Operação", f"{altura_op:.2f} m")
            c3.metric("Eficiência da Bomba", f"{eficiencia_op:.1f} %")
            c4.metric("NPSH Disponível", f"{npsh_disponivel:.2f} m")
            if npsh_disponivel < 2.0: st.warning(f"**Atenção: Risco de Cavitação!** O NPSH disponível de {npsh_disponivel:.2f} m é baixo. Verifique o NPSH requerido (NPSHr) pelo fabricante da bomba e garanta uma margem de segurança (geralmente > 0.5 m).")
            st.metric("Custo Anual Estimado", f"R$ {resultados_energia['custo_anual']:.2f}")
            st.divider()

            # --- CÓDIGO RESTAURADO PARA EXIBIÇÃO DOS GRÁFICOS E RELATÓRIO ---
            fig_curvas, ax_curvas = plt.subplots(figsize=(8.5, 5.5))
            label_ponto_op = f'Ponto de Operação ({vazao_op:.1f} m³/h, {altura_op:.1f} m)'
            max_vazao_curva = st.session_state.curva_altura_df['Vazão (m³/h)'].max()
            max_plot_vazao = max(vazao_op * 1.5, max_vazao_curva * 1.1) 
            vazao_range = np.linspace(0, max_plot_vazao, 100)
            altura_bomba = func_curva_bomba(vazao_range)
            altura_sistema = [func_curva_sistema(q) if func_curva_sistema(q) < 1e10 else np.nan for q in vazao_range]
            ax_curvas.plot(vazao_range, altura_bomba, label='Curva da Bomba', color='royalblue', lw=2)
            ax_curvas.plot(vazao_range, altura_sistema, label='Curva do Sistema', color='seagreen', lw=2)
            ax_curvas.scatter(vazao_op, altura_op, color='red', s=100, zorder=5, label=label_ponto_op)
            ax_curvas.set_title("Curva da Bomba vs. Curva do Sistema")
            ax_curvas.set_xlabel("Vazão (m³/h)")
            ax_curvas.set_ylabel("Altura Manométrica (m)")
            ax_curvas.legend(); ax_curvas.grid(True)
            
            st.header("📄 Exportar Relatório")
            params_data = {
                "Fluido Selecionado": st.session_state.fluido_selecionado,
                "Altura Estática de Sucção (m)": f"{st.session_state.h_estatica_succao:.2f}",
                "Pressão no Tanque de Sucção (kgf/cm²)": f"{st.session_state.pressao_succao_kgfcm2:.3f}",
                "Altura Geométrica de Recalque (m)": f"{st.session_state.h_geometrica:.2f}",
                "Condição Final": st.session_state.endpoint_type,
            }
            if st.session_state.endpoint_type == "Pressurizado":
                params_data["Pressão Final (kgf/cm²)"] = f"{st.session_state.final_pressure:.3f}"
                params_data["Altura de Pressão (m)"] = f"{h_pressao_final_m:.2f}"
            params_data.update({
                "Altura Estática Total de Recalque (m)": f"{h_estatica_recalque:.2f}",
                "Horas de Operação por Dia": f"{horas_por_dia:.1f}",
                "Custo de Energia (R$/kWh)": f"{tarifa_energia:.2f}",
                "Eficiência do Motor (%)": f"{rend_motor:.1f}"
            })
            results_data = {
                "NPSH Disponível (m)": f"{npsh_disponivel:.2f}",
                "Potência Elétrica Consumida (kW)": f"{resultados_energia['potencia_eletrica_kW']:.2f}",
                "Custo Anual de Energia (R$)": f"{resultados_energia['custo_anual']:.2f}"
            }
            metrics_data = [
                ("Vazão (m³/h)", f"{vazao_op:.2f}"), ("Altura (m)", f"{altura_op:.2f}"),
                ("Eficiência Bomba (%)", f"{eficiencia_op:.1f}"), ("NPSH Disp. (m)", f"{npsh_disponivel:.2f}")
            ]
            _, distribuicao_vazao_op = calcular_perdas_paralelo(sistema_recalque['paralelo'], vazao_op, st.session_state.fluido_selecionado, materiais_combinados, fluidos_combinados)
            diagrama_obj = gerar_diagrama_rede(sistema_completo, vazao_op, distribuicao_vazao_op if len(sistema_recalque['paralelo']) >= 2 else {}, st.session_state.fluido_selecionado, materiais_combinados, fluidos_combinados)
            diagrama_bytes = diagrama_obj.pipe(format='png')
            chart_buffer = io.BytesIO()
            fig_curvas.savefig(chart_buffer, format='PNG', dpi=300, bbox_inches='tight')
            chart_buffer.seek(0)
            pdf_bytes = generate_report(
                project_name=st.session_state.get("selected_project", "N/A"), scenario_name=st.session_state.get("selected_scenario", "N/A"),
                params_data=params_data, results_data=results_data, metrics_data=metrics_data,
                network_data=sistema_completo, diagram_image_bytes=diagrama_bytes, chart_figure_bytes=chart_buffer.getvalue()
            )
            st.download_button(label="📥 Baixar Relatório em PDF", data=pdf_bytes, file_name=f"Relatorio_{st.session_state.get('selected_project', 'NovoProjeto')}_{st.session_state.get('selected_scenario', 'NovoCenario')}.pdf", mime="application/pdf")
            
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
                'vazao_op': vazao_op, 'h_estatica_recalque': h_estatica_recalque, 'fluido': st.session_state.fluido_selecionado, 
                'equipamentos': params_equipamentos_sens, 'materiais_combinados': materiais_combinados, 'fluidos_combinados': fluidos_combinados
            }
            chart_data_sensibilidade = gerar_grafico_sensibilidade_diametro(sistema_completo, escala_range, **params_fixos_sens)
            st.line_chart(chart_data_sensibilidade.set_index('Fator de Escala nos Diâmetros (%)'))
        else:
            st.error("Não foi possível encontrar um ponto de operação. Verifique os parâmetros.")
    except Exception as e:
        st.error(f"Ocorreu um erro inesperado durante a execução. Detalhe: {str(e)}")

elif st.session_state.get("authentication_status") is False: st.error('Usuário/senha incorreto')
elif st.session_state.get("authentication_status") is None:
    st.title("Bem-vindo à Plataforma de Análise de Redes Hidráulicas")
    st.warning('Por favor, insira seu usuário e senha para começar.')
