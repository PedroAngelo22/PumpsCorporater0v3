import time
from fpdf import FPDF
from datetime import datetime
import io
from PIL import Image
import os

class PDFReport(FPDF):
    def __init__(self, project_name, scenario_name, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.project_name = project_name
        self.scenario_name = scenario_name
        self.set_auto_page_break(auto=True, margin=20)

    def header(self):
        self.set_font('Arial', 'B', 16)
        self.cell(0, 10, 'Relatório de Análise Hidráulica', 0, 1, 'C')
        self.set_font('Arial', '', 10)
        self.cell(0, 6, f'Projeto: {self.project_name} | Cenário: {self.scenario_name}', 0, 1, 'C')
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Página {self.page_no()}', 0, 0, 'C')
        self.set_x(10)
        self.cell(0, 10, f'Gerado em: {datetime.now().strftime("%d/%m/%Y %H:%M:%S")}', 0, 0, 'L')

    def add_section_title(self, title):
        self.ln(5)
        self.set_font('Arial', 'B', 12)
        self.set_fill_color(230, 230, 230)
        if self.get_y() + 12 > self.page_break_trigger:
            self.add_page()
        self.cell(0, 8, title, 0, 1, 'L', fill=True)
        self.ln(4)

    def add_key_value_table(self, data_dict):
        self.set_font('Arial', '', 10)
        for key, value in data_dict.items():
            if self.get_y() + 8 > self.page_break_trigger:
                self.add_page()
            self.set_font('', 'B')
            self.cell(70, 8, f'{key}:', border=1)
            self.set_font('', '')
            self.cell(0, 8, f' {value}', border=1)
            self.ln()
        self.ln(5)

    def add_results_metrics(self, metrics_data):
        self.set_font('Arial', 'B', 11)
        num_metrics = len(metrics_data)
        cell_width = (self.w - 20) / num_metrics if num_metrics > 0 else (self.w - 20)
        
        for title, _ in metrics_data:
            self.cell(cell_width, 7, title, 1, 0, 'C')
        self.ln()

        self.set_font('', '')
        for _, value in metrics_data:
            self.cell(cell_width, 10, value, 1, 0, 'C')
        self.ln()
        self.ln(5)

    def add_network_summary_table(self, network_data):
        self.set_font('Arial', 'B', 10)
        
        # Cabeçalho da Tabela
        self.cell(70, 7, 'Trecho / Ramal', 1, 0, 'C')
        self.cell(20, 7, 'L (m)', 1, 0, 'C')
        self.cell(20, 7, 'Ø (mm)', 1, 0, 'C')
        self.cell(30, 7, 'Perda Equip. (m)', 1, 0, 'C')
        self.cell(50, 7, 'Material', 1, 1, 'C')
        
        self.set_font('Arial', '', 9)

        def draw_rows(title, trechos):
            if trechos:
                if self.get_y() + 15 > self.page_break_trigger: self.add_page()
                self.set_font('', 'B')
                self.cell(0, 7, title, 1, 1, 'L', fill=True)
                self.set_font('', '')
                for i, trecho in enumerate(trechos):
                    if self.get_y() + 7 > self.page_break_trigger:
                        self.add_page()
                        self.set_font('Arial', 'B', 10)
                        self.cell(70, 7, 'Trecho / Ramal', 1, 0, 'C'); self.cell(20, 7, 'L (m)', 1, 0, 'C'); self.cell(20, 7, 'Ø (mm)', 1, 0, 'C'); self.cell(30, 7, 'Perda Equip. (m)', 1, 0, 'C'); self.cell(50, 7, 'Material', 1, 1, 'C')
                        self.set_font('Arial', '', 9)

                    nome_trecho = trecho.get('nome', f'Trecho {i+1}')
                    perda_equip_val = trecho.get('perda_equipamento_m', 0.0)
                    
                    self.cell(70, 7, f'  - {nome_trecho}', 1, 0, 'L')
                    self.cell(20, 7, f"{trecho['comprimento']:.2f}", 1, 0, 'C')
                    self.cell(20, 7, f"{trecho['diametro']:.2f}", 1, 0, 'C')
                    self.cell(30, 7, f"{perda_equip_val:.2f}", 1, 0, 'C')
                    self.cell(50, 7, trecho['material'], 1, 1, 'L')

        # Desenha as seções da rede
        draw_rows('Linha de Sucção', network_data.get('succao', []))
        recalque = network_data.get('recalque', {})
        draw_rows('Recalque - Série (Antes da Divisão)', recalque.get('antes', []))
        
        if recalque.get('paralelo'):
            if self.get_y() + 15 > self.page_break_trigger: self.add_page()
            self.set_font('', 'B')
            self.cell(0, 7, 'Recalque - Ramais em Paralelo', 1, 1, 'L', fill=True)
            self.set_font('', '')
            for ramal_name, trechos_ramal in recalque['paralelo'].items():
                for i, trecho in enumerate(trechos_ramal):
                    if self.get_y() + 7 > self.page_break_trigger: self.add_page()
                    nome_trecho_ramal = trecho.get('nome', f'{ramal_name} (T{i+1})')
                    perda_equip_val = trecho.get('perda_equipamento_m', 0.0)
                    self.cell(70, 7, f'  - {nome_trecho_ramal}', 1, 0, 'L')
                    self.cell(20, 7, f"{trecho['comprimento']:.2f}", 1, 0, 'C')
                    self.cell(20, 7, f"{trecho['diametro']:.2f}", 1, 0, 'C')
                    self.cell(30, 7, f"{perda_equip_val:.2f}", 1, 0, 'C')
                    self.cell(50, 7, trecho['material'], 1, 1, 'L')

        draw_rows('Recalque - Série (Depois da Junção)', recalque.get('depois', []))
        self.ln(5)

    def add_image_from_bytes(self, image_bytes):
        if not image_bytes: return
        temp_img_path = f"temp_image_{time.time()}.png"
        try:
            with open(temp_img_path, "wb") as f:
                f.write(image_bytes)
            
            img_pil = Image.open(temp_img_path)
            img_width, img_height = img_pil.size
            img_pil.close()

            max_page_width = self.w - 20
            new_width = max_page_width
            new_height = img_height * (new_width / img_width)

            available_height = self.page_break_trigger - self.get_y() - 5
            if new_height > available_height:
                self.add_page()
            
            self.image(temp_img_path, x='C', w=new_width)
        finally:
            if os.path.exists(temp_img_path):
                os.remove(temp_img_path)

def generate_report(project_name, scenario_name, params_data, results_data, metrics_data, network_data, diagram_image_bytes, chart_figure_bytes):
    pdf = PDFReport(project_name, scenario_name)
    pdf.add_page()
    
    # 1. Parâmetros Gerais
    pdf.add_section_title('Parâmetros Gerais da Simulação')
    pdf.add_key_value_table(params_data)

    # 2. Resumo da Rede
    pdf.add_section_title('Resumo da Rede de Tubulação')
    pdf.add_network_summary_table(network_data)

    # 3. Diagrama da Rede
    pdf.add_section_title('Diagrama da Rede')
    pdf.add_image_from_bytes(diagram_image_bytes)
    
    # 4. Análise de NPSH (NOVA SEÇÃO)
    npsh_keys = ["NPSH Disponível (m)", "NPSH Requerido (m)", "Margem de Segurança NPSH (m)"]
    npsh_data = {k: results_data[k] for k in npsh_keys if k in results_data}
    cost_data = {k: v for k, v in results_data.items() if k not in npsh_keys}
    if npsh_data:
        pdf.add_section_title('Análise de NPSH')
        pdf.add_key_value_table(npsh_data)

    # 5. Resultados no Ponto de Operação
    pdf.add_section_title('Resultados no Ponto de Operação')
    pdf.add_results_metrics(metrics_data)
    
    # 6. Análise de Custo Energético
    if cost_data:
        pdf.add_section_title('Análise de Custo Energético')
        pdf.add_key_value_table(cost_data)

    # 7. Gráfico das Curvas
    pdf.add_section_title('Gráfico: Curva da Bomba vs. Curva do Sistema')
    pdf.add_image_from_bytes(chart_figure_bytes)
    
    return bytes(pdf.output())
