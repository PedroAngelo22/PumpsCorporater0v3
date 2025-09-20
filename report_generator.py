import time
from fpdf import FPDF
from datetime import datetime
import io
from PIL import Image
import os

class PDF(FPDF):
    def __init__(self, project_name, scenario_name, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.project_name = project_name
        self.scenario_name = scenario_name
        self.set_auto_page_break(auto=True, margin=20)

    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'Relatório de Análise de Rede Hidráulica', 0, 1, 'C')
        self.set_font('Arial', '', 10)
        self.cell(0, 6, f'Projeto: {self.project_name} | Cenário: {self.scenario_name}', 0, 1, 'C')
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Página {self.page_no()}', 0, 0, 'C')
        self.set_x(10)
        self.cell(0, 10, f'Gerado em: {datetime.now().strftime("%d/%m/%Y %H:%M:%S")}', 0, 0, 'L')

    def chapter_title(self, title):
        self.set_font('Arial', 'B', 12)
        self.set_fill_color(230, 230, 230)
        self.cell(0, 8, title, 0, 1, 'L', fill=True)
        self.ln(4)

    def create_table(self, data, headers, title):
        self.chapter_title(title)
        self.set_font('Arial', 'B', 9)
        col_width = (self.w - 20) / 2
        for header in headers:
            self.cell(col_width, 7, header, 1, 0, 'C')
        self.ln()
        self.set_font('Arial', '', 9)
        for row_key, row_val in data.items():
            self.cell(col_width, 6, str(row_key), 1)
            self.cell(col_width, 6, str(row_val), 1)
            self.ln()
        self.ln(5)

    def write_network_details(self, segments, title):
        self.chapter_title(title)
        if not segments:
            self.set_font('Arial', 'I', 9)
            self.cell(0, 6, "Nenhum trecho definido nesta seção.")
            self.ln(5)
            return

        for i, trecho in enumerate(segments):
            self.set_font('Arial', 'B', 10)
            self.cell(0, 8, f"Trecho {i+1}: {trecho.get('nome', 'N/A')}", 0, 1)
            data = {
                "Comprimento (m)": f"{trecho.get('comprimento', 0):.2f}",
                "Diâmetro (mm)": f"{trecho.get('diametro', 0):.2f}",
                "Material": trecho.get('material', 'N/A'),
                "Perda em Equip. (m)": f"{trecho.get('perda_equipamento_m', 0):.2f}"
            }
            self.set_font('Arial', '', 9)
            for key, val in data.items():
                self.cell(45, 6, key, 1)
                self.cell(50, 6, val, 1)
                self.ln()
            if trecho.get('acessorios'):
                self.set_font('Arial', 'B', 9)
                self.cell(0, 7, "Acessórios neste trecho:", 0, 1)
                self.cell(95, 6, "Nome", 1)
                self.cell(25, 6, "Quantidade", 1)
                self.cell(25, 6, "Fator K", 1)
                self.ln()
                self.set_font('Arial', '', 9)
                for acc in trecho['acessorios']:
                    self.cell(95, 6, acc['nome'], 1)
                    self.cell(25, 6, str(acc['quantidade']), 1, 0, 'C')
                    self.cell(25, 6, str(acc['k']), 1, 0, 'C')
                    self.ln()
            self.ln(5)

    def add_image_from_bytes(self, image_bytes, title):
        self.chapter_title(title)
        if not image_bytes:
            self.set_font('Arial', 'I', 9)
            self.cell(0, 6, "Imagem não pôde ser gerada.")
            self.ln(5)
            return

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
    pdf = PDF(project_name, scenario_name)
    pdf.add_page()
    
    pdf.create_table(dict(metrics_data), ["Ponto de Operação", "Valor"], "Resumo do Ponto de Operação")
    pdf.create_table(results_data, ["Resultados Principais", "Valor"], "Resultados de Performance e Segurança")
    pdf.create_table(params_data, ["Parâmetros de Entrada", "Valor"], "Parâmetros da Simulação")

    pdf.add_page()
    pdf.chapter_title("Detalhamento da Rede de Tubulação")
    pdf.write_network_details(network_data.get('succao', []), "1. Linha de Sucção")
    recalque = network_data.get('recalque', {})
    pdf.write_network_details(recalque.get('antes', []), "2.1. Linha de Recalque (Antes da Divisão)")
    if recalque.get('paralelo'):
        pdf.chapter_title("2.2. Linha de Recalque (Ramais em Paralelo)")
        for nome_ramal, trechos_ramal in recalque['paralelo'].items():
            pdf.write_network_details(trechos_ramal, f"Ramal: {nome_ramal}")
    pdf.write_network_details(recalque.get('depois', []), "2.3. Linha de Recalque (Depois da Junção)")

    pdf.add_page()
    pdf.add_image_from_bytes(chart_figure_bytes, "Gráfico de Curvas: Bomba vs. Sistema")
    pdf.add_image_from_bytes(diagram_image_bytes, "Diagrama da Rede")

    # *** USANDO O MÉTODO DE RETORNO DO SEU CÓDIGO ORIGINAL E FUNCIONAL ***
    return bytes(pdf.output())
