import time
from fpdf import FPDF
from datetime import datetime
import io
from PIL import Image
import os

class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'Relatório de Análise de Rede Hidráulica', 0, 1, 'C')
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Página {self.page_no()}', 0, 0, 'C')

    def chapter_title(self, title):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, title, 0, 1, 'L')
        self.ln(4)

    def create_table(self, data, headers, title):
        self.chapter_title(title)
        self.set_font('Arial', 'B', 9)
        col_width = self.w / 2.2
        for i, header in enumerate(headers):
            self.cell(col_width, 7, header, 1, 0, 'C')
        self.ln()
        self.set_font('Arial', '', 9)
        for row_key, row_val in data.items():
            self.cell(col_width, 6, str(row_key), 1)
            self.cell(col_width, 6, str(row_val), 1)
            self.ln()
        self.ln(10)

    def write_network_details(self, segments, title):
        self.chapter_title(title)
        if not segments:
            self.set_font('Arial', 'I', 9)
            self.cell(0, 6, "Nenhum trecho definido nesta seção.")
            self.ln(10)
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
                self.cell(self.w / 4, 6, key, 1)
                self.cell(self.w / 4, 6, val, 1)
                self.ln()
            if trecho.get('acessorios'):
                self.set_font('Arial', 'B', 9)
                self.cell(0, 7, "Acessórios neste trecho:", 0, 1)
                self.cell(self.w / 2.5, 6, "Nome", 1)
                self.cell(self.w / 5, 6, "Quantidade", 1)
                self.cell(self.w / 5, 6, "Fator K (unitário)", 1)
                self.ln()
                self.set_font('Arial', '', 9)
                for acc in trecho['acessorios']:
                    self.cell(self.w / 2.5, 6, acc['nome'], 1)
                    self.cell(self.w / 5, 6, str(acc['quantidade']), 1)
                    self.cell(self.w / 5, 6, str(acc['k']), 1)
                    self.ln()
            self.ln(5)

    # *** MÉTODO DE IMAGEM CORRIGIDO E ROBUSTO (DO SEU CÓDIGO ANTIGO) ***
    def add_image_from_bytes(self, image_bytes, title):
        self.chapter_title(title)
        if not image_bytes:
            self.set_font('Arial', 'I', 9)
            self.cell(0, 6, "Imagem não pôde ser gerada.")
            self.ln(10)
            return

        temp_img_path = f"temp_image_{time.time()}.png"
        try:
            with open(temp_img_path, "wb") as f:
                f.write(image_bytes)
            
            img_pil = Image.open(temp_img_path)
            img_width, img_height = img_pil.size
            img_pil.close()

            max_page_width = self.w - 20 # Margens
            new_width = max_page_width
            new_height = img_height * (new_width / img_width)

            available_height = self.page_break_trigger - self.get_y() - 5
            if new_height > available_height:
                self.add_page()
            
            self.image(temp_img_path, x='C', w=new_width)
            self.ln(5)
        finally:
            if os.path.exists(temp_img_path):
                os.remove(temp_img_path)

def generate_report(project_name, scenario_name, params_data, results_data, metrics_data, network_data, diagram_image_bytes, chart_figure_bytes):
    pdf = PDF()
    pdf.add_page()

    pdf.set_font('Arial', 'B', 11); pdf.cell(40, 10, 'Projeto:'); pdf.set_font('Arial', '', 11); pdf.cell(0, 10, project_name); pdf.ln(5)
    pdf.set_font('Arial', 'B', 11); pdf.cell(40, 10, 'Cenário:'); pdf.set_font('Arial', '', 11); pdf.cell(0, 10, scenario_name); pdf.ln(5)
    pdf.set_font('Arial', 'B', 11); pdf.cell(40, 10, 'Data de Geração:'); pdf.set_font('Arial', '', 11); pdf.cell(0, 10, datetime.now().strftime("%d/%m/%Y %H:%M:%S")); pdf.ln(15)
    
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

    return pdf.output(dest='S')
