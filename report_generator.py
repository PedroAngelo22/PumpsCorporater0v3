from fpdf import FPDF
from datetime import datetime
import io
from PIL import Image

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
        # Larguras das colunas
        col_width = self.w / 2.2
        # Header
        for i, header in enumerate(headers):
            self.cell(col_width, 7, header, 1, 0, 'C')
        self.ln()
        # Dados
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
            self.ln()
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

            # Tabela de acessórios
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

def generate_report(project_name, scenario_name, params_data, results_data, metrics_data, network_data, diagram_image_bytes, chart_figure_bytes):
    pdf = PDF()
    pdf.add_page()

    # Informações do Projeto e Cenário
    pdf.set_font('Arial', 'B', 11)
    pdf.cell(40, 10, 'Projeto:')
    pdf.set_font('Arial', '', 11)
    pdf.cell(0, 10, project_name)
    pdf.ln(5)
    pdf.set_font('Arial', 'B', 11)
    pdf.cell(40, 10, 'Cenário:')
    pdf.set_font('Arial', '', 11)
    pdf.cell(0, 10, scenario_name)
    pdf.ln(5)
    pdf.set_font('Arial', 'B', 11)
    pdf.cell(40, 10, 'Data de Geração:')
    pdf.set_font('Arial', '', 11)
    pdf.cell(0, 10, datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
    pdf.ln(15)

    # Tabelas de Resumo
    pdf.create_table(dict(metrics_data), ["Ponto de Operação", "Valor"], "Resumo do Ponto de Operação")
    pdf.create_table(results_data, ["Resultados Principais", "Valor"], "Resultados de Performance e Segurança")
    pdf.create_table(params_data, ["Parâmetros de Entrada", "Valor"], "Parâmetros da Simulação")

    # Detalhamento da Rede
    pdf.add_page()
    pdf.chapter_title("Detalhamento da Rede de Tubulação")

    # Detalhes da Linha de Sucção
    pdf.write_network_details(network_data['succao'], "1. Linha de Sucção")

    # Detalhes da Linha de Recalque
    recalque = network_data['recalque']
    pdf.write_network_details(recalque['antes'], "2.1. Linha de Recalque (Antes da Divisão)")
    
    if recalque.get('paralelo'):
        pdf.chapter_title("2.2. Linha de Recalque (Ramais em Paralelo)")
        for nome_ramal, trechos_ramal in recalque['paralelo'].items():
            pdf.write_network_details(trechos_ramal, f"Ramal: {nome_ramal}")

    pdf.write_network_details(recalque['depois'], "2.3. Linha de Recalque (Depois da Junção)")

    # Imagens
    pdf.add_page()
    
    # Gráfico de Curvas
    pdf.chapter_title("Gráfico de Curvas: Bomba vs. Sistema")
    try:
        chart_image = Image.open(io.BytesIO(chart_figure_bytes))
        # Redimensionar mantendo a proporção
        w, h = chart_image.size
        aspect_ratio = h / w
        new_w = pdf.w - 20 # Largura da página com margens
        new_h = new_w * aspect_ratio
        pdf.image(io.BytesIO(chart_figure_bytes), x=10, y=None, w=new_w, h=new_h)
        pdf.ln(5)
    except Exception as e:
        pdf.set_text_color(255, 0, 0)
        pdf.cell(0, 10, f"Erro ao renderizar imagem do gráfico: {e}")
        pdf.set_text_color(0, 0, 0)
        pdf.ln()

    # Diagrama da Rede
    pdf.chapter_title("Diagrama da Rede")
    try:
        diagram_image = Image.open(io.BytesIO(diagram_image_bytes))
        w, h = diagram_image.size
        aspect_ratio = h / w
        new_w = pdf.w - 20
        new_h = new_w * aspect_ratio
        pdf.image(io.BytesIO(diagram_image_bytes), x=10, y=None, w=new_w, h=new_h)
    except Exception as e:
        pdf.set_text_color(255, 0, 0)
        pdf.cell(0, 10, f"Erro ao renderizar imagem do diagrama: {e}")
        pdf.set_text_color(0, 0, 0)
        pdf.ln()

    # Retorna os bytes do PDF para o Streamlit
    return pdf.output(dest='S').encode('latin-1')
