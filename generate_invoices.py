import os
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "dados_notas_fiscais")
os.makedirs(OUTPUT_DIR, exist_ok=True)

invoices_data = [
    {
        "filename": "NF_001_Gatinhos_Petshop.pdf",
        "numero": "000.001.101",
        "serie": "1",
        "data_emissao": "15/10/2026",
        "chave_acesso": "35261012345678000190550010000000011234567890",
        "emitente": {
            "razao_social": "Pet Shop Gatinho Mágico Ltda",
            "cnpj": "12.345.678/0001-90",
            "endereco": "Rua dos Felinos Fofos, 42 - Sala 1 - São Paulo/SP"
        },
        "destinatario": {
            "nome": "Gabriela Almeida",
            "cpf": "123.456.789-00",
            "endereco": "Av. Halloween das Bruxas, 1031 - São Paulo/SP"
        },
        "itens": [
            ("1", "Ração Especial Para Gatos Halloween 10kg", "1 UN", "R$ 180,00", "R$ 180,00"),
            ("2", "Arranhador Temático com Machado da Lizzie Borden", "1 UN", "R$ 250,00", "R$ 250,00"),
            ("3", "Petisco Sabor Abóbora Encantada 500g", "1 UN", "R$ 45,00", "R$ 45,00")
        ],
        "total": "R$ 475,00",
        "observacoes": "Mascote Rag aprovado. Entrega expressa para o Halloween."
    },
    {
        "filename": "NF_002_Servicos_Tech_Rag.pdf",
        "numero": "000.002.202",
        "serie": "1",
        "data_emissao": "20/10/2026",
        "chave_acesso": "35261098765432000121550010000000029876543210",
        "emitente": {
            "razao_social": "Rag Soluções de Inteligência Artificial Ltda",
            "cnpj": "98.765.432/0001-21",
            "endereco": "Alameda dos Algoritmos, 500 - Bloco B - Campinas/SP"
        },
        "destinatario": {
            "nome": "Tech Corp Inovações Educacionais S.A.",
            "cpf": "23.456.789/0001-88",
            "endereco": "Av. Paulista, 1000 - 15 andar - São Paulo/SP"
        },
        "itens": [
            ("1", "Consultoria em Arquitetura de Sistemas RAG Didáticos", "1 SRV", "R$ 4.500,00", "R$ 4.500,00"),
            ("2", "Implantação e Otimização de Banco Vetorial Supabase PGVector", "1 SRV", "R$ 3.200,00", "R$ 3.200,00")
        ],
        "total": "R$ 7.700,00",
        "observacoes": "Serviços técnicos especializados em IA Generativa e embeddings vetoriais com FastEmbed."
    },
    {
        "filename": "NF_003_Loja_Fantasias_Halloween.pdf",
        "numero": "000.003.303",
        "serie": "2",
        "data_emissao": "28/10/2026",
        "chave_acesso": "35261045678901000133550010000000034567890123",
        "emitente": {
            "razao_social": "Fantasias & Feitiços Artigos Festivos Ltda",
            "cnpj": "45.678.901/0001-33",
            "endereco": "Travessa da Meia-Noite, 31 - Curitiba/PR"
        },
        "destinatario": {
            "nome": "Gabriela Almeida",
            "cpf": "123.456.789-00",
            "endereco": "Av. Halloween das Bruxas, 1031 - São Paulo/SP"
        },
        "itens": [
            ("1", "Fantasia Pelúcia Gatinho Preto Encantado", "1 UN", "R$ 130,00", "R$ 130,00"),
            ("2", "Machado Cenográfico Colecionável Lizzie Borden em Espuma", "1 UN", "R$ 95,00", "R$ 95,00"),
            ("3", "Capa Gótica com Detalhes Roxo e Laranja Neon", "1 UN", "R$ 85,00", "R$ 85,00")
        ],
        "total": "R$ 310,00",
        "observacoes": "Artigos decorativos temáticos para sessão de fotos com gatos no Halloween."
    },
    {
        "filename": "NF_004_Equipamentos_Laboratorio.pdf",
        "numero": "000.004.404",
        "serie": "1",
        "data_emissao": "31/10/2026",
        "chave_acesso": "35261011222333000144550010000000041122233344",
        "emitente": {
            "razao_social": "Hardware & Servidores Avançados S.A.",
            "cnpj": "11.222.333/0001-44",
            "endereco": "Rodovia da Tecnologia, KM 12 - Florianópolis/SC"
        },
        "destinatario": {
            "nome": "Laboratório de IA Rag",
            "cpf": "98.765.432/0001-21",
            "endereco": "Alameda dos Algoritmos, 500 - Campinas/SP"
        },
        "itens": [
            ("1", "Placa Aceleradora de IA para Inferência Onnx/FastEmbed", "1 PC", "R$ 6.800,00", "R$ 6.800,00"),
            ("2", "Monitor Ultrawide 34 Polegadas Dark Mode Oled", "1 UN", "R$ 2.400,00", "R$ 2.400,00")
        ],
        "total": "R$ 9.200,00",
        "observacoes": "Equipamentos destinados a processamento e inferência local de embeddings vetoriais."
    }
]

def generate_pdf(doc_data):
    filepath = os.path.join(OUTPUT_DIR, doc_data["filename"])
    doc = SimpleDocTemplate(filepath, pagesize=letter, leftMargin=36, rightMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()

    header_style = ParagraphStyle(
        'HeaderStyle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=14,
        leading=18,
        textColor=colors.HexColor('#2c1e4a')
    )
    title_style = ParagraphStyle(
        'TitleStyle',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=14,
        textColor=colors.HexColor('#d9531e')
    )
    label_style = ParagraphStyle(
        'LabelStyle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9,
        leading=11,
        textColor=colors.HexColor('#333333')
    )
    body_style = ParagraphStyle(
        'BodyStyle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=12
    )

    elements = []

    # Cabeçalho da Nota Fiscal
    header_data = [
        [
            Paragraph(f"<b>DOCUMENTO AUXILIAR DA NOTA FISCAL (DANFE)</b><br/><b>Nº:</b> {doc_data['numero']} - <b>SÉRIE:</b> {doc_data['serie']}", header_style),
            Paragraph(f"<b>DATA DE EMISSÃO:</b> {doc_data['data_emissao']}<br/><b>CHAVE DE ACESSO:</b><br/>{doc_data['chave_acesso']}", body_style)
        ]
    ]
    t_header = Table(header_data, colWidths=[300, 240])
    t_header.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#f7f3fb')),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#7b42bc')),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('PADDING', (0,0), (-1,-1), 8),
    ]))
    elements.append(t_header)
    elements.append(Spacer(1, 10))

    # Emitente e Destinatário
    party_data = [
        [
            Paragraph("<b>EMITENTE</b>", title_style),
            Paragraph("<b>DESTINATÁRIO / CLIENTE</b>", title_style)
        ],
        [
            Paragraph(f"<b>Razão Social:</b> {doc_data['emitente']['razao_social']}<br/><b>CNPJ:</b> {doc_data['emitente']['cnpj']}<br/><b>Endereço:</b> {doc_data['emitente']['endereco']}", body_style),
            Paragraph(f"<b>Nome/Razão:</b> {doc_data['destinatario']['nome']}<br/><b>CPF/CNPJ:</b> {doc_data['destinatario']['cpf']}<br/><b>Endereço:</b> {doc_data['destinatario']['endereco']}", body_style)
        ]
    ]
    t_parties = Table(party_data, colWidths=[270, 270])
    t_parties.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#fafafa')),
        ('BOX', (0,0), (-1,-1), 0.5, colors.grey),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('PADDING', (0,0), (-1,-1), 6),
    ]))
    elements.append(t_parties)
    elements.append(Spacer(1, 10))

    # Tabela de Itens
    item_rows = [
        [
            Paragraph("<b>Item</b>", label_style),
            Paragraph("<b>Descrição do Produto / Serviço</b>", label_style),
            Paragraph("<b>Qtd / Un</b>", label_style),
            Paragraph("<b>Valor Unit.</b>", label_style),
            Paragraph("<b>Valor Total</b>", label_style)
        ]
    ]
    for it in doc_data["itens"]:
        item_rows.append([
            Paragraph(it[0], body_style),
            Paragraph(it[1], body_style),
            Paragraph(it[2], body_style),
            Paragraph(it[3], body_style),
            Paragraph(it[4], body_style)
        ])

    t_items = Table(item_rows, colWidths=[35, 275, 60, 85, 85])
    t_items.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#e8ddfc')),
        ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor('#7b42bc')),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#d6c7ee')),
        ('ALIGN', (0,0), (0,-1), 'CENTER'),
        ('ALIGN', (2,0), (-1,-1), 'CENTER'),
        ('PADDING', (0,0), (-1,-1), 5),
    ]))
    elements.append(t_items)
    elements.append(Spacer(1, 10))

    # Rodapé / Totais
    footer_data = [
        [
            Paragraph(f"<b>DADOS ADICIONAIS / INFORMAÇÕES COMPLEMENTARES:</b><br/>{doc_data['observacoes']}", body_style),
            Paragraph(f"<b>VALOR TOTAL DA NOTA:</b><br/><font size='14' color='#d9531e'><b>{doc_data['total']}</b></font>", title_style)
        ]
    ]
    t_footer = Table(footer_data, colWidths=[380, 160])
    t_footer.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#fff7f2')),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#d9531e')),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('PADDING', (0,0), (-1,-1), 8),
    ]))
    elements.append(t_footer)

    doc.build(elements)
    print(f"[+] Gerado PDF: {filepath}")

def main():
    print(f"Gerando {len(invoices_data)} Notas Fiscais ficticias em: {OUTPUT_DIR}")
    for item in invoices_data:
        generate_pdf(item)
    print("[+] Concluida a geracao dos 4 PDFs ficticios!")

if __name__ == "__main__":
    main()
