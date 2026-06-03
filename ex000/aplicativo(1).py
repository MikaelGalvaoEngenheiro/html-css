from flask import Flask, request, jsonify, render_template_string
import pickle
import pandas as pd

app = Flask(__name__)

# ==========================================
# CARREGAR MODELO DIRETAMENTE
# ==========================================
# Como seu arquivo .pkl contém apenas o KNN, lemos diretamente na variável
with open('modelo_knn.pkl', 'rb') as f:
    modelo = pickle.load(f)

# ==========================================
# DICIONÁRIO DAS 30 VARIÁVEIS
# ==========================================
DICIONARIO_VARIAVEIS = {
    "radius_mean": "Raio (Média)", "texture_mean": "Textura (Média)", "perimeter_mean": "Perímetro (Média)", "area_mean": "Área (Média)", "smoothness_mean": "Suavidade (Média)",
    "compactness_mean": "Compacidade (Média)", "concavity_mean": "Concavidade (Média)", "concave points_mean": "Pontos Côncavos (Média)", "symmetry_mean": "Simetria (Média)", "fractal_dimension_mean": "Dimensão Fractal (Média)",
    "radius_se": "Raio (Erro Padrão)", "texture_se": "Textura (Erro Padrão)", "perimeter_se": "Perímetro (Erro Padrão)", "area_se": "Área (Erro Padrão)", "smoothness_se": "Suavidade (Erro Padrão)",
    "compactness_se": "Compacidade (Erro Padrão)", "concavity_se": "Concavidade (Erro Padrão)", "concave points_se": "Pontos Côncavos (Erro Padrão)", "symmetry_se": "Simetria (Erro Padrão)", "fractal_dimension_se": "Dimensão Fractal (Erro Padrão)",
    "radius_worst": "Raio (Pior Cenário)", "texture_worst": "Textura (Pior Cenário)", "perimeter_worst": "Perímetro (Pior Cenário)", "area_worst": "Área (Pior Cenário)", "smoothness_worst": "Suavidade (Pior Cenário)",
    "compactness_worst": "Compacidade (Pior Cenário)", "concavity_worst": "Concavidade (Pior Cenário)", "concave points_worst": "Pontos Côncavos (Pior Cenário)", "symmetry_worst": "Simetria (Pior Cenário)", "fractal_dimension_worst": "Dimensão Fractal (Pior Cenário)"
}

COLUNAS = list(DICIONARIO_VARIAVEIS.keys())

# ==========================================
# TEMPLATE HTML
# ==========================================
HTML = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Predição de Câncer de Mama</title>
    <style>
        body{ font-family: Arial; background:#f4f4f4; text-align:center; padding:20px; }
        .box{ width:950px; margin:auto; background:white; padding:30px; border-radius:10px; box-shadow:0px 0px 10px rgba(0,0,0,0.1); }
        .grid-container{ display:grid; grid-template-columns:repeat(3,1fr); gap:12px; }
        .input-group{ text-align:left; }
        label{ font-weight:bold; font-size:14px; }
        input{ width:100%; padding:10px; border-radius:5px; border:1px solid #ccc; box-sizing: border-box; }
        .secao-titulo{ grid-column:span 3; color:#e91e63; font-weight:bold; margin-top:15px; font-size:20px; }
        button{ padding:12px 40px; background:#e91e63; color:white; border:none; margin-top:25px; cursor:pointer; border-radius:5px; font-size:16px; }
        button:hover{ background:#c2185b; }
        #resultado{ margin-top:25px; font-size:1.5em; font-weight:bold; }
        .maligno{ color:red; } .benigno{ color:green; }
    </style>
</head>
<body>
<div class="box">
    <h2>Predição de Câncer de Mama</h2>
    <p>Preencha os 30 parâmetros abaixo</p>
    <div class="grid-container">
        <div class="secao-titulo">Valores Médios</div>
        {% for chave, rotulo in variaveis.items() if 'mean' in chave %}
        <div class="input-group"><label>{{ rotulo }}</label><input id="{{ chave }}" type="number" step="any"></div>
        {% endfor %}
        <div class="secao-titulo">Erro Padrão</div>
        {% for chave, rotulo in variaveis.items() if '_se' in chave %}
        <div class="input-group"><label>{{ rotulo }}</label><input id="{{ chave }}" type="number" step="any"></div>
        {% endfor %}
        <div class="secao-titulo">Piores Valores</div>
        {% for chave, rotulo in variaveis.items() if 'worst' in chave %}
        <div class="input-group"><label>{{ rotulo }}</label><input id="{{ chave }}" type="number" step="any"></div>
        {% endfor %}
    </div>
    <button onclick="prever()">Prever Resultado</button>
    <h3 id="resultado"></h3>
</div>
<script>
const campos = {{ colunas|tojson }};
function prever(){
    let dados = campos.map(id => parseFloat(document.getElementById(id).value));
    if(dados.some(isNaN)){
        document.getElementById("resultado").innerText = "Preencha todos os campos.";
        return;
    }
    fetch('/predict', {
        method:'POST',
        headers:{ 'Content-Type':'application/json' },
        body:JSON.stringify({ dados:dados })
    })
    .then(r=>r.json())
    .then(d=>{
        let div = document.getElementById("resultado");
        if(d.resultado == 1){
            div.innerText = "Resultado: Maligno";
            div.className = "maligno";
        } else {
            div.innerText = "Resultado: Benigno";
            div.className = "benigno";
        }
    })
    .catch(err=>{
        document.getElementById("resultado").innerText = "Erro ao processar previsão.";
    });
}
</script>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML, variaveis=DICIONARIO_VARIAVEIS, colunas=COLUNAS)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        dados_lista = request.json['dados']
        dados_df = pd.DataFrame([dados_lista], columns=COLUNAS)
        
        # Predição direta do KNN
        pred = modelo.predict(dados_df)[0]
        
        return jsonify({'resultado': int(pred)})
    except Exception as e:
        return jsonify({'erro': str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=8080)