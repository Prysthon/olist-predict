# Olist Predict - Machine Learning para Previsão de Atrasos em Entregas

[![PySpark](https://img.shields.io/badge/PySpark-4.0.0-orange.svg)](https://spark.apache.org/docs/latest/api/python/)
[![Python](https://img.shields.io/badge/Python-3.13-blue.svg)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.4.1-green.svg)](https://xgboost.readthedocs.io/)

Projeto de Machine Learning para predição de pedidos com risco de atraso na entrega após a aprovação do pagamento, utilizando dados do e-commerce brasileiro Olist.

## 📊 Sobre o Projeto

Este projeto implementa um pipeline de Machine Learning utilizando PySpark para:

* **Processar** dados de pedidos, itens e clientes do dataset Olist
* **Criar features** temporais e agregadas relevantes para previsão
* **Comparar modelos** utilizando Cross Validation
* **Selecionar o threshold** que maximiza o F1 da classe positiva
* **Selecionar e treinar** somente o melhor modelo
* **Avaliar** o modelo final em um conjunto de teste separado
* **Salvar** o modelo vencedor e seus metadados
* **Executar inferência** utilizando o modelo treinado

### Objetivo

Identificar, logo após a aprovação do pagamento, quais pedidos possuem maior risco de serem entregues depois do prazo prometido.

A previsão pode apoiar decisões operacionais como:

* Priorização de pedidos com maior risco
* Revisão preventiva do prazo de entrega
* Acompanhamento operacional dos pedidos
* Comunicação preventiva com clientes

---

## 🔄 Fluxo de Modelagem

O conjunto de dados é inicialmente dividido em **treino (80%)** e **teste (20%)**.

O conjunto de teste permanece separado durante toda a etapa de seleção do modelo.

```text
Dataset
   │
   ├── Train (80%)
   │      │
   │      └── Cross Validation
   │             │
   │             ├── Gradient Boosting
   │             │
   │             └── XGBoost
   │             │
   │             └── Predições Out-of-Fold
   │                    │
   │                    └── Busca do melhor threshold
   │
   │             └── Seleção pelo F1
   │
   │      └── Treinamento do modelo vencedor
   │          utilizando todo o conjunto de treino
   │
   └── Test (20%)
          │
          └── Avaliação final do modelo vencedor
```

Dessa forma, o conjunto de teste não participa da escolha do modelo nem da seleção do threshold.

---

## 🎯 Resultados

Após a Cross Validation, o **XGBoost** foi selecionado como modelo vencedor.

O modelo foi então treinado novamente utilizando todo o conjunto de treino e avaliado uma única vez no conjunto de teste.

| Modelo         |  F1 Score | Precision |    Recall |   ROC-AUC | Threshold |
| -------------- | --------: | --------: | --------: | --------: | --------: |
| **XGBoost 🏆** | **0.343** | **0.289** | **0.422** | **0.784** |  **0.16** |

### Interpretação dos Resultados

* **F1 de 0.343** representa o equilíbrio entre Precision e Recall para a classe de pedidos atrasados
* **ROC-AUC de 0.784** indica capacidade do modelo de discriminar pedidos com maior e menor risco de atraso
* **Recall de 0.422** significa que o modelo identifica aproximadamente 42% dos pedidos que efetivamente atrasam
* **Precision de 0.289** indica que aproximadamente 29% dos pedidos classificados como risco de atraso efetivamente atrasam
* O **threshold de 0.16** foi selecionado durante a validação, em vez de utilizar o threshold padrão de 0.50

### Dataset

* **Total**: 96.456 registros
* **Treino**: aproximadamente 80%
* **Teste**: aproximadamente 20%
* **Target**: `is_late`
* **Classe positiva**: pedido entregue após o prazo prometido

---

## 🏗️ Arquitetura do Projeto

```text
olist-predict-main/
│
├── data/
│   ├── bronze/                     # Dados brutos (CSV)
│   ├── silver/
│   │   └── olist_features/         # Dataset processado (Parquet)
│   └── gold/
│       └── predictions/            # Predições geradas na inferência
│
├── models/
│   ├── best_model/                 # Modelo vencedor treinado
│   └── metadata.json               # Nome do modelo e threshold
│
├── module_olist/
│   ├── config.py                   # Configurações e paths
│   ├── dataset.py                  # Construção do dataset
│   ├── features.py                 # Engenharia de features
│   ├── main.py                     # Pipeline de treinamento
│   ├── inference.py                # Pipeline de inferência
│   │
│   └── modeling/
│       ├── pipeline.py             # Definição dos pipelines de ML
│       ├── split.py                # Train/test split
│       ├── cross_validation.py     # CV, thresholds e seleção do modelo
│       ├── train.py                # Treinamento do modelo vencedor
│       ├── evaluate.py             # Avaliação final no conjunto de teste
│       └── predict.py              # Geração das previsões
│
├── notebooks/
├── pyproject.toml                  # Dependências do projeto
└── README.md
```

### Responsabilidade dos módulos

| Arquivo               | Responsabilidade                                                                         |
| --------------------- | ---------------------------------------------------------------------------------------- |
| `cross_validation.py` | Comparar modelos, gerar predições Out-of-Fold, testar thresholds e selecionar o vencedor |
| `train.py`            | Treinar somente o modelo vencedor utilizando todo o conjunto de treino                   |
| `evaluate.py`         | Avaliar o modelo final no conjunto de teste                                              |
| `predict.py`          | Aplicar o modelo e o threshold para gerar previsões                                      |
| `main.py`             | Orquestrar o pipeline de treinamento e avaliação                                         |
| `inference.py`        | Carregar o modelo salvo e executar novas previsões                                       |

---

## 🚀 Como Executar

### Pré-requisitos

* **Docker** ou **Colima** para Docker no macOS
* **VS Code** com extensão Dev Containers
* **4 GB+ de RAM** disponível para o container

### Opção 1: Usando Colima (macOS/Linux)

```bash
# Instalar Colima
brew install colima

# Iniciar Colima
colima start --memory 4 --cpu 2 --disk 50

# Verificar status
colima status
```

### Opção 2: Usando Docker Desktop

1. Abra **Docker Desktop**
2. Vá em **Settings → Resources → Memory**
3. Configure pelo menos **4 GB**
4. Clique em **Apply & Restart**

### Configuração do projeto

Clone o repositório:

```bash
git clone <repo-url>
cd olist-predict-main
```

Abra no VS Code:

```bash
code .
```

Reabra utilizando o Dev Container:

1. Pressione `Cmd/Ctrl + Shift + P`
2. Selecione **Dev Containers: Reopen in Container**
3. Aguarde a construção do ambiente

---

## 🧠 Treinamento

Execute:

```bash
uv run python -m module_olist.main
```

O pipeline executará:

```text
Carregamento dos dados
        ↓
Construção do dataset
        ↓
Engenharia de features
        ↓
Train / Test
        ↓
Cross Validation
        ↓
Seleção do modelo + threshold
        ↓
Treinamento do vencedor
        ↓
Salvamento do modelo
        ↓
Avaliação final no Test
```

Exemplo de resultado final:

```text
Modelo: XGBoost
Threshold: 0.16
Precision: 0.289
Recall: 0.422
F1: 0.343
ROC-AUC: 0.784

RESULTADOS FINAIS
XGBoost | F1=0.343 | Precision=0.289 | Recall=0.422 | ROC-AUC=0.784 | Threshold=0.16

Pipeline executado com sucesso.
```

Após o treinamento, são gerados:

```text
models/
├── best_model/
└── metadata.json
```

O `metadata.json` registra automaticamente o modelo vencedor e o threshold selecionado:

```json
{
    "model_name": "XGBoost",
    "threshold": 0.16
}
```

---

## 🔮 Inferência

Depois que o modelo estiver treinado, a inferência pode ser executada separadamente:

```bash
uv run python -m module_olist.inference
```

O pipeline de inferência:

```text
metadata.json
      ↓
carrega modelo vencedor
      ↓
carrega threshold
      ↓
carrega dados
      ↓
predict.py
      ↓
probabilidade de atraso
      ↓
classificação final
      ↓
data/gold/predictions
```

O threshold não é definido manualmente na inferência. Ele é recuperado do `metadata.json` criado durante o treinamento.

Exemplo:

```text
probability_positive | prediction
0.0122               | 0
0.0975               | 0
0.3769               | 1
0.0131               | 0
0.1678               | 1
```

Onde:

* `probability_positive` representa a probabilidade estimada de atraso
* `prediction = 1` representa um pedido classificado como risco de atraso
* `prediction = 0` representa um pedido classificado como sem risco de atraso segundo o threshold selecionado

> Atualmente, o pipeline de inferência utiliza o dataset histórico processado para demonstrar o funcionamento da arquitetura. Em um cenário produtivo, a entrada seria composta por novos pedidos submetidos ao mesmo processo de engenharia de features utilizado durante o treinamento.

---

## 📦 Features Criadas

### Features Temporais

* `purchase_month` - Mês da compra
* `purchase_weekday` - Dia da semana da compra
* `purchase_hour` - Hora da compra
* `promised_days` - Quantidade de dias prometidos para entrega

### Features Agregadas

* `item_count` - Quantidade de itens do pedido
* `seller_count` - Número de vendedores distintos
* `total_price` - Valor total dos itens
* `total_freight` - Valor total do frete

### Features Categóricas

* `customer_state` - Estado do cliente

### Target

* `is_late` - `1` quando o pedido foi entregue após o prazo prometido e `0` caso contrário

---

## 🔧 Tecnologias Utilizadas

### Core

* **Python 3.13** - Linguagem de programação
* **PySpark** - Processamento e Machine Learning
* **XGBoost** - Gradient Boosting para classificação
* **PyArrow** - Integração e processamento de dados

### Ferramentas

* **uv** - Gerenciamento de dependências
* **loguru** - Logging
* **Docker / Colima** - Containerização
* **VS Code Dev Containers** - Ambiente de desenvolvimento reproduzível

---

## 🎓 Ambiente de Desenvolvimento

O projeto utiliza **Dev Containers** para garantir reprodutibilidade do ambiente.

### Configuração

* Java 17 OpenJDK
* Python 3.13
* uv
* PySpark
* XGBoost

### Spark

```python
spark = (
    SparkSession.builder
    .appName("olist-pipeline")
    .master("local[*]")
    .config(
        "spark.driver.memory",
        "2g",
    )
    .config(
        "spark.executor.memory",
        "2g",
    )
    .config(
        "spark.sql.shuffle.partitions",
        "4",
    )
    .getOrCreate()
)
```

---

## 👥 Autores

Projeto desenvolvido como parte da disciplina de Machine Learning.
