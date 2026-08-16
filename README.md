# olist-predict
Repositório da aula Machine Learning na predição de quais pedidos possuem maior risco de serem entregues depois do prazo prometido, após pagamento

## Ambiente de desenvolvimento

Este projeto inclui um Dev Container em `.devcontainer/` para abrir o repositório no VS Code sem instalar Python, Java ou PySpark diretamente na máquina.

Para usar:

1. Instale Docker e a extensão **Dev Containers** no VS Code.
2. Abra o repositório no VS Code.
3. Use o comando **Dev Containers: Reopen in Container**.

Na primeira criação, o container instala Java 17, `uv` e sincroniza as dependências do projeto com:

```bash
uv sync --dev
```

O PySpark fica configurado para uso local, ideal para estudo e experimentação:

```python
from pyspark.sql import SparkSession

spark = (
    SparkSession.builder
    .appName("olist-predict")
    .master("local[*]")
    .getOrCreate()
)
```
