# Projeto 1 — Pipeline MLP (Breast Cancer Wisconsin)

Pipeline completo de deep learning ponta a ponta: um MLP construído do zero em
PyTorch (`nn.Module`, sem frameworks de alto nível como PyTorch Lightning),
levado a um estado de treinamento estável e diagnosticado.

## Dataset

Um único dataset é usado no projeto inteiro: **Breast Cancer Wisconsin**
(`sklearn.datasets.load_breast_cancer`, embutido no scikit-learn, sem
download externo). 569 amostras, 30 features numéricas contínuas extraídas de
imagens digitalizadas de núcleos celulares (raio, textura, perímetro, área,
suavidade, compacidade, concavidade, pontos côncavos, simetria e dimensão
fractal — cada uma em três agregações: média, erro padrão e "pior" valor).

Duas variantes do mesmo problema, ambas sobre esse dataset:

- **Classificação** — alvo binário (`0` = maligno, `1` = benigno), usando as
  30 features.
- **Regressão** — alvo contínuo `mean concavity` (severidade média das
  reentrâncias no contorno do núcleo), previsto a partir das outras 27
  features. As outras duas colunas que medem a mesma grandeza em agregações
  diferentes (`concavity error`, `worst concavity`) são removidas da entrada
  para evitar vazamento de alvo (target leakage).

Todo split usa 70% treino / 15% validação / 15% teste (estratificado por
classe na classificação), com `StandardScaler` ajustado **somente** no
treino e aplicado (`transform`) em validação/teste.

## Como rodar

### 1. Ambiente

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac
pip install -r requirements.txt
```

Também roda em Google Colab: suba `notebooks/01_pipeline_mlp.ipynb`, o
Colab já vem com PyTorch/TensorBoard instalados (só rodar a primeira célula
pra conferir a versão).

### 2. Notebook principal (pipeline completo)

```bash
jupyter lab
```

Abra `notebooks/01_pipeline_mlp.ipynb` e rode célula a célula, em ordem —
a Parte 2 (regressão) depende de classes/funções (`TabularMLP`,
`train_stable`, `TabularDataset`) definidas na Parte 1. Esse notebook é a
fonte de verdade do projeto: contém as duas variantes (classificação e
regressão), todas as comparações e o diagnóstico completo do treinamento.

Checkpoints (`.pt`) e logs do TensorBoard são gravados relativos ao diretório
do notebook, em `notebooks/checkpoints/` e `notebooks/runs/`.

### 3. TensorBoard

```bash
tensorboard --logdir notebooks/runs
```

Abre em `http://localhost:6006`. Mostra loss (treino/validação), acurácia,
norma L2 do gradiente e learning rate por época, para os treinos finais
(Adam vs AdamW, classificação e regressão).

### 4. Scripts standalone (`src/` + pastas de experimento)

Além do notebook, existem scripts independentes que reproduzem partes
específicas do pipeline usando os módulos reutilizáveis em `src/`. Rodam a
partir da raiz do projeto:

```bash
python baseline_comparison/train.py
python hyperparameter_tuning/train.py
python activation_function_comparison/train.py
python activation_comparison/train.py
python weight_init_comparison/train.py
```

Cada um salva seu(s) checkpoint(s) em `checkpoints/` na raiz.

## Estrutura do projeto

```
├── notebooks/
│   └── 01_pipeline_mlp.ipynb      # pipeline completo (classificação + regressão)
├── src/                           # módulos reutilizáveis, usados pelos scripts standalone
│   ├── model.py                   # BreastCancerMLP (nn.Module)
│   ├── data.py                    # load_breast_cancer_splits (split 70/15/15 + scaler)
│   ├── engine.py                  # loop de treino, grad_norm, checkpoint, predict_probs
│   ├── metrics.py                 # accuracy/precision/recall/f1/roc_auc/matriz de confusão
│   └── utils.py                   # set_seed, BreastCancerDataset
├── baseline_comparison/train.py   # DummyClassifier + Regressão Logística (baseline não trivial)
├── hyperparameter_tuning/train.py # busca via Optuna (25 trials, TPE + pruning)
├── activation_function_comparison/train.py  # ReLU vs ELU, com hiperparâmetros do Optuna
├── activation_comparison/train.py           # com ativação (ELU) vs sem ativação (Identity)
├── weight_init_comparison/train.py          # default vs Xavier vs He
├── checkpoints/                   # pesos salvos (.pt / .joblib) — gitignored (exceto baseline)
└── requirements.txt
```

Os scripts standalone existem porque cada comparação (ativação, init,
baseline, tuning) foi desenvolvida e testada isoladamente antes de ser
consolidada no notebook principal — o notebook é a versão final e completa;
os scripts servem como registro reproduzível de cada experimento em
separado.

## Arquitetura do modelo

`TabularMLP` / `BreastCancerMLP` — MLP genérico escrito manualmente com
`nn.Module`, sem abstrações de alto nível:

```
Linear(in, h_1) → [BatchNorm1d|LayerNorm]? → Ativação → [Dropout]? →
Linear(h_1, h_2) → ... →
Linear(h_n, output_dim)   # logit (classificação) ou valor contínuo (regressão)
```

- Normalização (`BatchNorm1d`/`LayerNorm`) e `Dropout` são opcionais por
  camada, ligados via parâmetro (`norm_type`, `dropout_rate`).
- Saída final sem ativação: logit cru para `BCEWithLogitsLoss` (mais
  estável numericamente que Sigmoid + BCELoss) na classificação, valor
  contínuo direto para `MSELoss` na regressão.
- Inicialização de pesos configurável: `default` (PyTorch), `xavier_uniform`
  ou `kaiming_normal`/`he`.

**Por que rede rasa (1–3 camadas, 32–128 neurônios):** o dataset tem 30
features e só ~398 amostras de treino — é pequeno e quase linearmente
separável (uma Regressão Logística simples já roda perto de 98% de
acurácia). Uma rede funda (4+ camadas) teria muito mais parâmetros do que
dado para sustentar, com overfitting praticamente garantido e mais risco de
gradiente instável. Por isso a busca de hiperparâmetros (Optuna) varre
`n_layers` entre 1 e 3 e `n_units` entre 32 e 128 — a faixa já nasce
restrita por essa análise, o Optuna só confirma empiricamente o ponto ideal
dentro dela.

## O que o pipeline cobre

- Ambiente (PyTorch + TensorBoard) verificado antes de qualquer treino.
- `DataLoader` + ciclo de validação por época.
- Checkpoint via `state_dict` (melhor modelo por loss de validação) com
  prova de recarregamento e reprodução de predições.
- Diagnóstico de gradiente: norma L2 por época, rastreada e plotada.
- Comparação de inicialização de pesos: default vs Xavier vs He.
- Comparação de função de ativação: ReLU vs ELU (e com/sem ativação).
- Comparação de normalização (BatchNorm1d/LayerNorm) e Dropout, com análise
  do gap treino/validação.
- Busca de hiperparâmetros com Optuna (TPE + pruning, 25 trials).
- Treino final com Adam vs AdamW + `StepLR` scheduling, monitorado no
  TensorBoard.
- Variante de regressão (MSE loss), avaliada com MAE, RMSE e R².
- Baseline não trivial (Regressão Logística) como piso de comparação.
- Métricas finais sempre no conjunto de **teste**, nunca validação.

## Resultados (última execução)

**Classificação** (teste, 86 amostras):

| Modelo | Accuracy | Precision (maligno) | Recall (maligno) | ROC-AUC |
|---|---|---|---|---|
| Baseline trivial (classe majoritária) | 0.6279 | — | — | — |
| Baseline não trivial (Regressão Logística) | 0.9884 | 1.0000 | 0.9688 | 0.9954 |
| MLP final (Optuna + AdamW) | 0.9535 | 0.9375 | 0.9375 | 0.9757 |

O baseline linear supera o MLP tunado neste dataset — evidência de que o
problema é quase linearmente separável (30 features, só 398 amostras de
treino). Análise completa em `RELATORIO.md`.

**Regressão** (teste, 86 amostras, alvo `mean concavity`):

| Métrica | Valor |
|---|---|
| MAE | 0.0142 |
| RMSE | 0.0251 |
| R² | 0.8977 |

Números completos (accuracy, ROC-AUC, MAE/RMSE/R², matriz de confusão,
diagnóstico de gradiente) ficam registrados nos outputs do notebook
executado (`notebooks/01_pipeline_mlp.ipynb`) e no relatório completo
(`RELATORIO.md`).
