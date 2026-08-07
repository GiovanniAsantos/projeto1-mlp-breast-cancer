import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from sklearn.datasets import load_breast_cancer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import optuna

from src.model import BreastCancerMLP
from src.utils import BreastCancerDataset, set_seed

set_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Carregar dataset
data = load_breast_cancer()
X_raw, y_raw = data.data, data.target
print(f"📊 Dimensão do dataset: {X_raw.shape}")
print(f"Classes: {list(data.target_names)}")

# 2. Divisão Estratificada: Treino (70%), Validação (15%), Teste (15%)
X_train_raw, X_temp_raw, y_train, y_temp = train_test_split(
    X_raw, y_raw, test_size=0.30, random_state=42, stratify=y_raw
)
X_val_raw, X_test_raw, y_val, y_test = train_test_split(
    X_temp_raw, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

# 3. Padronização de Atributos (ajuste EXCLUSIVO no conjunto de Treino)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_raw)
X_val = scaler.transform(X_val_raw)
X_test = scaler.transform(X_test_raw)

print(f"✅ Formato X_train: {X_train.shape} | X_val: {X_val.shape} | X_test: {X_test.shape}")

def objective(trial):
    # Tentativa de Hiperparâmetros
    n_layers = trial.suggest_int('n_layers', 1, 3)
    hidden_dims = [
        trial.suggest_int(f'n_units_l{i}', 32, 128, step=32)
        for i in range(n_layers)
    ]

    activation_name = trial.suggest_categorical('activation', ['relu', 'leaky_relu', 'elu'])
    act_map = {'relu': nn.ReLU, 'leaky_relu': nn.LeakyReLU, 'elu': nn.ELU}
    act_cls = act_map[activation_name]

    norm_type = trial.suggest_categorical('norm_type', [None, 'batchnorm', 'layernorm'])
    dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.3)
    lr = trial.suggest_float('lr', 1e-4, 1e-1, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])

    # DataLoaders com o batch_size sugerido
    train_loader = DataLoader(BreastCancerDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(BreastCancerDataset(X_val, y_val), batch_size=batch_size, shuffle=False)

    # Instanciando Modelo, Função de Perda e Otimizador
    model = BreastCancerMLP(X_train.shape[1], hidden_dims, act_cls, norm_type, dropout_rate).to(device)
   
    # Utilizando BCEWithLogitsLoss pois como meu alvo é binário e não contínuo, utiliza-se essa para melhor classificação.
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float('inf')

    # 4. Loop de Treinamento e Pruning
    for epoch in range(35):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            out = model(X_batch)
            loss = criterion(out, y_batch)
            loss.backward()
            optimizer.step()

        # Validação
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                out = model(X_batch)
                val_loss += criterion(out, y_batch).item() * len(y_batch)
        val_loss /= len(val_loader.dataset)

        if val_loss < best_val_loss:
            best_val_loss = val_loss

        # Reportar e checar necessidade de Pruning
        trial.report(val_loss, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return best_val_loss


optuna.logging.set_verbosity(optuna.logging.WARNING)

set_seed(42)
study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))

print("⏳ Iniciando a busca de hiperparâmetros com Optuna (25 trials)...")
study.optimize(objective, n_trials=25)

print("\n🎉 Otimização concluída!")
print(f"🏆 Melhor Perda de Validação (BCE): {study.best_value:.4f}")
print("📌 Melhores Hiperparâmetros Encontrados:")
for k, v in study.best_params.items():
    print(f"   • {k}: {v}")
    
    # Plot 1: Histórico de Otimização
df_trials = study.trials_dataframe()

plt.figure(figsize=(10, 4))
plt.plot(df_trials['number'], df_trials['value'], marker='o', color='#1BB5D8', linewidth=2)
plt.axhline(study.best_value, color='#FF7043', linestyle='--', label=f"Melhor Valor: {study.best_value:.4f}")
plt.title("Histórico de Progresso da Otimização (Optuna TPE)", fontsize=12, fontweight='bold')
plt.xlabel("Número do Trial")
plt.ylabel("Perda de Validação (BCE)")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# Plot 2: Importância dos Hiperparâmetros
importances = optuna.importance.get_param_importances(study)
param_names = list(importances.keys())
importance_values = list(importances.values())

plt.figure(figsize=(8, 4))
sns.barplot(x=importance_values, y=param_names, palette='Blues_r')
plt.title("Importância Relativa dos Hiperparâmetros", fontsize=12, fontweight='bold')
plt.xlabel("Grau de Importância (Optuna)")
plt.tight_layout()
plt.show()

best_p = study.best_params

n_layers = best_p['n_layers']
hidden_dims = [best_p[f'n_units_l{i}'] for i in range(n_layers)]
act_cls = {'relu': nn.ReLU, 'leaky_relu': nn.LeakyReLU, 'elu': nn.ELU}[best_p['activation']]
norm_type = best_p['norm_type']
dropout_rate = best_p['dropout_rate']
lr = best_p['lr']
batch_size = best_p['batch_size']

set_seed(42)
final_model = BreastCancerMLP(X_train.shape[1], hidden_dims, act_cls, norm_type, dropout_rate).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(final_model.parameters(), lr=lr)

train_loader_final = DataLoader(BreastCancerDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
test_loader_final = DataLoader(BreastCancerDataset(X_test, y_test), batch_size=batch_size, shuffle=False)

# Treinamento Completo
print("🏋️ Treinando modelo definitivo...")
for epoch in range(1, 51):
    final_model.train()
    for X_batch, y_batch in train_loader_final:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        out = final_model(X_batch)
        loss = criterion(out, y_batch)
        loss.backward()
        optimizer.step()

# Avaliação no Teste
final_model.eval()
y_probs = []
with torch.no_grad():
    for X_batch, _ in test_loader_final:
        X_batch = X_batch.to(device)
        logits = final_model(X_batch).cpu().numpy()
        y_probs.extend((1 / (1 + np.exp(-logits))).flatten())

# No exemplo da aula (Insurance, regressão) a saida do modelo ja era a previsao final,
# usada direto. Aqui a saida e logit de classificacao (BCEWithLogitsLoss nao aplica
# sigmoid internamente, por estabilidade numerica) -- por isso e preciso aplicar sigmoid
# manualmente pra virar probabilidade, e so entao arredondar num limiar (0.5) pra
# decidir a classe (0=maligno, 1=benigno). Passo que nao existe em regressao.
y_probs = np.array(y_probs)
y_pred = (y_probs >= 0.5).astype(int)

# Cálculo de Métricas Finais de Classificação
# Regressao tem 1 eixo de erro (distancia numerica) -- MAE/RMSE/R2 sao 3 jeitos de
# resumir esse mesmo eixo. Classificacao binaria tem 2 eixos de erro (falso positivo x
# falso negativo), que nenhuma metrica unica captura -- por isso mais metricas aqui,
# cada uma respondendo uma pergunta diferente sobre o mesmo resultado.
# load_breast_cancer usa label 0 = maligno, 1 = benigno. precision/recall/f1_score do
# sklearn assumem pos_label=1 por padrao -- sem isso explicito, mediriam a classe
# benigna, nao a maligna (a que clinicamente importa: falso negativo = cancer nao
# detectado). Por isso pos_label=0 abaixo.
acc = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, pos_label=0)
recall = recall_score(y_test, y_pred, pos_label=0)
f1 = f1_score(y_test, y_pred, pos_label=0)
auc = roc_auc_score(y_test, y_probs)
cm = confusion_matrix(y_test, y_pred)

print("\n🎯 Métricas Finais no Conjunto de Teste:")
print(f"   • Acurácia: {acc:.4f}")
print(f"   • Precision: {precision:.4f}")
print(f"   • Recall (sensibilidade p/ classe maligna): {recall:.4f}")
print(f"   • F1-score: {f1:.4f}")
print(f"   • ROC-AUC: {auc:.4f}")

# Gráfico de Matriz de Confusão e Curva ROC
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Matriz de Confusão
sns.heatmap(
    cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
    xticklabels=data.target_names, yticklabels=data.target_names,
)
axes[0].set_title("Matriz de Confusão", fontsize=12, fontweight='bold')
axes[0].set_xlabel("Previsto")
axes[0].set_ylabel("Real")

# Curva ROC
fpr, tpr, _ = roc_curve(y_test, y_probs)
axes[1].plot(fpr, tpr, color='#0A345D', label=f"ROC-AUC = {auc:.4f}")
axes[1].plot([0, 1], [0, 1], 'r--', label='Aleatório')
axes[1].set_title("Curva ROC", fontsize=12, fontweight='bold')
axes[1].set_xlabel("Taxa de Falso Positivo")
axes[1].set_ylabel("Taxa de Verdadeiro Positivo")
axes[1].legend()
axes[1].grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()