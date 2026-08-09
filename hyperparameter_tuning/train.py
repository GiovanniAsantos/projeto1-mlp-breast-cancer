import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve
import optuna
from src.data import load_breast_cancer_splits
from src.engine import fit, predict_probs
from src.metrics import classification_report_dict
from src.model import BreastCancerMLP
from src.utils import BreastCancerDataset, set_seed

set_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Carregar e dividir dataset (treino 70% / validação 15% / teste 15%, padronizado no treino)
splits = load_breast_cancer_splits(seed=42)
X_train, y_train = splits["X_train"], splits["y_train"]
X_val, y_val = splits["X_val"], splits["y_val"]
X_test, y_test = splits["X_test"], splits["y_test"]
target_names = splits["target_names"]

print(f"📊 Dimensão do dataset: {X_train.shape[0] + X_val.shape[0] + X_test.shape[0], X_train.shape[1]}")
print(f"Classes: {list(target_names)}")
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

train_loader_final = DataLoader(BreastCancerDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
val_loader_final = DataLoader(BreastCancerDataset(X_val, y_val), batch_size=batch_size, shuffle=False)
test_loader_final = DataLoader(BreastCancerDataset(X_test, y_test), batch_size=batch_size, shuffle=False)

# Treinamento Completo
print("🏋️ Treinando modelo definitivo...")
_, best_state = fit(final_model, train_loader_final, val_loader_final, epochs=50, lr=lr, device=device)
final_model.load_state_dict(best_state)
torch.save(best_state, "checkpoints/hyperparameter_tuning_best_model.pt")

# Avaliação no Teste
y_probs, y_test_arr = predict_probs(final_model, test_loader_final, device)
metrics = classification_report_dict(y_test_arr, y_probs)
acc, precision, recall, f1, auc, cm = (
    metrics["accuracy"], metrics["precision"], metrics["recall"],
    metrics["f1"], metrics["roc_auc"], metrics["confusion_matrix"],
)

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
    xticklabels=target_names, yticklabels=target_names,
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