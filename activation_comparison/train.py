from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data import load_breast_cancer_splits
from src.engine import fit, predict_probs
from src.metrics import classification_report_dict
from src.model import BreastCancerMLP
from src.utils import BreastCancerDataset, set_seed

ROOT_DIR = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = ROOT_DIR / "checkpoints"

set_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Carregar dataset (mesma divisão 70/15/15 usada na busca de hiperparâmetros)
splits = load_breast_cancer_splits(seed=42)
X_train, y_train = splits["X_train"], splits["y_train"]
X_val, y_val = splits["X_val"], splits["y_val"]
X_test, y_test = splits["X_test"], splits["y_test"]

train_loader = DataLoader(BreastCancerDataset(X_train, y_train), batch_size=64, shuffle=True)
val_loader = DataLoader(BreastCancerDataset(X_val, y_val), batch_size=64, shuffle=False)
test_loader = DataLoader(BreastCancerDataset(X_test, y_test), batch_size=64, shuffle=False)

# Hiperparâmetros ideais encontrados via Optuna (hyperparameter_tuning/train.py)
input_dim = X_train.shape[1]
hidden_dims = [96]
norm_type = "layernorm"
dropout_rate = 0.1303068815269234
lr = 0.09440561352765275
epochs = 50

# Instanciar modelos com e sem função de ativação (ELU foi a vencedora na busca do Optuna)
set_seed(42)
model_com_ativacao = BreastCancerMLP(input_dim, hidden_dims, nn.ELU, norm_type, dropout_rate).to(device)
set_seed(42)
model_sem_ativacao = BreastCancerMLP(input_dim, hidden_dims, nn.Identity, norm_type, dropout_rate).to(device)

print("Estrutura do Modelo MLP Breast Cancer (com ativação):")
print(model_com_ativacao)

# Treinamento (forward -> loss -> backward -> step por época, com validação e
# checkpoint automático do melhor state_dict) -- mesma rotina usada na busca de hiperparâmetros
history_com, best_state_com = fit(model_com_ativacao, train_loader, val_loader, epochs, lr, device)
history_sem, best_state_sem = fit(model_sem_ativacao, train_loader, val_loader, epochs, lr, device)

model_com_ativacao.load_state_dict(best_state_com)
model_sem_ativacao.load_state_dict(best_state_sem)

torch.save(best_state_com, CHECKPOINT_DIR / "activation_comparison_com_ativacao.pt")
torch.save(best_state_sem, CHECKPOINT_DIR / "activation_comparison_sem_ativacao.pt")

# Avaliação no conjunto de teste
y_probs_com, y_true = predict_probs(model_com_ativacao, test_loader, device)
y_probs_sem, _ = predict_probs(model_sem_ativacao, test_loader, device)

metrics_com = classification_report_dict(y_true, y_probs_com)
metrics_sem = classification_report_dict(y_true, y_probs_sem)

print("\nCom ativação (ELU) -- métricas no teste:")
for k, v in metrics_com.items():
    if k != "confusion_matrix":
        print(f"   • {k}: {v:.4f}")

print("\nSem ativação (Identity) -- métricas no teste:")
for k, v in metrics_sem.items():
    if k != "confusion_matrix":
        print(f"   • {k}: {v:.4f}")

# Prova de que o checkpoint salvo funciona: recarrega do disco e reavalia
modelo_recarregado = BreastCancerMLP(input_dim, hidden_dims, nn.ELU, norm_type, dropout_rate).to(device)
modelo_recarregado.load_state_dict(torch.load(CHECKPOINT_DIR / "activation_comparison_com_ativacao.pt"))
y_probs_reload, y_true_reload = predict_probs(modelo_recarregado, test_loader, device)
metrics_reload = classification_report_dict(y_true_reload, y_probs_reload)
print(f"\nModelo recarregado do checkpoint -- acurácia: {metrics_reload['accuracy']:.4f} "
      f"(deve bater com 'Com ativação' acima)")

# Curvas de loss: com vs sem função de ativação
plt.figure(figsize=(9, 4))
plt.plot(history_com["train_loss"], label="Com ativação (treino)", color="#1BB5D8", linewidth=2)
plt.plot(history_com["val_loss"], label="Com ativação (validação)", color="#1BB5D8", linestyle="--", linewidth=2)
plt.plot(history_sem["train_loss"], label="Sem ativação (treino)", color="#FF7043", linewidth=2)
plt.plot(history_sem["val_loss"], label="Sem ativação (validação)", color="#FF7043", linestyle="--", linewidth=2)
plt.title("Efeito da Função de Ativação no Treinamento (Breast Cancer Dataset)")
plt.xlabel("Épocas")
plt.ylabel("Loss (BCE)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
