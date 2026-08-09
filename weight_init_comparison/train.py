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

# Hiperparâmetros ideais encontrados via Optuna (hyperparameter_tuning/train.py) -- fixos;
# só a inicialização de pesos varia entre os treinos abaixo
input_dim = X_train.shape[1]
hidden_dims = [96]
norm_type = "layernorm"
dropout_rate = 0.1303068815269234
lr = 0.09440561352765275
epochs = 50

# Comparação de inicialização de pesos: padrão do PyTorch vs Xavier/Glorot vs He/Kaiming.
# O grad_norm de cada época (rastreado em src/engine.fit) serve de diagnóstico: se alguma
# inicialização deixar o gradiente explodindo ou desaparecendo, aparece aqui.
inits = ["default", "xavier", "he"]
histories = {}

for init_name in inits:
    set_seed(42)
    model = BreastCancerMLP(
        input_dim, hidden_dims, nn.ELU, norm_type, dropout_rate, init_method=init_name,
    ).to(device)

    history, best_state = fit(model, train_loader, val_loader, epochs, lr, device)
    model.load_state_dict(best_state)
    torch.save(best_state, CHECKPOINT_DIR / f"weight_init_comparison_{init_name}.pt")

    y_probs, y_true = predict_probs(model, test_loader, device)
    metrics = classification_report_dict(y_true, y_probs)
    histories[init_name] = history

    print(f"\nInit '{init_name}' -- métricas no teste:")
    for k, v in metrics.items():
        if k != "confusion_matrix":
            print(f"   • {k}: {v:.4f}")

# Diagnóstico: perda de treino e norma L2 do gradiente por inicialização
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for init_name, hist in histories.items():
    axes[0].plot(hist["train_loss"], label=init_name)
    axes[1].plot(hist["grad_norm"], label=init_name)

axes[0].set_title("Perda de Treinamento por Inicialização", fontsize=12, fontweight="bold")
axes[0].set_xlabel("Época")
axes[0].set_ylabel("Loss (BCE)")
axes[0].legend()
axes[0].grid(True, linestyle="--", alpha=0.5)

axes[1].set_title("Evolução da Norma L2 do Gradiente por Inicialização", fontsize=12, fontweight="bold")
axes[1].set_xlabel("Época")
axes[1].set_ylabel("Norma L2 (Gradiente)")
axes[1].legend()
axes[1].grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
plt.show()
