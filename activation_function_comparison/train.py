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
print(f"🚀 Usando dispositivo: {device}")

# Carregar dataset (mesma divisão 70/15/15 usada na busca de hiperparâmetros)
splits = load_breast_cancer_splits(seed=42)
X_train, y_train = splits["X_train"], splits["y_train"]
X_val, y_val = splits["X_val"], splits["y_val"]
X_test, y_test = splits["X_test"], splits["y_test"]

print(f"✅ Formato X_train: {X_train.shape} | X_val: {X_val.shape} | X_test: {X_test.shape}")

train_loader = DataLoader(BreastCancerDataset(X_train, y_train), batch_size=64, shuffle=True)
val_loader = DataLoader(BreastCancerDataset(X_val, y_val), batch_size=64, shuffle=False)
test_loader = DataLoader(BreastCancerDataset(X_test, y_test), batch_size=64, shuffle=False)

# Hiperparâmetros ideais encontrados via Optuna (hyperparameter_tuning/train.py) -- fixos;
# só a função de ativação varia entre os treinos abaixo
input_dim = X_train.shape[1]
hidden_dims = [96]
norm_type = "layernorm"
dropout_rate = 0.1303068815269234
lr = 0.09440561352765275
epochs = 50

# Comparação real entre duas funções de ativação: ReLU (padrão amplamente usado, mas
# sujeito a "dying ReLU" -- gradiente zerado para entradas negativas) vs ELU (vencedora
# da busca Optuna, saturação suave no lado negativo). Mede efeito na curva de loss e na
# norma do gradiente, não só o resultado final de acurácia.
activations = {"relu": nn.ReLU, "elu": nn.ELU}
histories = {}

print(f"\n🏋️ Treinando {len(activations)} modelos (ReLU vs ELU), {epochs} épocas cada...")
for act_name, act_cls in activations.items():
    set_seed(42)
    model = BreastCancerMLP(input_dim, hidden_dims, act_cls, norm_type, dropout_rate).to(device)

    history, best_state = fit(model, train_loader, val_loader, epochs, lr, device)
    model.load_state_dict(best_state)
    torch.save(best_state, CHECKPOINT_DIR / f"activation_function_comparison_{act_name}.pt")

    y_probs, y_true = predict_probs(model, test_loader, device)
    metrics = classification_report_dict(y_true, y_probs)
    histories[act_name] = history

    print(f"\n🎯 Ativação '{act_name}' -- métricas no teste:")
    for k, v in metrics.items():
        if k != "confusion_matrix":
            print(f"   • {k}: {v:.4f}")
    print(f"   💾 Checkpoint salvo em checkpoints/activation_function_comparison_{act_name}.pt")

print("\n📈 Plotando diagnóstico: perda de treino/validação e norma L2 do gradiente por ativação...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for act_name, hist in histories.items():
    axes[0].plot(hist["train_loss"], label=f"{act_name} (treino)")
    axes[0].plot(hist["val_loss"], label=f"{act_name} (validação)", linestyle="--")
    axes[1].plot(hist["grad_norm"], label=act_name)

axes[0].set_title("Perda por Função de Ativação (ReLU vs ELU)", fontsize=12, fontweight="bold")
axes[0].set_xlabel("Época")
axes[0].set_ylabel("Loss (BCE)")
axes[0].legend()
axes[0].grid(True, linestyle="--", alpha=0.5)

axes[1].set_title("Evolução da Norma L2 do Gradiente por Ativação", fontsize=12, fontweight="bold")
axes[1].set_xlabel("Época")
axes[1].set_ylabel("Norma L2 (Gradiente)")
axes[1].legend()
axes[1].grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
plt.show()
