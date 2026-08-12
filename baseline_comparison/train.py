import joblib
from pathlib import Path

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression

from src.data import load_breast_cancer_splits
from src.metrics import classification_report_dict
from src.utils import set_seed

ROOT_DIR = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = ROOT_DIR / "checkpoints"

set_seed(42)

# Carregar dataset (mesma divisão 70/15/15 usada em todos os outros experimentos)
splits = load_breast_cancer_splits(seed=42)
X_train, y_train = splits["X_train"], splits["y_train"]
X_test, y_test = splits["X_test"], splits["y_test"]
target_names = splits["target_names"]

print(f"✅ Formato X_train: {X_train.shape} | X_test: {X_test.shape}")

# Baseline trivial (chute pela classe majoritária) -- só pra dar chão ao baseline não trivial
print("\n🎲 Baseline trivial: DummyClassifier (estratégia 'most_frequent')...")
dummy = DummyClassifier(strategy="most_frequent", random_state=42)
dummy.fit(X_train, y_train)
y_probs_dummy = dummy.predict_proba(X_test)[:, 1]
metrics_dummy = classification_report_dict(y_test, y_probs_dummy)
for k, v in metrics_dummy.items():
    if k != "confusion_matrix":
        print(f"   • {k}: {v:.4f}")

# Baseline não trivial: Regressão Logística. É um modelo linear simples, sem representação
# hierárquica de features -- serve de referência pra medir o quanto a não-linearidade do MLP
# realmente agrega nesse problema.
print("\n📐 Baseline não trivial: Regressão Logística (max_iter=1000)...")
baseline = LogisticRegression(max_iter=1000, random_state=42)
baseline.fit(X_train, y_train)

y_probs = baseline.predict_proba(X_test)[:, 1]
metrics = classification_report_dict(y_test, y_probs)

print("\n🎯 Métricas da Regressão Logística no Conjunto de Teste:")
for k, v in metrics.items():
    if k != "confusion_matrix":
        print(f"   • {k}: {v:.4f}")

print("\n🔢 Matriz de Confusão (linhas = real, colunas = previsto):")
print(f"   {target_names}")
print(f"   {metrics['confusion_matrix']}")

CHECKPOINT_DIR.mkdir(exist_ok=True)
joblib.dump(baseline, CHECKPOINT_DIR / "baseline_logistic_regression.joblib")
print(f"\n💾 Baseline salvo em checkpoints/baseline_logistic_regression.joblib")
print("\nℹ️ Use essas métricas como piso de comparação no relatório: o MLP só se justifica")
print("   se superar (ou empatar com vantagem de calibração/robustez) esse baseline linear.")
