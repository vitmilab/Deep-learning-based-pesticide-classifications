#!/usr/bin/env python
# coding: utf-8

# In[1]:


# ============================================
# 0. REPRODUCIBILITY (SEED)
# ============================================

import torch
import numpy as np
import random

seed = 42

torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

print("Seed set to:", seed)


# In[2]:


# ============================================
# 1. SETUP & REPRODUCIBILITY
# ============================================

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import pandas as pd
import random

from rdkit import Chem
from rdkit.Chem.rdchem import HybridizationType, BondType

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINEConv, global_mean_pool

from sklearn.metrics import f1_score

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Seed
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

print("Seed set to:", seed)


# In[3]:


# ============================================
# 2. LOAD DATA
# ============================================

train_df = pd.read_csv("train_augmented_balanced_clean.csv")
val_df = pd.read_csv("val.csv")
test_df = pd.read_csv("test.csv")

label_columns = ["Herbicide", "Fungicide", "Insecticide", "Microbicide"]

print(train_df.shape, val_df.shape, test_df.shape)


# In[4]:


# ============================================
# 3. GRAPH FEATURES
# ============================================

def atom_features(atom):
    hybridization = atom.GetHybridization()
    return [
        atom.GetAtomicNum(),
        atom.GetDegree(),
        atom.GetFormalCharge(),
        int(atom.GetIsAromatic()),
        atom.GetTotalValence(),
        atom.GetTotalNumHs(),
        int(atom.IsInRing()),
        int(hybridization == HybridizationType.SP),
        int(hybridization == HybridizationType.SP2),
        int(hybridization == HybridizationType.SP3)
    ]

def bond_features(bond):
    bt = bond.GetBondType()
    return [
        int(bt == BondType.SINGLE),
        int(bt == BondType.DOUBLE),
        int(bt == BondType.TRIPLE),
        int(bt == BondType.AROMATIC),
        int(bond.IsInRing())
    ]

def smiles_to_graph(smiles, label):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    x = torch.tensor([atom_features(a) for a in mol.GetAtoms()], dtype=torch.float)

    edge_index, edge_attr = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bf = bond_features(bond)

        edge_index += [[i, j], [j, i]]
        edge_attr += [bf, bf]

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    y = torch.tensor(label, dtype=torch.float).view(1, -1)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)


# In[5]:


# ============================================
# 4. BUILD GRAPH DATASETS
# ============================================

from tqdm import tqdm

def build_graph_dataset(df):
    graphs = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        g = smiles_to_graph(row["Smile"], row[label_columns].values.tolist())
        if g:
            graphs.append(g)
    return graphs

train_graphs = build_graph_dataset(train_df)
val_graphs = build_graph_dataset(val_df)
test_graphs = build_graph_dataset(test_df)


# In[6]:


# ============================================
# 5. DATALOADERS
# ============================================

generator = torch.Generator().manual_seed(seed)

train_loader = DataLoader(train_graphs, batch_size=32, shuffle=True, generator=generator)
val_loader = DataLoader(val_graphs, batch_size=32, shuffle=False)
test_loader = DataLoader(test_graphs, batch_size=32, shuffle=False)


# In[7]:


# ============================================
# 6. MODEL (Residual GINE)
# ============================================

class ResidualGINE(nn.Module):

    def __init__(self):
        super().__init__()
        hidden = 256

        self.lin_in = nn.Linear(10, hidden)

        def block():
            return nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden)
            )

        self.conv1 = GINEConv(block(), edge_dim=5)
        self.conv2 = GINEConv(block(), edge_dim=5)
        self.conv3 = GINEConv(block(), edge_dim=5)

        self.bn1 = nn.BatchNorm1d(hidden)
        self.bn2 = nn.BatchNorm1d(hidden)
        self.bn3 = nn.BatchNorm1d(hidden)

        self.dropout = nn.Dropout(0.3)
        self.lin_out = nn.Linear(hidden, 4)

    def forward(self, x, edge_index, edge_attr, batch):
        x = self.lin_in(x)

        h = self.conv1(x, edge_index, edge_attr)
        x = F.relu(self.bn1(h) + x)

        h = self.conv2(x, edge_index, edge_attr)
        x = F.relu(self.bn2(h) + x)

        h = self.conv3(x, edge_index, edge_attr)
        x = F.relu(self.bn3(h) + x)

        x = global_mean_pool(x, batch)
        x = self.dropout(x)

        return self.lin_out(x)


# In[8]:


# ============================================
# 7. LOSS + OPTIMIZER
# ============================================

labels = train_df[label_columns].values
pos_weight = torch.tensor((len(labels) - labels.sum(0)) / labels.sum(0), dtype=torch.float)

model = ResidualGINE().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))


# In[9]:


# ============================================
# 8. OUTPUT COLLECTION
# ============================================

def collect_outputs(loader):
    model.eval()
    probs, labels = [], []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            probs.append(torch.sigmoid(out).cpu())
            labels.append(batch.y.cpu())

    return torch.cat(probs).numpy(), torch.cat(labels).numpy()


# In[10]:


# ============================================
# 9. TRAINING
# ============================================

best_f1 = 0

for epoch in range(1, 101):

    model.train()
    total_loss = 0

    for batch in train_loader:
        batch = batch.to(device)

        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

        loss = criterion(out, batch.y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    val_probs, val_labels = collect_outputs(val_loader)
    preds = (val_probs > 0.5).astype(int)

    val_f1 = f1_score(val_labels, preds, average="macro")

    print(f"Epoch {epoch} | Loss: {total_loss:.4f} | Val F1: {val_f1:.4f}")

    if val_f1 > best_f1:
        best_f1 = val_f1
        torch.save(model.state_dict(), "best_weighted_gnn_final.pt")

print("Best Val F1:", best_f1)


# #Training is over

# In[11]:


# ============================================
# 10. LOAD BEST MODEL (AFTER TRAINING)
# ============================================

model = ResidualGINE().to(device)
model.load_state_dict(torch.load("best_weighted_gnn_final.pt", map_location=device))
model.eval()

print("Best model loaded for evaluation.")


# In[12]:


# ============================================
# 11. GENERATE PREDICTIONS
# ============================================

val_probs, val_labels = collect_outputs(val_loader)
test_probs, test_labels = collect_outputs(test_loader)

print("Test probs shape:", test_probs.shape)


# In[15]:


# ============================================
# RE-OPTIMIZE GLOBAL THRESHOLD (VALIDATION)
# ============================================

from sklearn.metrics import accuracy_score
import numpy as np

best_acc = 0
best_thresh = 0

for t in np.arange(0.1, 0.9, 0.01):
    y_pred_temp = (val_probs >= t).astype(int)
    acc = accuracy_score(val_labels, y_pred_temp)

    if acc > best_acc:
        best_acc = acc
        best_thresh = t

print("Best Threshold:", best_thresh)
print("Best Validation Exact:", best_acc)


# In[17]:


# ============================================
# 12. APPLY FINAL THRESHOLD
# ============================================

from sklearn.metrics import accuracy_score

final_threshold = 0.72 #final_threshold = best_thresh

y_pred = (test_probs >= final_threshold).astype(int)

exact_acc = accuracy_score(test_labels, y_pred)

print("Exact Match Accuracy:", exact_acc)


# In[16]:


# Apply optimal threshold on test
y_pred = (test_probs >= best_thresh).astype(int)

from sklearn.metrics import accuracy_score
print("New Test Exact:", accuracy_score(test_labels, y_pred))


# In[19]:


# ============================================
# 13. FINAL METRICS
# ============================================

from sklearn.metrics import (
    classification_report,
    f1_score,
    hamming_loss,
    accuracy_score,
    roc_auc_score,
    average_precision_score
)
import numpy as np

# Apply FINAL threshold
final_threshold = 0.72 
y_pred = (test_probs >= final_threshold).astype(int)

print("========== FINAL TEST METRICS ==========\n")

# -------------------------------
# Classification report
# -------------------------------
print(classification_report(
    test_labels,
    y_pred,
    target_names=label_columns,
    zero_division=0
))

# -------------------------------
# F1 scores
# -------------------------------
micro_f1 = f1_score(test_labels, y_pred, average="micro")
macro_f1 = f1_score(test_labels, y_pred, average="macro")
weighted_f1 = f1_score(test_labels, y_pred, average="weighted")

# -------------------------------
# Exact match & Hamming
# -------------------------------
exact_match = accuracy_score(test_labels, y_pred)
hamming_acc = 1 - hamming_loss(test_labels, y_pred)

# -------------------------------
# ROC & PR (use probabilities)
# -------------------------------
macro_roc = roc_auc_score(test_labels, test_probs, average="macro")
macro_pr = average_precision_score(test_labels, test_probs, average="macro")

# -------------------------------
# Print everything
# -------------------------------
print("Micro F1:", round(micro_f1, 4))
print("Macro F1:", round(macro_f1, 4))
print("Weighted F1:", round(weighted_f1, 4))
print("Exact Match:", round(exact_match, 4))
print("Hamming Accuracy:", round(hamming_acc, 4))

print("\n========== ROC & PR ==========\n")
print("Macro ROC-AUC:", round(macro_roc, 4))
print("Macro PR-AUC:", round(macro_pr, 4))


# In[21]:


# ============================================
# FINAL ROC CURVE (UPDATED MODEL)
# ============================================

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

y_true = test_labels
y_score = test_probs

n_classes = y_true.shape[1]

fpr, tpr, roc_auc = {}, {}, {}

# Per-class ROC
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Micro-average
fpr["micro"], tpr["micro"], _ = roc_curve(y_true.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Plot
plt.figure(figsize=(8, 6))

colors = ['blue', 'green', 'red', 'purple']

for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], lw=2,
             label=f'{label_columns[i]} (AUC = {roc_auc[i]:.2f})')

plt.plot(fpr["micro"], tpr["micro"], 'k--', lw=2,
         label=f'Micro-average (AUC = {roc_auc["micro"]:.2f})')

plt.plot([0, 1], [0, 1], 'gray', linestyle='--')

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (Final GNN Model)")
plt.legend(loc="lower right")
plt.grid(alpha=0.3)

plt.savefig("ROC_Curve_Github.png", dpi=300, bbox_inches='tight')
plt.show()


# In[22]:


# ============================================
# FINAL PR CURVE
# ============================================

from sklearn.metrics import precision_recall_curve, average_precision_score

precision, recall, pr_auc = {}, {}, {}

for i in range(n_classes):
    precision[i], recall[i], _ = precision_recall_curve(y_true[:, i], y_score[:, i])
    pr_auc[i] = average_precision_score(y_true[:, i], y_score[:, i])

precision["micro"], recall["micro"], _ = precision_recall_curve(
    y_true.ravel(), y_score.ravel()
)
pr_auc["micro"] = average_precision_score(y_true, y_score, average="micro")

plt.figure(figsize=(8, 6))

for i, color in zip(range(n_classes), colors):
    plt.plot(recall[i], precision[i], lw=2,
             label=f'{label_columns[i]} (AP = {pr_auc[i]:.2f})')

plt.plot(recall["micro"], precision["micro"], 'k--', lw=2,
         label=f'Micro-average (AP = {pr_auc["micro"]:.2f})')

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve (Final GNN Model)")
plt.legend(loc="lower left")
plt.grid(alpha=0.3)

plt.savefig("PR_Curve_Github.png", dpi=300, bbox_inches='tight')
plt.show()


# In[23]:


# ============================================
# PROBABILITY DISTRIBUTION
# ============================================

import seaborn as sns

plt.figure(figsize=(10, 6))

for i, color in zip(range(n_classes), colors):
    sns.kdeplot(test_probs[:, i], label=label_columns[i], linewidth=2)

plt.title("Prediction Probability Distribution (Per Class)")
plt.xlabel("Predicted Probability")
plt.ylabel("Density")
plt.legend()
plt.grid(alpha=0.3)

plt.savefig("Probability_Distribution_Github.png", dpi=300, bbox_inches='tight')
plt.show()


# In[20]:


# ============================================
# SAVE FINAL OUTPUTS (GITHUB VERSION)
# ============================================

import numpy as np
import json
import pandas as pd
import torch

# -------------------------------
# 1. Save model
# -------------------------------
torch.save(model.state_dict(), "best_weighted_gnn_github.pt")
print("Saved: best_weighted_gnn_github.pt")

# -------------------------------
# 2. Save threshold
# -------------------------------
threshold_dict = {
    "global_threshold": 0.72
}

with open("final_threshold_github.json", "w") as f:
    json.dump(threshold_dict, f, indent=4)

print("Saved: final_threshold_github.json")

# -------------------------------
# 3. Save predictions
# -------------------------------
np.save("test_probs_github.npy", test_probs)
np.save("test_preds_github.npy", y_pred)
np.save("test_labels_github.npy", test_labels)

print("Saved: predictions (github version)")

# -------------------------------
# 4. Save metrics
# -------------------------------
metrics = {
    "Micro_F1": float(micro_f1),
    "Macro_F1": float(macro_f1),
    "Weighted_F1": float(weighted_f1),
    "Exact_Match": float(exact_match),
    "Hamming_Accuracy": float(hamming_acc),
    "Macro_ROC_AUC": float(roc_auc_score(test_labels, test_probs, average="macro")),
    "Macro_PR_AUC": float(average_precision_score(test_labels, test_probs, average="macro"))
}

with open("final_metrics_github.json", "w") as f:
    json.dump(metrics, f, indent=4)

print("Saved: final_metrics_github.json")

# -------------------------------
# 5. Save classification report
# -------------------------------
report_df = pd.DataFrame(classification_report(
    test_labels,
    y_pred,
    target_names=label_columns,
    output_dict=True
)).transpose()

report_df.to_csv("classification_report_github.csv")

print("Saved: classification_report_github.csv")








