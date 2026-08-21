#!/usr/bin/env python
# coding: utf-8

# In[2]:


#to load data
import numpy as np
import pandas as pd

probs = np.load("tppt_probs.npy")
preds = np.load("tppt_preds.npy")
df = pd.read_excel("tppt_input_copy.xlsx")


# In[13]:


import pandas as pd

df = pd.read_excel("TPPT.xlsx")
print(df.columns)
print(df.head())


# In[31]:


# ============================================
# 1. SETUP
# ============================================

import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np

from rdkit import Chem
from rdkit.Chem.rdchem import HybridizationType, BondType

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINEConv, global_mean_pool

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

label_columns = ["Herbicide", "Fungicide", "Insecticide", "Microbicide"]


# In[32]:


# ============================================
# 2. GRAPH FEATURES
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
        int(bt == 1),
        int(bt == 2),
        int(bt == 3),
        int(bt == 12),
        int(bond.IsInRing())
    ]

def smiles_to_graph(smiles):
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

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


# In[33]:


# ============================================
# 3. MODEL
# ============================================

import torch.nn as nn

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


# In[34]:


# ============================================
# 4. LOAD TRAINED MODEL
# ============================================

model = ResidualGINE().to(device)
model.load_state_dict(torch.load("best_weighted_gnn_final.pt", map_location=device))
model.eval()

print("Model loaded")


# In[35]:


# ============================================
# 5. LOAD EXTERNAL DATA
# ============================================

df = pd.read_excel("TPPT.xlsx")

print(df.shape)


# In[36]:


# ============================================
# 6. SMILES → GRAPH
# ============================================

graphs = []
valid_ids = []
valid_smiles = []

for _, row in df.iterrows():
    g = smiles_to_graph(row["SMILES"])
    if g is not None:
        graphs.append(g)
        valid_ids.append(row["COCONUT ID"])
        valid_smiles.append(row["SMILES"])

loader = DataLoader(graphs, batch_size=32, shuffle=False)

print("Valid molecules:", len(graphs))


# In[37]:


# ============================================
# 7. PREDICTION
# ============================================

all_probs = []

with torch.no_grad():
    for batch in loader:
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        probs = torch.sigmoid(out)
        all_probs.append(probs.cpu())

probs = torch.cat(all_probs).numpy()


# In[38]:


# ============================================
# 8. APPLY THRESHOLD
# ============================================

threshold = 0.72
preds = (probs >= threshold).astype(int)


# In[40]:


# ============================================
# 9. SAVE RESULTS
# ============================================

result_df = pd.DataFrame({
    "Compound ID": valid_ids,
    "Smiles": valid_smiles,
    "Herbicide": preds[:,0],
    "Fungicide": preds[:,1],
    "Insecticide": preds[:,2],
    "Microbicide": preds[:,3]
})

result_df.to_excel("TPPT_predictions_2.xlsx", index=False)

print("Saved: TPPT_predictions_2.xlsx")


# In[12]:


# Find BL003 index
idx = result_df[result_df["Compound ID"] == "BL003"].index[0]

print("Probabilities:", probs[idx])


# In[41]:


# ============================================
# ANALYSIS OF TPPT PREDICTIONS
# ============================================

import pandas as pd

# Load your prediction file
df = pd.read_excel("TPPT_predictions_2.xlsx")

# --------------------------------------------
# 1. Count per class
# --------------------------------------------
herb_count = df["Herbicide"].sum()
fung_count = df["Fungicide"].sum()
insect_count = df["Insecticide"].sum()
micro_count = df["Microbicide"].sum()

# --------------------------------------------
# 2. No activity (all zeros)
# --------------------------------------------
no_activity = df[
    (df["Herbicide"] == 0) &
    (df["Fungicide"] == 0) &
    (df["Insecticide"] == 0) &
    (df["Microbicide"] == 0)
].shape[0]

# --------------------------------------------
# 3. Multi-label compounds (optional)
# --------------------------------------------
df["Total_Labels"] = df[["Herbicide","Fungicide","Insecticide","Microbicide"]].sum(axis=1)

multi_label = df[df["Total_Labels"] > 1].shape[0]

# --------------------------------------------
# PRINT RESULTS
# --------------------------------------------
print("Total compounds:", len(df))
print("\n--- Class Counts ---")
print("Herbicide:", int(herb_count))
print("Fungicide:", int(fung_count))
print("Insecticide:", int(insect_count))
print("Microbicide:", int(micro_count))

print("\n--- Other ---")
print("No activity (0 0 0 0):", no_activity)
print("Multi-label compounds:", multi_label)


# In[42]:


# ============================================
# MULTI-LABEL SPLIT ANALYSIS
# ============================================

import pandas as pd

df = pd.read_excel("TPPT_predictions_2.xlsx")

labels = ["Herbicide", "Fungicide", "Insecticide", "Microbicide"]

# --------------------------------------------
# 1. Count combinations
# --------------------------------------------

comb_counts = {}

for _, row in df.iterrows():
    active = tuple([label for label in labels if row[label] == 1])

    if len(active) > 1:  # multi-label only
        comb_counts[active] = comb_counts.get(active, 0) + 1

# --------------------------------------------
# 2. Sort combinations
# --------------------------------------------

sorted_combs = sorted(comb_counts.items(), key=lambda x: x[1], reverse=True)

# --------------------------------------------
# 3. Print results
# --------------------------------------------

print("=== MULTI-LABEL COMBINATIONS ===\n")

for comb, count in sorted_combs:
    print(f"{' + '.join(comb)} : {count}")


# In[50]:


# ============================================
# COMPLETE SAFE SAVE (NO ERRORS)
# ============================================

import pandas as pd
import numpy as np

# 1. Reload original dataset
df = pd.read_excel("TPPT_predictions.xlsx")

# 2. Check lengths
print("Original data:", len(df))
print("Predictions:", len(probs))

# 3. Align data (IMPORTANT)
df = df.iloc[:len(probs)].copy()

# 4. Add predictions
df["Herbicide"] = preds[:,0]
df["Fungicide"] = preds[:,1]
df["Insecticide"] = preds[:,2]
df["Microbicide"] = preds[:,3]

# 5. Add probabilities
df["Prob_Herbicide"] = probs[:,0]
df["Prob_Fungicide"] = probs[:,1]
df["Prob_Insecticide"] = probs[:,2]
df["Prob_Microbicide"] = probs[:,3]

# 6. Save
df.to_excel("TPPT_predictions_with_scores.xlsx", index=False)

print("Saved successfully ✅")


# In[51]:


# ============================================
# SAVE SESSION (IMPORTANT)
# ============================================

import numpy as np
import pandas as pd

# Save predictions
np.save("tppt_probs.npy", probs)
np.save("tppt_preds.npy", preds)

# Save original dataset copy
df.to_excel("tppt_input_copy.xlsx", index=False)


# ============================================
# ACTIVE vs INACTIVE
# ============================================

import numpy as np

total_labels = preds.sum(axis=1)

active = np.sum(total_labels > 0)
inactive = np.sum(total_labels == 0)

plt.figure(figsize=(5,5))
plt.pie([active, inactive],
        labels=["Active", "No Activity"],
        autopct="%1.1f%%")

plt.title("Activity Distribution (TPPT Dataset)")

plt.savefig("TPPT_Activity_Pie.png", dpi=300, bbox_inches='tight')
plt.show()


# In[6]:


