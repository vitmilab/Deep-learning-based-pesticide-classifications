# Deep-learning-based-pesticide-classification

This repository contains the computational workflow for deep-learning-based multi-label classification of pesticide compounds across four classes: Herbicide, Fungicide, Insecticide, and Microbicide.

The workflow consists of LSTM-based SMILES data augmentation, followed by GINE-based graph neural network (GNN) classification, and external dataset validation.
## Repository Files

**1. LSTM_Github.ipynb**

This notebook performs class-specific SMILES generation using LSTM models for the four pesticide classes.

The workflow includes:
- Loading the original training dataset
- SMILES preprocessing and tokenization
- LSTM model construction
- SMILES generation
- Validity evaluation using RDKit
- Uniqueness evaluation
- Novelty evaluation against the training data
- Selection of required synthetic compounds
- Construction of the balanced augmented training dataset

The notebook begins with the original master training dataset (masterlist as mentioned in the code).

**2. GNN_Github.py**

This script performs multi-label pesticide classification using a GINE-based Graph Neural Network.
The four prediction classes are:
1. Herbicide
2. Fungicide
3. Insecticide
4. Microbicide

**3. External_dataset_GNN.py**

This script performs external validation of the trained GNN model.
It reads the independent TPPT dataset, applies the trained model, and generates class predictions and prediction scores.

## Additional Files

The larger datasets and prediction files are provided separately as additional files.

_**File	Description**_

**Additionalfile_1.csv** - Original master training dataset used as the input for the LSTM workflow

**Additionalfile_2.csv** - Preprocessed dataset used for LSTM augmentation

**Additionalfile_3.xlsx** - TPPT external dataset used as input for external validation

**Additionalfile_4.xlsx** - Prediction output generated for the TPPT external dataset

## File correspondence in the code

1. For the LSTM workflow:
df = pd.read_csv("masterlist.csv")
corresponds to Additionalfile_1.csv.

2. For external validation:
df = pd.read_excel("TPPT.xlsx")
corresponds to Additionalfile_3.xlsx.

## Execution Order
For reproducing the complete workflow, follow this order:

### Step 1 — LSTM-based data augmentation
Run:
LSTM_Github.ipynb

Use Additionalfile_1.csv as the original training dataset which is mentioned as masterlist in the code. After the preprocessing step, produces Additional file 2.

### Step 2 — GINE-GNN classification
Run:
GNN_Github.py

### Step 3 — External validation
Run:
External_dataset_GNN.py

Use Additionalfile_3.xlsx as the external dataset input.
The script generates predictions and prediction scores for the external compounds. The corresponding prediction output is provided as Additionalfile_4.xlsx.

### Software Requirements
The workflow requires Python and the following major packages:
Python
PyTorch
RDKit
Pandas
NumPy
Scikit-learn
Matplotlib

## Workflow

Additional File 1
Original masterlist.csv

        │
        
        ▼
        
LSTM_Github.ipynb

        │
        
        ├── Preprocessing - Additional File 2
        
        ├── Train / validation / test preparation
        
        │
        
        ▼
        
LSTM-based augmentation

        │
        
        ▼
        
Balanced augmented training dataset

        │
        
        ▼
        
GNN_Github.py

        │
        
        |
        
        ├── GINE-GNN training
        
        └── Model evaluation
        
        │
        
        ▼
        
External_dataset_GNN.py

        ▲
        
        │
        
Additional File 3 (TPPT.xlsx input)

        │
        
        ▼
        
Additional File 4 (TPPT predictions)



