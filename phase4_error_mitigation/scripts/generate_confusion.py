#!/usr/bin/env python
"""
Generate confusion matrices for all qubits
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

# Paths
iq_dir = Path.home() / 'Research' / 'Datasets' / 'qsymphony' / 'raw_simulations' / 'readout_data' / 'iq_data'
fig_dir = Path('results/phase4/figures')
fig_dir.mkdir(parents=True, exist_ok=True)

qubits = ['q2', 'q3', 'q4', 'q5', 'q6']
titles = ['RQC Q2', 'RQC Q3', 'RQC Q4', 'RQC Q5', 'RQC Q6']
error_rates = [2.81, 1.58, 1.43, 1.63, 1.04]

print("="*60)
print("Generating Confusion Matrices")
print("="*60)

for qubit, title, error in zip(qubits, titles, error_rates):
    # Load data
    file_path = iq_dir / f"{qubit}_iq_perfect.csv"
    if not file_path.exists():
        print(f"⚠️ File not found: {file_path}")
        continue
    
    df = pd.read_csv(file_path)
    X = df[['I', 'Q']].values
    y = df['state'].values
    
    # Split and scale
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train classifier (use best one for each)
    if qubit in ['q2', 'q5']:  # LDA best for Q2, Q5
        clf = LinearDiscriminantAnalysis()
    else:  # SVM best for others
        clf = SVC(kernel='rbf', gamma='scale')
    
    clf.fit(X_train_scaled, y_train)
    y_pred = clf.predict(X_test_scaled)
    cm = confusion_matrix(y_test, y_pred)
    
    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, cmap='Blues', interpolation='nearest')
    
    # Add labels
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Predicted |0⟩', 'Predicted |1⟩'])
    ax.set_yticklabels(['True |0⟩', 'True |1⟩'])
    
    # Add numbers
    for i in range(2):
        for j in range(2):
            text = ax.text(j, i, f'{cm[i, j]}', ha='center', va='center',
                          color='white' if cm[i, j] > cm.max()/2 else 'black',
                          fontsize=14, fontweight='bold')
    
    ax.set_title(f'{title} Confusion Matrix\nError Rate: {error}%')
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    
    # Save
    plot_path = fig_dir / f'{qubit}_confusion_final.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {plot_path}")
    plt.close()

print("\n✅ All confusion matrices generated!")
