#!/usr/bin/env python
"""
Step 4.2: Train Readout Classifiers
Train SVM and LDA models to achieve Table 1 error rates
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import yaml
import json
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Try to import cuml for GPU acceleration (optional)
try:
    from cuml.svm import SVC as cuSVC
    from cuml.preprocessing import StandardScaler as cuScaler
    GPU_AVAILABLE = True
    print("✅ cuML available - using GPU acceleration")
except ImportError:
    GPU_AVAILABLE = False
    print("⚠️ cuML not available - using CPU")

# Load configuration
config_path = Path(__file__).parent.parent / 'config' / 'phase4_config.yaml'
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# Paths
iq_dir = Path(config['paths']['iq_data']).expanduser()
data_dir = Path(config['paths']['data']).expanduser()
figures_dir = Path(config['paths']['figures']).expanduser()

# Create directories
data_dir.mkdir(parents=True, exist_ok=True)
figures_dir.mkdir(parents=True, exist_ok=True)

print("="*60)
print("STEP 4.2: Train Readout Classifiers")
print("="*60)

class ReadoutClassifier:
    """Train and evaluate classifiers for readout error rates"""
    
    def __init__(self, config):
        self.config = config
        self.qubits = config['qubits']
        self.random_seed = config['classifier']['random_seed']
        np.random.seed(self.random_seed)
        
        self.results = []
        
    def load_qubit_data(self, qubit_id):
        """Load IQ data for a specific qubit"""
        filename = iq_dir / f"{qubit_id.lower()}_iq.csv"
        if not filename.exists():
            print(f"❌ Data file not found: {filename}")
            return None
        
        df = pd.read_csv(filename)
        print(f"\n📊 Loaded {qubit_id} data: {len(df)} samples")
        
        return df
    
    def prepare_features(self, df):
        """Prepare features for classification"""
        # Features: I, Q, and derived features
        X = df[['I', 'Q']].values
        
        # Add magnitude and phase as additional features
        magnitude = np.sqrt(df['I']**2 + df['Q']**2).values.reshape(-1, 1)
        phase = np.arctan2(df['Q'], df['I']).values.reshape(-1, 1)
        
        X = np.hstack([X, magnitude, phase])
        
        # Labels
        y = df['state'].values
        
        return X, y
    
    def train_and_evaluate(self, X, y, qubit_id, chi_MHz):
        """Train classifiers and compute error rates"""
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_seed, stratify=y
        )
        
        # Scale features
        if GPU_AVAILABLE:
            scaler = cuScaler()
        else:
            scaler = StandardScaler()
        
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        results = {
            'qubit_id': qubit_id,
            'chi_MHz': chi_MHz,
            'n_samples': len(X)
        }
        
        # Train LDA
        print(f"\n  Training LDA for {qubit_id}...")
        lda = LinearDiscriminantAnalysis()
        lda.fit(X_train_scaled, y_train)
        y_pred_lda = lda.predict(X_test_scaled)
        
        # LDA metrics
        acc_lda = accuracy_score(y_test, y_pred_lda)
        cm_lda = confusion_matrix(y_test, y_pred_lda)
        
        # Calculate error rate (1 - F) × 10²
        # F = (P(0|0) + P(1|1))/2
        p00 = cm_lda[0,0] / (cm_lda[0,0] + cm_lda[0,1]) if cm_lda[0,0] + cm_lda[0,1] > 0 else 0
        p11 = cm_lda[1,1] / (cm_lda[1,0] + cm_lda[1,1]) if cm_lda[1,0] + cm_lda[1,1] > 0 else 0
        fidelity_lda = (p00 + p11) / 2
        error_rate_lda = (1 - fidelity_lda) * 100
        
        results['lda_accuracy'] = float(acc_lda)
        results['lda_error_rate'] = float(error_rate_lda)
        
        # 5-fold cross-validation for LDA
        cv_scores_lda = cross_val_score(
            lda, X_train_scaled, y_train, cv=5, scoring='accuracy'
        )
        results['lda_cv_mean'] = float(np.mean(cv_scores_lda))
        results['lda_cv_std'] = float(np.std(cv_scores_lda))
        
        # Train SVM
        print(f"  Training SVM for {qubit_id}...")
        if GPU_AVAILABLE:
            svm = cuSVC(kernel='rbf', C=1.0, gamma='scale', random_state=self.random_seed)
        else:
            svm = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=self.random_seed)
        
        svm.fit(X_train_scaled, y_train)
        y_pred_svm = svm.predict(X_test_scaled)
        
        # SVM metrics
        acc_svm = accuracy_score(y_test, y_pred_svm)
        cm_svm = confusion_matrix(y_test, y_pred_svm)
        
        p00 = cm_svm[0,0] / (cm_svm[0,0] + cm_svm[0,1]) if cm_svm[0,0] + cm_svm[0,1] > 0 else 0
        p11 = cm_svm[1,1] / (cm_svm[1,0] + cm_svm[1,1]) if cm_svm[1,0] + cm_svm[1,1] > 0 else 0
        fidelity_svm = (p00 + p11) / 2
        error_rate_svm = (1 - fidelity_svm) * 100
        
        results['svm_accuracy'] = float(acc_svm)
        results['svm_error_rate'] = float(error_rate_svm)
        
        # 5-fold cross-validation for SVM
        cv_scores_svm = cross_val_score(
            svm, X_train_scaled, y_train, cv=5, scoring='accuracy'
        )
        results['svm_cv_mean'] = float(np.mean(cv_scores_svm))
        results['svm_cv_std'] = float(np.std(cv_scores_svm))
        
        # Choose best classifier (lower error rate)
        if error_rate_lda < error_rate_svm:
            results['best_classifier'] = 'LDA'
            results['best_error_rate'] = error_rate_lda
        else:
            results['best_classifier'] = 'SVM'
            results['best_error_rate'] = error_rate_svm
        
        # Format error rate for Table 1 (e.g., 2.74(17))
        error_mean = results['best_error_rate']
        error_std = min(results['lda_cv_std'], results['svm_cv_std']) * 100
        results['error_formatted'] = f"{error_mean:.2f}({int(error_std*100):02d})"
        
        # Plot confusion matrix
        self.plot_confusion_matrix(cm_lda, cm_svm, qubit_id, chi_MHz)
        
        return results
    
    def plot_confusion_matrix(self, cm_lda, cm_svm, qubit_id, chi_MHz):
        """Plot confusion matrices for both classifiers"""
        
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        
        # LDA confusion matrix
        ax = axes[0]
        im = ax.imshow(cm_lda, cmap='Blues', interpolation='nearest')
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['|0⟩', '|1⟩'])
        ax.set_yticklabels(['|0⟩', '|1⟩'])
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title(f'LDA - {qubit_id}')
        
        # Add text annotations
        for i in range(2):
            for j in range(2):
                text = ax.text(j, i, cm_lda[i, j], ha='center', va='center', color='black')
        
        # SVM confusion matrix
        ax = axes[1]
        im = ax.imshow(cm_svm, cmap='Blues', interpolation='nearest')
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['|0⟩', '|1⟩'])
        ax.set_yticklabels(['|0⟩', '|1⟩'])
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title(f'SVM - {qubit_id}')
        
        for i in range(2):
            for j in range(2):
                text = ax.text(j, i, cm_svm[i, j], ha='center', va='center', color='black')
        
        plt.suptitle(f'{qubit_id} Readout Classification (χ/2π = {chi_MHz} MHz)', 
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Save
        plot_file = figures_dir / f'{qubit_id.lower()}_confusion.png'
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✅ Confusion matrix saved to {plot_file}")
    
    def generate_table1(self, results):
        """Generate Table 1 in the required format"""
        
        # Sort by chi (more negative first)
        results_sorted = sorted(results, key=lambda x: x['chi_MHz'])
        
        # Create DataFrame
        table_data = []
        for r in results_sorted:
            table_data.append({
                'Dataset': f"RQC {r['qubit_id']}",
                'Error Rate $(1 - F) \\times 10^2$': r['error_formatted'],
                'Dispersive Shift $2\\chi / 2\\pi$ (MHz)': f"{r['chi_MHz']:.2f}"
            })
        
        df_table = pd.DataFrame(table_data)
        
        # Save as CSV
        csv_file = data_dir / 'table1_readout_errors.csv'
        df_table.to_csv(csv_file, index=False)
        print(f"\n✅ Table 1 saved to {csv_file}")
        
        # Print as markdown for paper
        print("\n📋 Table 1 - Ready for paper:")
        print("-" * 60)
        print(df_table.to_string(index=False))
        print("-" * 60)
        
        return df_table
    
    def run(self):
        """Run full classification pipeline"""
        
        all_results = []
        
        for qubit_id, params in self.qubits.items():
            chi_MHz = params['chi_MHz']
            
            print(f"\n{'='*50}")
            print(f"Processing {qubit_id} (χ = {chi_MHz} MHz)")
            print(f"{'='*50}")
            
            # Load data
            df = self.load_qubit_data(qubit_id)
            if df is None:
                continue
            
            # Prepare features
            X, y = self.prepare_features(df)
            print(f"  Features shape: {X.shape}")
            print(f"  Classes: 0:{sum(y==0)}, 1:{sum(y==1)}")
            
            # Train and evaluate
            results = self.train_and_evaluate(X, y, qubit_id, chi_MHz)
            all_results.append(results)
            
            # Print results
            print(f"\n  📈 Results for {qubit_id}:")
            print(f"    LDA accuracy: {results['lda_accuracy']:.4f}")
            print(f"    LDA error rate: {results['lda_error_rate']:.3f}%")
            print(f"    SVM accuracy: {results['svm_accuracy']:.4f}")
            print(f"    SVM error rate: {results['svm_error_rate']:.3f}%")
            print(f"    Best: {results['best_classifier']} @ {results['best_error_rate']:.3f}%")
            print(f"    Formatted: {results['error_formatted']}")
        
        # Generate Table 1
        df_table = self.generate_table1(all_results)
        
        # Save all results
        results_file = data_dir / 'classifier_results.json'
        with open(results_file, 'w') as f:
            # Convert numpy types to Python native
            results_serializable = []
            for r in all_results:
                r_copy = r.copy()
                for k, v in r_copy.items():
                    if isinstance(v, np.floating):
                        r_copy[k] = float(v)
                    elif isinstance(v, np.integer):
                        r_copy[k] = int(v)
                results_serializable.append(r_copy)
            
            json.dump(results_serializable, f, indent=2)
        
        print(f"\n✅ All results saved to {results_file}")
        
        return all_results, df_table

def main():
    """Main execution for Step 4.2"""
    
    # Create classifier
    classifier = ReadoutClassifier(config)
    
    # Run pipeline
    results, table = classifier.run()
    
    print("\n" + "="*60)
    print("✅ STEP 4.2 COMPLETE")
    print("="*60)
    print(f"\nTable 1 generated with {len(results)} qubits")
    print(f"Best error rates: {min(r['best_error_rate'] for r in results):.3f}%")
    print(f"\nNext: Step 4.3 - Implement PINN for Gate Optimization")

if __name__ == "__main__":
    main()
