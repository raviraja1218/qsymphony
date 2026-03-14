Q-SYMPHONY

Official research implementation of Q-SYMPHONY, an autonomous AI framework for entanglement stabilization and circuit optimization in hybrid superconducting–mechanical quantum systems.

This repository contains the full codebase used in the research project:

Autonomous Entanglement Stabilization and Circuit Optimization in Hybrid Superconducting–Mechanical Quantum Systems

Overview

Scaling superconducting quantum processors toward fault-tolerant quantum computing is limited by:

hardware complexity

decoherence and environmental noise

large quantum error-correction overhead

Q-SYMPHONY addresses these challenges through an integrated AI-driven digital twin framework combining:

Symplectic Graph Neural Networks (SympGNN) for hardware topology discovery

Recurrent Reinforcement Learning (PPO + LSTM) for entanglement stabilization

Physics-Informed Neural Networks (PINNs) for noise-aware circuit optimization

The framework models hybrid piezo-optomechanical quantum systems where superconducting transmon qubits are coupled to acoustic resonators.

Key Results
Metric	Result
Steady-state entanglement	Eₙ = 0.506 ± 0.027
Training seeds	3 (1000, 1001, 1002)
Circuit depth reduction	43.2%
Hardware robustness	<5% degradation under ±10% fabrication variation
Readout error rates	1.04–2.81%

The system autonomously discovers feedback-driven control policies that stabilize entanglement under realistic dissipative dynamics.

Repository Structure
qsymphony/

phase1_hardware/
   gnn/
   scripts/
   pyepr/
   config/

phase2_quantum_sim/
   scripts/
   config/

phase3_rl_control/
   utils/
   scripts/
   models/
   config/

phase4_error_mitigation/
   scripts/
   checkpoints/
   config/

results/
   phase1/
   phase2/
   phase3/
   phase4/

paper/
   figures/
   tables/

reports/
   PHASE1_COMPLETION_REPORT.md
   PHASE2_COMPLETION_REPORT.md
   PHASE3_COMPLETION_REPORT.md
   PHASE4_COMPLETION_REPORT.md

zenodo_export/
   iq_data/
   models/
Installation

Clone the repository:

git clone https://github.com/raviraja1218/qsymphony.git
cd qsymphony

Install dependencies:

pip install -r requirements.txt
Running Experiments
Phase 1 — Hardware Topology Discovery
python phase1_hardware/gnn/train_sympgnn_fixed3.py
Phase 2 — Quantum Simulation
python phase2_quantum_sim/optimize_tms_final.py
Phase 3 — Reinforcement Learning Control
python phase3_rl_control/scripts/train_phase3_final.py
Phase 4 — Error Mitigation & Circuit Optimization
python phase4_error_mitigation/scripts/train_pinn_final.py
Reproducing Results

Generate figures and evaluation metrics used in the manuscript.

Run evaluation:

python phase3_rl_control/scripts/evaluate_agent_final.py

Run robustness sweep:

python phase3_rl_control/scripts/robustness_sweep_final.py

Evaluate PINN noise model:

python phase4_error_mitigation/scripts/evaluate_pinn_noise_final.py

Citation

If you use this repository in academic work, please cite:

Autonomous Entanglement Stabilization and Circuit Optimization
in Hybrid Superconducting–Mechanical Quantum Systems

Ravi Raja
2026
License

MIT License

Author

Ravi Raja
Quantum Computing Research
