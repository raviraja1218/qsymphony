# Q-SYMPHONY  
### Autonomous Entanglement Stabilization and Circuit Optimization in Hybrid Superconducting–Mechanical Quantum Systems

Q-SYMPHONY is an end-to-end AI-driven quantum hardware design and control framework for hybrid superconducting–mechanical systems. The project integrates geometric machine learning, reinforcement learning, and physics-informed neural networks to autonomously design, simulate, and control multimode quantum architectures.

The framework discovers robust quantum control strategies that stabilize entanglement and reduce circuit complexity in noisy quantum environments.

---

## Key Results

- **Steady-state entanglement stabilization:**  
  \(E_N = 0.506 \pm 0.027\)

- **Reinforcement-learning discovery of control policies**  
  exploiting weak continuous measurement backaction.

- **Circuit depth reduction:**  
  **43.2% improvement** over the standard Qiskit transpiler.

- **Hardware-robust architecture**  
  tolerant to **±10% fabrication variations**.

---

## Repository Structure


---

## Project Pipeline

The Q-SYMPHONY framework consists of four major phases:

### Phase 1 — Hardware Topology Discovery
Symplectic Graph Neural Networks (SympGNN) optimize chip layouts while preserving Hamiltonian physics constraints.

Outputs:

- optimized transmon-mechanical layout
- electromagnetic parameters via pyEPR
- fabrication robustness analysis

---

### Phase 2 — Quantum System Simulation
Open-system dynamics simulated using **QuTiP**.

Features:

- stochastic master equation solver
- Hilbert-space truncation validation
- Wigner function analysis
- entanglement metric evaluation

---

### Phase 3 — Reinforcement Learning Control
A **Recurrent PPO agent** learns adaptive control policies under continuous measurement.

Agent discovers dynamic detuning strategies that stabilize remote entanglement.

Final performance:


across three independent seeds.

---

### Phase 4 — Error Mitigation & Circuit Optimization

Physics-Informed Neural Networks (PINNs) generate noise-aware control pulses and identify Liouvillian exceptional points.

Achievements:

- **43.2% reduction** in parity-check circuit depth
- machine-learning-assisted readout calibration
- noise robustness validation

---

## Technologies Used

- Python 3.10
- PyTorch
- QuTiP
- NumPy
- Qiskit
- CUDA GPU acceleration
- PyEPR electromagnetic simulation

---

## Installation

Clone the repository:

```bash
git clone https://github.com/raviraja1218/qsymphony.git
cd qsymphony
pip install -r requirements.txt
Running the Pipeline

Example workflow:

Hardware optimization
python phase1_hardware/gnn/train_sympgnn_fixed3.py
Quantum simulation
python phase2_quantum_sim/optimize_tms_final.py
RL training
python phase3_rl_control/scripts/train_phase3_final.py
Error mitigation and circuit optimization
python phase4_error_mitigation/scripts/train_pinn_final.py
Figures

Main figures used in the manuscript are located in:

results/

including:

hardware topology optimization

entanglement stabilization curves

exceptional point dynamics

circuit depth comparison

All source code used for simulations, reinforcement learning training, and circuit optimization is available in this repository.

Citation

If you use this code, please cite:


Q-SYMPHONY: Autonomous Entanglement Stabilization and Circuit Optimization
in Hybrid Superconducting–Mechanical Quantum Systems.

License

This project is released under the MIT License.

Author

Ravi Raja

Quantum AI Research


---

# 3️⃣ Push README to GitHub

```bash
git add README.md
git commit -m "Add project README"
git push
