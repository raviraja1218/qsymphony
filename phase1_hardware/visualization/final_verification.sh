#!/bin/bash
echo "=================================================="
echo "PHASE 1 FINAL VERIFICATION - ALL FILES CHECK"
echo "=================================================="

# Check Figure 1a
echo -n "Figure 1a (3D render): "
if [ -f ~/projects/qsymphony/results/phase1/figures/fig1a_3d_render_final.png ]; then
    SIZE=$(du -h ~/projects/qsymphony/results/phase1/figures/fig1a_3d_render_final.png | cut -f1)
    echo "✅ FOUND ($SIZE)"
else
    echo "❌ MISSING"
fi

# Check Figure 1b
echo -n "Figure 1b (GNN arch): "
if [ -f ~/projects/qsymphony/results/phase1/figures/fig1b_sympgnn_arch.png ]; then
    SIZE=$(du -h ~/projects/qsymphony/results/phase1/figures/fig1b_sympgnn_arch.png | cut -f1)
    echo "✅ FOUND ($SIZE)"
else
    echo "❌ MISSING"
fi

# Check heatmaps count
echo -n "Figure 1c heatmaps: "
COUNT=$(ls -1 ~/projects/qsymphony/results/phase1/epr_results/heatmap_*.png 2>/dev/null | wc -l)
if [ $COUNT -eq 100 ]; then
    echo "✅ 100 heatmaps found"
else
    echo "❌ Found $COUNT/100"
fi

# Check EPR summary
echo -n "EPR summary CSV: "
if [ -f ~/projects/qsymphony/results/phase1/epr_results/epr_summary_top100.csv ]; then
    SIZE=$(du -h ~/projects/qsymphony/results/phase1/epr_results/epr_summary_top100.csv | cut -f1)
    echo "✅ FOUND ($SIZE)"
else
    echo "❌ MISSING"
fi

# Check hardware parameters
echo -n "Hardware params for Phase 2: "
if [ -f ~/projects/qsymphony/phase2_quantum_sim/hardware_params.json ]; then
    SIZE=$(du -h ~/projects/qsymphony/phase2_quantum_sim/hardware_params.json | cut -f1)
    echo "✅ FOUND ($SIZE)"
    
    # Extract and display key parameters
    echo ""
    echo "📊 OPTIMAL LAYOUT PARAMETERS:"
    python3 -c "
import json
with open('~/projects/qsymphony/phase2_quantum_sim/hardware_params.json') as f:
    p = json.load(f)
print(f\"  Layout ID: {p['metadata']['layout_id']}\")
print(f\"  Confinement: {p['metadata']['confinement_percent']}%\")
print(f\"  Qubit freq: {p['qubit']['frequency_ghz']} GHz\")
print(f\"  g0: {p['couplings']['g0_qubit_mech_mhz']} MHz\")
print(f\"  EC: {p['qubit']['ec_ghz']} GHz\")
print(f\"  EJ: {p['qubit']['ej_ghz']} GHz\")
print(f\"  EJ/EC: {p['qubit']['ej_ghz']/p['qubit']['ec_ghz']:.1f}\")
"
else
    echo "❌ MISSING"
fi

# Check GNN model
echo -n "GNN model file: "
if [ -f ~/projects/qsymphony/results/models/sympgnn_best_opt.pt ]; then
    SIZE=$(du -h ~/projects/qsymphony/results/models/sympgnn_best_opt.pt | cut -f1)
    echo "✅ FOUND ($SIZE)"
else
    echo "❌ MISSING"
fi

# Check training logs
echo -n "Training logs: "
if [ -f ~/projects/qsymphony/results/phase1/training_logs/training_logs.json ]; then
    echo "✅ FOUND"
    echo ""
    echo "📈 GNN PERFORMANCE:"
    python3 -c "
import json
with open('~/projects/qsymphony/results/phase1/training_logs/training_logs.json') as f:
    logs = json.load(f)
print(f\"  Best validation loss: {logs['best_val_loss']}\")
print(f\"  Constraint satisfaction: {logs['constraint_satisfaction']}%\")
print(f\"  Target achieved: {logs['target_achieved']}\")
"
else
    echo "❌ MISSING"
fi

echo ""
echo "=================================================="
if [ -f ~/projects/qsymphony/phase2_quantum_sim/hardware_params.json ] && \
   [ -f ~/projects/qsymphony/results/phase1/figures/fig1a_3d_render_final.png ] && \
   [ -f ~/projects/qsymphony/results/phase1/figures/fig1b_sympgnn_arch.png ] && \
   [ $(ls -1 ~/projects/qsymphony/results/phase1/epr_results/heatmap_*.png 2>/dev/null | wc -l) -eq 100 ]; then
    echo "✅✅✅ PHASE 1 COMPLETE - ALL FILES VERIFIED ✅✅✅"
    echo ""
    echo "🚀 Ready to proceed to PHASE 2!"
    echo "cd ~/projects/qsymphony/phase2_quantum_sim/"
else
    echo "⚠️ Some files missing - please check above"
fi
echo "=================================================="
