## Inference and Visualization

### 🎯 Demo Inference (Quick Analysis)
For quick demonstration and understanding of a few samples:

```powershell
python demo_inference.py
```

**Output:**
- `inference_analysis_results/demo_sample_X.png` - Individual sample visualizations
- `inference_analysis_results/demo_sample_X_summary.txt` - Detailed text analysis  
- `inference_analysis_results/overall_analysis.txt` - Overall summary

### 🚀 Comprehensive Inference (Research-Grade Analysis)
For comprehensive analysis with 50 samples (30 Head + 20 Tail):

```powershell
python comprehensive_inference.py
```

**Output Structure:**
```
comprehensive_inference_results/
├── inference_results.json          # Raw data (50 samples)
├── inference_results.csv           # Tabular format for analysis
├── comprehensive_summary.txt       # Statistical analysis
├── comprehensive_analysis.png      # Main dashboard visualization
├── sample_highlights.png           # Interesting cases overview
└── individual_samples/             # Detailed per-sample analysis
    ├── sample_X.png               # Individual visualizations (4-panel)
    ├── sample_X_summary.txt       # Mathematical explanations
    └── individual_analysis_summary.txt
```

### 📊 Analysis Results Summary
To get a quick summary of the comprehensive results:

```powershell
python analyze_results.py
```

### 🔍 Compare Analysis Methods
To compare demo vs comprehensive inference:

```powershell
python compare_inference_results.py
```

### Visualization Features

#### Individual Sample Analysis (Both Scripts)
Each sample visualization includes:
1. **Expert Posteriors**: How each expert (CE, LogitAdj, BalSoftmax) assigns probabilities across all 100 classes
2. **Gating Weights**: Which expert the model trusts most for this sample
3. **Mixture Distribution**: Final combined probabilities after expert weighting
4. **Decision Process**: Margin calculation leading to accept/reject decision

#### Text Summaries Include
- Ground truth information
- Expert predictions and confidence scores
- Gating network weights and dominant expert
- Mixture results and prediction accuracy
- Selective decision process with mathematical details
- Step-by-step margin calculations
- Overall evaluation and decision quality assessment

#### Dashboard Visualizations (Comprehensive Only)
- Accuracy comparison by group (Head vs Tail)
- Acceptance rates and decision quality matrix
- Expert weight distributions
- Margin distributions and accept probability analysis

### Use Case Recommendations

| Script | Best For | Sample Size | Output |
|--------|----------|-------------|---------|
| `demo_inference.py` | Quick demos, education, debugging | 3 samples | 5 files |
| `comprehensive_inference.py` | Research, evaluation, publication | 50 samples | 27+ files |
| `analyze_results.py` | Quick summary of comprehensive results | - | Console output |

### Sample Output Interpretation

**Decision Quality Categories:**
- ✅ **Correct Accept**: High confidence + Correct prediction (Good)  
- ❌ **Incorrect Accept**: High confidence + Wrong prediction (Bad)
- ✅ **Correct Reject**: Low confidence + Wrong prediction (Good)
- ❌ **Incorrect Reject**: Low confidence + Correct prediction (Bad)

**Margin Interpretation:**
- `Raw Margin > 0`: Model confident about prediction
- `Margin with Cost < 0`: Rejection cost outweighs benefits → REJECT
- `Accept Prob > 0.5`: High confidence → ACCEPT prediction