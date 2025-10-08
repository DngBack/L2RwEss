# EVALUATION_GUIDE.md

# Hướng Dẫn Evaluation So Sánh với Paper Benchmarks

Tài liệu này hướng dẫn cách đánh giá mô hình AR-GSE và so sánh với các paper như **SelectiveNet** và **Learning to Reject Meets Long-tail Learning**.

---

## 📋 Tổng Quan

Chúng tôi cung cấp 3 script evaluation:

1. **`eval_comprehensive.py`** - Evaluation toàn diện với nhiều phương pháp tính confidence
2. **`eval_paper_benchmark.py`** - So sánh trực tiếp với paper benchmarks  
3. **`run_benchmark_evaluation.py`** - Script nhanh để chạy evaluation

---

## 🎯 Metrics Chính

### 1. AURC (Area Under Risk-Coverage Curve)
- **Định nghĩa**: Diện tích dưới đường cong Risk-Coverage
- **Công thức**: $\text{AURC} = \int_{c_{min}}^{c_{max}} \text{Risk}(c) \, dc$
- **Tốt hơn khi**: Thấp hơn (lower is better)
- **Range**: [0, 1]

### 2. E-AURC (Excess AURC)
- **Định nghĩa**: AURC vượt quá baseline ngẫu nhiên
- **Công thức**: $\text{E-AURC} = \text{AURC}_{\text{method}} - \text{AURC}_{\text{random}}$
- **Tốt hơn khi**: Âm (negative is better)

### 3. Coverage-Error Trade-offs
- Đánh giá hiệu suất tại các mức coverage cụ thể: 60%, 70%, 80%, 90%, 95%
- So sánh Standard Error, Balanced Error, Worst-Group Error

---

## 🚀 Cách Sử Dụng

### Phương án 1: Script nhanh (Khuyến nghị)

```bash
# Evaluation cơ bản
python run_benchmark_evaluation.py

# Với custom checkpoint
python run_benchmark_evaluation.py \
    --checkpoint ./checkpoints/my_model/gse_plugin.ckpt \
    --output ./my_results

# Chỉ định dataset và experts
python run_benchmark_evaluation.py \
    --checkpoint ./checkpoints/argse_balanced/cifar100_lt_if100/gse_plugin.ckpt \
    --dataset cifar100_lt_if100 \
    --experts ce_baseline logitadjust_baseline balsoftmax_baseline \
    --output ./benchmark_results_balanced
```

### Phương án 2: Python script chi tiết

```python
from src.train.eval_paper_benchmark import PaperBenchmarkEvaluator

config = {
    'dataset': {
        'name': 'cifar100_lt_if100',
        'splits_dir': './data/cifar100_lt_if100_splits',
        'num_classes': 100,
    },
    'experts': {
        'names': ['ce_baseline', 'logitadjust_baseline', 'balsoftmax_baseline'],
        'logits_dir': './outputs/logits',
    },
    'checkpoint_path': './checkpoints/argse_worst_eg_improved/cifar100_lt_if100/gse_balanced_plugin.ckpt',
    'output_dir': './paper_benchmark_results',
    'seed': 42
}

evaluator = PaperBenchmarkEvaluator(config)
results = evaluator.run_paper_benchmark()
```

### Phương án 3: Comprehensive evaluation (so sánh nhiều methods)

```python
from src.train.eval_comprehensive import ComprehensiveEvaluator

config = {
    'dataset': {
        'name': 'cifar100_lt_if100',
        'splits_dir': './data/cifar100_lt_if100_splits',
        'num_classes': 100,
    },
    'experts': {
        'names': ['ce_baseline', 'logitadjust_baseline', 'balsoftmax_baseline'],
        'logits_dir': './outputs/logits',
    },
    'checkpoint_path': './checkpoints/argse_worst_eg_improved/cifar100_lt_if100/gse_balanced_plugin.ckpt',
    'output_dir': './comprehensive_evaluation_results',
    'coverage_min': 0.2,
    'coverage_max': 1.0,
    'num_points': 81,
    'seed': 42
}

evaluator = ComprehensiveEvaluator(config)
results = evaluator.run_full_evaluation()
```

---

## 📊 Output Files

Sau khi chạy evaluation, bạn sẽ có các file:

### 1. `paper_benchmark_results.json`
```json
{
    "dataset": "cifar100_lt_if100",
    "num_test_samples": 10000,
    "method": "gse_margin",
    "aurc_metrics": {
        "standard": {
            "aurc": 0.123456,
            "eaurc": -0.012345
        },
        "balanced": {
            "aurc": 0.234567,
            "eaurc": -0.023456
        },
        "worst": {
            "aurc": 0.345678,
            "eaurc": -0.034567
        }
    },
    "oracle": {
        "aurc_standard": 0.056789,
        "aurc_balanced": 0.067890,
        "aurc_worst": 0.078901
    },
    "coverage_metrics": {
        "cov_0.60": {
            "standard_accuracy": 0.85,
            "balanced_accuracy": 0.82,
            "worst_accuracy": 0.78
        }
    }
}
```

### 2. `paper_benchmark_figures.png/pdf`
6-panel figure chất lượng publication:
- (a) Standard Error RC Curve
- (b) Balanced Error RC Curve  
- (c) Worst-Group Error RC Curve
- (d) Per-Group Selective Errors
- (e) AURC Metrics Comparison
- (f) Coverage-Accuracy Trade-off

### 3. `latex_table.tex`
Bảng LaTeX sẵn sàng cho paper submission:
```latex
\begin{table}[t]
\centering
\caption{Selective Classification Performance on CIFAR-100-LT}
\label{tab:main_results}
\begin{tabular}{lccc}
\toprule
\textbf{Method} & \textbf{AURC-Std} & \textbf{AURC-Bal} & \textbf{AURC-Worst} \\
\midrule
AR-GSE (Ours) & 0.1234 & 0.2345 & \textbf{0.3456} \\
...
\end{tabular}
\end{table}
```

### 4. `rc_curve_paper_benchmark.csv`
Raw data của RC curve để vẽ lại hoặc phân tích thêm

---

## 🔬 Confidence Scoring Methods

Framework hỗ trợ nhiều phương pháp tính confidence score:

### 1. GSE Margin (Khuyến nghị)
```python
confidence = compute_margin(eta_mix, alpha, mu, c=0.0, class_to_group)
```
- Sử dụng GSE decision function: $\text{score} - \text{threshold}$
- Phù hợp nhất với AR-GSE framework

### 2. GSE Raw Margin
```python
confidence = compute_raw_margin(eta_mix, alpha, mu, class_to_group)
```
- Không trừ rejection cost $c$

### 3. Maximum Posterior
```python
confidence = max_y P(y|x)
```
- Standard baseline từ SelectiveNet

### 4. Entropy-based
```python
confidence = -H(P(y|x)) = -Σ_y P(y|x) log P(y|x)
```
- Negative entropy làm confidence score

### 5. Margin (Top-2 difference)
```python
confidence = P(y_1|x) - P(y_2|x)
```
- Hiệu số giữa 2 xác suất cao nhất

---

## 📈 Cách Hiểu Kết Quả

### ✅ Kết quả tốt:
- **AURC thấp** (< 0.3 là tốt, < 0.2 là rất tốt)
- **E-AURC âm** (càng âm càng tốt so với random)
- **AURC gần Oracle** (< 2x Oracle là acceptable)
- **Worst-Group Error không quá cao** (< 2x Balanced Error)

### ⚠️ Cảnh báo:
- AURC > 0.5: Model không học được gì
- E-AURC > 0: Tệ hơn random rejection
- Worst-Group Error >> Balanced Error: Thiên lệch nghiêm trọng

### 📊 So sánh với Papers:

#### SelectiveNet (ICML 2019)
- **CIFAR-10**: AURC ≈ 0.02-0.03
- **CIFAR-100**: AURC ≈ 0.10-0.15
- **ImageNet**: AURC ≈ 0.05-0.08

#### Learning to Reject Meets Long-tail Learning (2024)
- **CIFAR-100-LT**: 
  - Balanced AURC ≈ 0.15-0.20
  - Worst-Group AURC ≈ 0.25-0.35
  - Coverage 0.7: Balanced Acc ≈ 0.75-0.80

---

## 🔍 Debug và Troubleshooting

### Lỗi thường gặp:

#### 1. File not found
```
FileNotFoundError: Logits file not found
```
**Giải pháp**: Kiểm tra đường dẫn logits:
```bash
ls -la outputs/logits/cifar100_lt_if100/*/test_lt_logits.pt
```

#### 2. AURC quá cao (> 0.5)
**Nguyên nhân**: Model chưa converge hoặc confidence scores không hợp lý

**Giải pháp**:
- Kiểm tra confidence score distribution:
```python
import torch
confidence = evaluator.compute_confidence_scores('gse_margin')
print(f"Min: {confidence.min():.3f}, Max: {confidence.max():.3f}")
print(f"Mean: {confidence.mean():.3f}, Std: {confidence.std():.3f}")
```
- Thử phương pháp khác: `max_posterior`, `entropy`

#### 3. Worst > Balanced > Standard không thỏa mãn
**Nguyên nhân**: Logic tính toán hoặc grouping không đúng

**Giải pháp**:
- Kiểm tra class_to_group mapping
- Verify per-group error computation

---

## 🎨 Customization

### Thay đổi coverage range:
```python
config['coverage_min'] = 0.0  # Start from 0%
config['coverage_max'] = 1.0  # Go to 100%
config['num_points'] = 101    # 101 points
```

### Thêm paper baseline:
```python
# Trong eval_paper_benchmark.py
def _load_paper_results(self):
    return {
        'selectivenet': {
            'cifar100': {
                'aurc_0.2_1.0': 0.12,  # Fill with actual values
                'accuracy_at_90': 0.85,
            }
        }
    }
```

### Custom visualization:
```python
# Modify plot_paper_style_figures() trong eval_paper_benchmark.py
def plot_paper_style_figures(self, results: Dict):
    # Your custom plotting code
    pass
```

---

## 📚 References

1. **SelectiveNet**: Geifman & El-Yaniv, "Selective Classification for Deep Neural Networks", ICML 2019
2. **Learning to Reject Meets Long-tail Learning**: Cao et al., 2024
3. **Rejection Option**: Chow, "On Optimum Recognition Error and Reject Tradeoff", IEEE Trans. 1970

---

## 💡 Tips

### Để có kết quả tốt nhất:

1. **Train nhiều checkpoints** và chọn checkpoint tốt nhất dựa trên validation AURC
2. **Thử nhiều confidence methods** để tìm phương pháp phù hợp nhất
3. **So sánh với Oracle** để biết upper bound
4. **Kiểm tra per-group fairness** (Head vs Tail)
5. **Bootstrap confidence intervals** nếu muốn statistical significance

### Để paper được accept:

1. **Báo cáo đầy đủ**: Standard, Balanced, Worst-Group metrics
2. **So sánh với baselines**: SelectiveNet, Random, Oracle
3. **Confidence intervals**: Report mean ± std over multiple runs
4. **Ablation studies**: Analyze α, μ impact
5. **Qualitative analysis**: Show sample rejection examples

---

## ✨ Example Results

Ví dụ output console:

```
================================================================================
PAPER BENCHMARK EVALUATION
Comparison with SelectiveNet & Learning to Reject Meets Long-tail Learning
================================================================================

✅ Initialized evaluator with 10000 test samples
✅ Loaded model with α=[0.9234, 1.1567], μ=[0.4123, 0.5234]

🔄 Computing confidence scores using: gse_margin
🔄 Generating Risk-Coverage curve...
🔄 Computing selective risk metrics...
🔄 Evaluating at specific coverage points...
🔄 Computing oracle baseline...

================================================================================
EVALUATION SUMMARY
================================================================================

📊 AURC Metrics (Coverage 0.2-1.0):
  Standard Error:
    AURC:    0.123456
    E-AURC:  -0.012345

  Balanced Error:
    AURC:    0.234567
    E-AURC:  -0.023456

  Worst-Group Error:
    AURC:    0.345678
    E-AURC:  -0.034567

📊 Oracle Baseline:
  Standard:    0.056789
  Balanced:    0.067890
  Worst-Group: 0.078901

📊 Metrics at Specific Coverages:

  Coverage ≈ 0.601 (target 0.60):
    Standard Accuracy:    0.8456
    Balanced Accuracy:    0.7890
    Worst-Group Accuracy: 0.7234

================================================================================

📊 Saved paper-style figures to ./paper_benchmark_results
📝 Saved LaTeX table to ./paper_benchmark_results/latex_table.tex
💾 Saved detailed results to ./paper_benchmark_results

================================================================================
✅ PAPER BENCHMARK EVALUATION COMPLETE
================================================================================
```

---

## 🤝 Support

Nếu gặp vấn đề:
1. Kiểm tra log messages để xác định lỗi
2. Verify input files (logits, labels, checkpoint)
3. Thử với confidence method đơn giản hơn (`max_posterior`)
4. Giảm `num_points` nếu chạy quá lâu

Good luck với paper submission! 🎉
