Get data command 

```bash
 python -c "from src.data.enhanced_datasets import create_full_cifar100_lt_splits; create_full_cifar100_lt_splits()"
```

train_expert.py

```bash
 python -m src.train.train_expert
```

train_argse.pu 

```bash
 python -m src.train.train_argse
```

run eval 

```bash
 python -m src.train.eval_argse
```


## Plugin 
 Balanced training + per-group thresholds
```bash
python -m src.train.train_gating_only
python -m src.train.gse_balanced_plugin
python -m src.train.eval_gse_plugin
```

## Plugin 
 Worst-group expert optimization + per-group thresholds

```bash
python -m src.train.train_gating_only
python run_gse_worst_eg.py                 # NEW EG-outer optimization
python -m src.train.eval_gse_plugin
```