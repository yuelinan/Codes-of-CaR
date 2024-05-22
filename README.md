Official implementation of "Complement-aware Self-explaining GNNs for Graph Classification".

## Data download
- Spurious-Motif: this dataset can be generated via `spmotif_gen/spmotif.ipynb` in [DIR](https://github.com/Wuyxin/DIR-GNN/tree/main). 
- Open Graph Benchmark (OGB): this dataset can be downloaded when running car.sh.


## How to run CaR?

To train CaR on OGB dataset:

```python
sh car.sh
```

To train CaR on Spurious-Motif dataset:

```python
# cd spmotif_codes
sh car.sh
```



