# Extension of Propositon 1

To produce the plots shown in Figure 1, you need to walkthrough the following 4 sections:

## NCCC evaluation

```python
python src/nccc_eval.py --config <path-to-config-yaml> --ckpt_path <checkpoints_dir or checkpoints_file> --output_path <output-dir>
```

This shall results in `results.csv` file in your output directory. 

## CDNV evaluation

```python
python src/cdnv_eval.py --config <path-to-config-yaml> --ckpt_path <checkpoints_dir or checkpoints_file> --output_path <output-dir>
```

This shall results in `cdnv.csv` file in your output directory. 

## Pairwise geometric metrics for new error bound

```python
python src/bound_eval.py --config <path-to-config-yaml> --ckpt_path <checkpoints_dir or checkpoints_file> --output_path <output-dir>
```

This shall results in `train_pairwise_metrics.json` and `test_pairwise_metrics.json` files in your output directory.


## Producing plots

Please refer to [error_bounds.ipynb](https://github.com/DLFundamentals/directional_cdnv_bounds/blob/main/notebooks/error_bounds.ipynb) notebook where you will need to provide correct path to the metrics generated in the above three steps.
