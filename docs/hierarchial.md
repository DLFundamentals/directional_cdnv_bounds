## Hierarchial Clustering Experiment


```python
python src/granular_cdnv.py --config <path-to-config-yaml> --ckpt_dir <checkpoints_dir or checkpoints_file> --output_path <output-dir> --label_level super --superclass_mapping_json <path-to-mapping-json-file>
```

This shall results in `cdnv.csv` file in your output directory with respect to the 10 superclasses. 


To plot the dynamics of training for both fine-grained and clustered classes, please refer to [cdnv_plotter.ipynb]().
