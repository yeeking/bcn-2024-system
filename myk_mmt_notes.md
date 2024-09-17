## Notes on preparing data and training mmt

This does the first big converrsion on the dataset: 
```
python mmt/convert_sod.py -j 20
```

This does another conversion: 
```
python mmt/extract.py -d sod
```

This does something else necessary:
```
python mmt/split.py -d sod
```

This runs the training with a batch size of 4 which is the most my 2070 can handle
```
python mmt/train.py -d sod -o exp/sod/rpe --no-abs_pos_emb --rel_pos_emb  -g 0 -bs 4
```

This generates example outputs:
```
python mmt/generate.py -d sod -o exp/sod/npe -g 0 -ns 1
```
- if you set ns to 1, it gnerates one set of examples, based on the first item in the dataset I think

