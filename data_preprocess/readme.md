1. Get the diff information
- input: commitpack (java)
- output: diff_java
```
python c1_get_diff.py
```

2. The diffusion process
- input: diff_java
- output: diff_java_diffusion

```
python c2_diffusion.py
```

3. The length filter
- input: diff_java_diffusion
- output: diff_java_diffusion_len_constrain

```
python c3_len_constrain.py
```

3. Merge all data
- input: diff_java_diffusion_len_constrain + ccn + commitpack (java)
- output: pre-training dataset

```
c4_merge.ipynb
```
