# Ablation Experiments
| **Models** | **CodeReview EM** | **BugFix-Small EM** | **BugFix-Medium EM** | **NL-CodeRefinement EM** | **Refine-Small EM** | **Refine-Medium EM** | **Java-CShapr EM** | **CSharp-Java EM** |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DivoT5-small~(60M) | **40.16** | **35.84** | **28.72** | **29.42** | **22.31** | **14.62** | **68.30** | **72.50** |
| w/o KSM | 35.45 | 35.28 | 27.36 | 28.69 | 21.75 | 13.67 | 68.00 | 71.20 |
| w/o RM | 37.02 | 35.30 | 28.10 | 28.61 | 21.54 | 13.91 | 66.30 | 69.80 |
| w/o EDAE | 35.22 | 35.50 | 27.65 | 28.74 | 20.99 | 13.86 | 67.50 | 68.60 |
| w/o EDR | 37.31 | 34.76 | 28.37 | 28.72 | 21.68 | 14.03 | 67.70 | 70.20 |
| w/o Artificial Noise |  36.61|  35.52|  28.40|  28.85|  21.41|  13.80|  67.00|  68.60|
| w/o ALL | 34.75 | 35.35 | 27.36 | 28.50 | 20.84 | 13.41 | 67.60 | 68.20 |


# Experimental Records

The experimental setting and summary for NL-based Code Refinement dataset.

Setting:
```
{'add_lang_ids': False,
 'add_task_prefix': False,
 'always_save_model': False,
 'batch_size_per_replica': 12,
 'beam_size': 10,
 'cache_path': './saved_models/CodeRefinement_CodeT5base-12-0.0001-1/cache_file',
 'config_name': '',
 'data_dir': './Dataset/CodeRefinement',
 'data_num': -1,
 'do_eval': True,
 'do_eval_bleu': True,
 'do_test': True,
 'do_train': True,
 'epochs': 20,
 'from_scratch': False,
 'grad_acc_steps': 1,
 'load': 'Salesforce/codet5-base',
 'load_model_path': None,
 'local_rank': -1,
 'lr': 0.0001,
 'lr_warmup_steps': 500,
 'max_source_len': 512,
 'max_target_len': 400,
 'model_name_or_path': 'roberta-base',
 'model_type': 'codet5',
 'output_dir': './saved_models/CodeRefinement_CodeT5base-12-0.0001-1',
 'patience': 3,
 'seed': 42,
 'task': 'CodeRefinement',
 'tokenizer_name': 'roberta-base'}
 ```

Summary of CodeT5:
```
[0] Best bleu+em+cbleu changed into 179.92 (bleu: 76.95, em: 23.98, cbleu: 78.99)
[1] Best bleu+em+cbleu changed into 183.13 (bleu: 77.72, em: 25.77, cbleu: 79.64)
[2] Best bleu+em+cbleu changed into 185.55 (bleu: 78.30, em: 26.99, cbleu: 80.26)
[3] Best bleu+em+cbleu changed into 186.19 (bleu: 78.33, em: 27.54, cbleu: 80.32)
[4] Best bleu+em+cbleu (186.19) does not drop changed for 1 epochs, cur bleu+em+cbleu: 186.18 (bleu: 78.07, em: 28.15, cbleu: 79.96)
[5] Best bleu+em+cbleu (186.19) does not drop changed for 2 epochs, cur bleu+em+cbleu: 186.10 (bleu: 78.31, em: 27.51, cbleu: 80.28)
[6] Best bleu+em+cbleu changed into 186.63 (bleu: 78.61, em: 27.38, cbleu: 80.64)
[7] Best bleu+em+cbleu (186.63) does not drop changed for 1 epochs, cur bleu+em+cbleu: 186.42 (bleu: 78.35, em: 27.67, cbleu: 80.40)
[8] Best bleu+em+cbleu (186.63) does not drop changed for 2 epochs, cur bleu+em+cbleu: 186.22 (bleu: 78.09, em: 27.96, cbleu: 80.17)
[9] Best bleu+em+cbleu changed into 187.30 (bleu: 78.48, em: 28.34, cbleu: 80.48)
[10] Best bleu+em+cbleu (187.30) does not drop changed for 1 epochs, cur bleu+em+cbleu: 185.80 (bleu: 77.90, em: 28.03, cbleu: 79.87)
[11] Best bleu+em+cbleu (187.30) does not drop changed for 2 epochs, cur bleu+em+cbleu: 187.13 (bleu: 78.46, em: 28.02, cbleu: 80.65)
[12] Best bleu+em+cbleu (187.30) does not drop changed for 3 epochs, cur bleu+em+cbleu: 187.02 (bleu: 78.44, em: 28.07, cbleu: 80.51)
[13] Best bleu+em+cbleu (187.30) does not drop changed for 4 epochs, cur bleu+em+cbleu: 187.16 (bleu: 78.38, em: 28.29, cbleu: 80.49)
[13] Early stop as not_bleu_em_inc_cnt=4, and not_loss_dec_cnt=0
[best-bleu] bleu-4: 78.64, em: 29.44, codebleu: 80.56
Finish and take 44h52m
```

# Java and CSharp Examples
**example 1:**

java 
```java
public static  String   stripExtension   ( String   filename    )   { int   idx   = filename   . indexOf   ( '.'  )       ;  if ( idx   !=  -  1      )   { filename   =  filename   . substring   ( 0  , idx  )      ;  }    return filename  ;  }
```

csharp
```csharp
public  static  string   StripExtension   ( string   filename    )   { int   idx  = filename   . IndexOf     ( '.'   )       ;  if ( idx   !=  - 1      ) { filename   =  filename   . Substring     ( 0   , idx   )      ;  }    return filename  ;  }
```

**example 2:**

java
```java
public  java  . nio   . ByteBuffer    encode   ( string   s    )   { return encode   ( java   . nio     . CharBuffer     . wrap     ( java   . lang     . CharSequenceProxy     . Wrap     ( s   )     )     )    ;  }   
```

csharp
```csharp
public final  ByteBuffer   encode   ( String   s    )   { return encode   ( CharBuffer   . wrap   ( s  )    )    ;  }   
```

**example 3:**

java
```java
public  boolean   canEncode   ( )   { return true ;  }
```

csharp
```csharp
public  virtual  bool   canEncode   ( )   { return true  ;  }   
```


# Algorithm of Constructing Code Changes


We use the following notations and algorithm to provide a more detailed and clear explanation of how our diffusion strategy constructs the data for each step. 

> Since mathematical formulas are difficult to display in anonymous links, we have included screenshots instead.


![diffusion](../images/diffusion_step.png)
