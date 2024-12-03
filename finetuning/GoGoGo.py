import os
import threading

# Set length
# src len, tgt len, 
task_length = { 
    "RefineSmall": [130,130],
    "RefineMedium": [250,250],
    "CodeReview": [512,400],
    "BFsmall": [512,130],
    "BFmdeium": [512,400],
    "trans": [512,512],
    'CodeRefinement': [512, 400],
}

def run(port, gpuid, model, model_tag, task, maxSL, maxTL, dataDir, BSize, GAcc=5, LR=5e-5):
    os.system(f'bash ./run.sh "{port}" "{gpuid}" "{model}" "{model_tag}" "{task}" "{maxSL}" "{maxTL}" "{dataDir}" "{BSize}" "{GAcc}" "{LR}"')

port = 8788
mtag = 'vbase300'
ctag = '64000'
base_dir = f'./models/pre-training-{mtag}'
model = f'{base_dir}/checkpoints-{ctag}/pytorch_model.bin'
tag = f'EvoCoder_{mtag}_{ctag}'
DTAG = 'Dataset'

task='CodeReview'
maxSL, maxTL = task_length[task]
dataDir = f'./{DTAG}/code-review'
BSize=10
GAcc=1
LR=1e-4
gpuid = 3
port = port+1
model_tag = f'{tag.replace("/","-")}'+f'-{str(BSize)}-{str(LR)}-{str(GAcc)}'
t = threading.Thread(target=run, args=(port, gpuid, model, model_tag, task, maxSL, maxTL, dataDir, BSize, GAcc, LR))
t.start()


# BugFix
task='BFsmall'
maxSL, maxTL = task_length[task]
dataDir = f'./{DTAG}/bf-small'
BSize=20
GAcc=1
LR=3e-4
gpuid = 3
port = port+1
model_tag = f'{tag.replace("/","-")}'+f'-{str(BSize)}-{str(LR)}-{str(GAcc)}'
t = threading.Thread(target=run, args=(port, gpuid, model, model_tag, task, maxSL, maxTL, dataDir, BSize, GAcc, LR))
t.start()

DTAG = 'Dataset'
task='BFmedium'
maxSL, maxTL = task_length[task]
dataDir = f'./{DTAG}/bf-medium'
BSize=12
GAcc=1
LR = 1e-4
gpuid = 3
port = port+1
model_tag = f'{tag.replace("/","-")}'+f'-{str(BSize)}-{str(LR)}-{str(GAcc)}'
t = threading.Thread(target=run, args=(port, gpuid, model, model_tag, task, maxSL, maxTL, dataDir, BSize, GAcc, LR))
t.start()

# Trans
task='trans'
DTAG = 'Dataset'
maxSL, maxTL = task_length[task]
dataDir = f'./{DTAG}/code-trans-data/;j2c'
GAcc=1
BSize=30
gpuid = 1
port = port+1
task = 'TransJ2C'
LR=2e-4
model_tag = f'{tag.replace("/","-")}'+f'-{str(BSize)}-{str(LR)}-{str(GAcc)}'
t = threading.Thread(target=run, args=(port, gpuid, model, model_tag, task, maxSL, maxTL, dataDir, BSize, GAcc, LR))
t.start()

dataDir = f'./{DTAG}/code-trans-data/;c2j'
BSize=30
gpuid = 1
port = port+1
task = 'TransC2J'
model_tag = f'{tag.replace("/","-")}'+f'-{str(BSize)}-{str(LR)}-{str(GAcc)}'
t = threading.Thread(target=run, args=(port, gpuid, model, model_tag, task, maxSL, maxTL, dataDir, BSize, GAcc, LR))
t.start()

# Refinement
task='RefineSmall'
maxSL, maxTL = task_length[task]
DTAG = 'Dataset'
dataDir = f'./{DTAG}/code-refinement/small'
GAcc=1
LR=3e-4
BSize=120
gpuid = 2
port = port+1
model_tag = f'{tag.replace("/","-")}'+f'-{str(BSize)}-{str(LR)}-{str(GAcc)}'
t = threading.Thread(target=run, args=(port, gpuid, model, model_tag, task, maxSL, maxTL, dataDir, BSize, GAcc, LR))
t.start()

task='RefineMedium'
maxSL, maxTL = task_length[task]
DTAG = 'Dataset'
dataDir = f'./{DTAG}/code-refinement/medium'
BSize=50
LR=3e-4
GAcc=1
gpuid = 3
port = port+1
model_tag = f'{tag.replace("/","-")}'+f'-{str(BSize)}-{str(LR)}-{str(GAcc)}'
t = threading.Thread(target=run, args=(port, gpuid, model, model_tag, task, maxSL, maxTL, dataDir, BSize, GAcc, LR))
t.start()

task='CodeRefinement'
maxSL, maxTL = task_length[task]
DTAG = 'Dataset'
dataDir = f'./{DTAG}/CodeRefinement'
BSize=30
LR=2e-4
GAcc=1
gpuid = 3
port = port+1
model_tag = f'{tag.replace("/","-")}'+f'-{str(BSize)}-{str(LR)}-{str(GAcc)}'
t = threading.Thread(target=run, args=(port, gpuid, model, model_tag, task, maxSL, maxTL, dataDir, BSize, GAcc, LR))
t.start()
