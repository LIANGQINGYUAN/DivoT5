import os
import threading

# Set length
# src len, tgt len, 
task_length = { 
    "RefineSmall": [130,130],
    "RefineMedium": [250,250],
    "CodeReview": [512,400],
    "CommentUpdate": [512,150],
    "CU": [512,100],
    "BFsmall": [512,130],
    "BFmdeium": [512,400],
    "trans": [512,512],
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
