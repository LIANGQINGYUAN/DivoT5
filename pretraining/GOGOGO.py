import os
import threading

def run(tag):
    if 'base' in tag:
        LOG=f'.pretraining-{tag}-log.txt'
        os.system(f'bash scripts/pre-train-{tag}.sh 2>&1 | tee {LOG}')
    else:
        LOG=f'.pretraining-small-{tag}-log.txt'
        # os.system(f'bash scripts/pre-train-small-wo-{tag}.sh 2>&1 | tee {LOG}')
        os.system(f'bash scripts/pre-train-small-{tag}.sh 2>&1 | tee {LOG}')

tag = 'vbase300'   
t = threading.Thread(target=run, args=(tag,))
t.start()