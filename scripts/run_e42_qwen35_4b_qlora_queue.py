#!/usr/bin/env python3
"""Run four Qwen3.5-4B QLoRA variants after the queued E41 block succeeds."""
from __future__ import annotations
import argparse, json, os, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
PYTHON=Path('/home/zodrak/cuda13.x/bin/python')
E41_QUEUE=ROOT/'artifacts/training_queues/e41_qwen35_layer_removal/queue.json'
BASE_REVISION='1001bb4d826a52d1f399e183466143f4da7b741b'
MODEL_SNAPSHOT=Path.home()/'.cache/huggingface/hub/models--Qwen--Qwen3.5-4B-Base'/f'snapshots/{BASE_REVISION}'

def now(): return datetime.now(timezone.utc).isoformat(timespec='seconds')
def save(path,obj):
    path=Path(path); tmp=path.with_suffix(path.suffix+'.tmp'); tmp.write_text(json.dumps(obj,indent=2)+'\n'); os.replace(tmp,path)
def prerequisite():
    try:
        q=json.loads(E41_QUEUE.read_text())
        cached=(MODEL_SNAPSHOT/'config.json').is_file() and (MODEL_SNAPSHOT/'model.safetensors.index.json').is_file()
        ready=q.get('status')=='complete' and cached
        return ready, {'e41_status':q.get('status'),'e41_finished_at':q.get('finished_at'),
                       'base_model_snapshot_ready':cached,'base_model_snapshot':str(MODEL_SNAPSHOT)}
    except Exception as exc: return False, {'reason':str(exc)}
def gpu_idle():
    try:
        p=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
    except (FileNotFoundError,subprocess.CalledProcessError): return False,'nvidia-smi unavailable; holding queue'
    active=[x.strip() for x in p.stdout.splitlines() if x.strip()]
    return not active,active
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--queue',required=True); ap.add_argument('--poll-seconds',type=int,default=60); a=ap.parse_args()
    qp=Path(a.queue); qp=qp if qp.is_absolute() else ROOT/qp; q=json.loads(qp.read_text())
    q['worker_pid']=os.getpid(); q['worker_started_at']=now(); q['status']='queued_waiting_for_e41'; save(qp,q)
    while True:
        ready,dep=prerequisite(); q['dependency_check']=dep; q['last_dependency_check_at']=now()
        if ready:
            idle,gpu=gpu_idle(); q['gpu_check']=gpu
            if idle: break
            q['status']='queued_waiting_for_gpu'
        else: q['status']='queued_waiting_for_e41'
        save(qp,q); time.sleep(max(10,a.poll_seconds))
    if not PYTHON.is_file():
        q['status']='failed'; q['failure']=f'missing Python environment: {PYTHON}'; save(qp,q); raise SystemExit(q['failure'])
    env=os.environ.copy(); env.update({'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1'})
    q['status']='running'; q['started_at']=now(); save(qp,q)
    for job in q['jobs']:
        job['status']='running'; job['started_at']=now(); save(qp,q)
        cmd=[str(PYTHON),'-u','-m','src.pipeline','models','train','--model','qwen35_4b',
             '--dataset','v2.14','--validation-dataset','v0','--config',job['config'],
             '--seed','2026','--max-length','562','--accept-config-defaults','--refresh-checkpoint-catalog']
        job['command']=cmd; logpath=ROOT/job['log']; logpath.parent.mkdir(parents=True,exist_ok=True)
        with logpath.open('w') as log: proc=subprocess.run(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        job['finished_at']=now(); job['return_code']=proc.returncode; job['status']='complete' if proc.returncode==0 else 'failed'; save(qp,q)
        if proc.returncode:
            q['status']='failed'; q['failure']=f"{job['name']} exited {proc.returncode}"; q['finished_at']=now(); save(qp,q); raise SystemExit(q['failure'])
    q['status']='evaluating'; q['evaluation_started_at']=now(); save(qp,q)
    logpath=ROOT/q['evaluation_log']; logpath.parent.mkdir(parents=True,exist_ok=True)
    with logpath.open('w') as log: proc=subprocess.run([str(PYTHON),'-u','scripts/evaluate_e42_qwen35_4b_qlora.py','--queue',str(qp)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
    q['evaluation_return_code']=proc.returncode; q['finished_at']=now(); q['status']='complete' if proc.returncode==0 else 'evaluation_failed'; save(qp,q)
    if proc.returncode: raise SystemExit(f"E42 evaluation failed; see {logpath}")
if __name__=='__main__': main()
