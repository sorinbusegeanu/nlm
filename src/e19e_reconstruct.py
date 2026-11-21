"""Resumable E19-E prompt reconstruction with the local Qwen3.5-9B Q4_K_M GGUF."""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import socket
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

from . import e19b_reconstruct
from .pipeline import _dataset_rows

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/analysis/e19e"
MODEL_ID = "unsloth/Qwen3.5-9B-GGUF"
MODEL_REVISION = "3885219b6810b007914f3a7950a8d1b469d598a5"
MODEL_FILE_SHA256 = "03b74727a860a56338e042c4420bb3f04b2fec5734175f4cb9fa853daf52b7e8"
PROMPT_VERSION = "e19e_prompt_v1"
DEFAULT_BATCHES = (10, 8, 6, 4, 2)
MAX_INPUT_TOKENS = e19b_reconstruct.MAX_INPUT_TOKENS
MAX_NEW_TOKENS = 1024
RETRY_ATTEMPTS = 3
LLAMA_CPP_ROOT = Path.home() / ".cache/llama.cpp"
LLAMA_SERVER = LLAMA_CPP_ROOT / "build/bin/llama-server"
GGUF_PATH = (Path.home() / ".cache/huggingface/hub/models--unsloth--Qwen3.5-9B-GGUF"
    / f"snapshots/{MODEL_REVISION}/Qwen3.5-9B-Q4_K_M.gguf")


def _model_path(path=None):
    path = Path(path or GGUF_PATH)
    if not path.is_file():
        raise FileNotFoundError(f"E19-E GGUF checkpoint not found: {path}")
    return path.resolve()


def _free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _http_json(url, payload=None, *, timeout=60):
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data,
        headers={"Content-Type": "application/json"} if data is not None else {})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {body}") from exc


def _oom_text(value):
    text = str(value).casefold()
    return any(term in text for term in (
        "out of memory", "cuda error", "failed to allocate", "allocation failed",
        "ggml_backend_sched_reserve", "not enough memory", "cublas_status_alloc_failed"))


def _tail(path, max_bytes=16000):
    try:
        with Path(path).open("rb") as handle:
            handle.seek(max(0, Path(path).stat().st_size - max_bytes))
            return handle.read().decode("utf-8", errors="replace")
    except OSError:
        return ""


class LlamaServer:
    """One local llama.cpp server with N concurrent request slots."""
    def __init__(self, *, model_path, parallelism, log_path, port=None,
                 max_input_tokens=MAX_INPUT_TOKENS, max_new_tokens=MAX_NEW_TOKENS,
                 batch_tokens=4096, micro_batch_tokens=512, n_gpu_layers="all"):
        self.model_path = Path(model_path)
        self.parallelism = int(parallelism)
        self.log_path = Path(log_path)
        self.port = int(port or _free_port())
        self.ctx_per_slot = int(max_input_tokens + max_new_tokens + 256)
        self.context_size = self.ctx_per_slot * self.parallelism
        self.batch_tokens = int(batch_tokens)
        self.micro_batch_tokens = int(micro_batch_tokens)
        self.n_gpu_layers = n_gpu_layers
        self.process = None
        self._log_handle = None

    @property
    def base_url(self):
        return f"http://127.0.0.1:{self.port}"

    def start(self, *, startup_timeout=240):
        if not LLAMA_SERVER.is_file():
            raise FileNotFoundError(f"Build llama.cpp first; missing {LLAMA_SERVER}")
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_handle = self.log_path.open("ab", buffering=0)
        command = [str(LLAMA_SERVER), "--model", str(self.model_path),
            "--host", "127.0.0.1", "--port", str(self.port),
            "--n-gpu-layers", str(self.n_gpu_layers),
            "--ctx-size", str(self.context_size), "--parallel", str(self.parallelism),
            "--batch-size", str(self.batch_tokens), "--ubatch-size", str(self.micro_batch_tokens),
            "--flash-attn", "on", "--jinja",
            "--chat-template-kwargs", '{"enable_thinking":false}',
            "--no-webui", "--no-warmup"]
        self.process = subprocess.Popen(command, stdout=self._log_handle,
            stderr=subprocess.STDOUT, env={**os.environ,"GGML_CUDA_ENABLE_UNIFIED_MEMORY":"0"})
        deadline = time.monotonic() + float(startup_timeout)
        last_error = None
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                self.stop()
                log = _tail(self.log_path)
                if _oom_text(log):
                    raise RuntimeError("CUDA_OUT_OF_MEMORY during llama-server startup\n"+log[-4000:])
                raise RuntimeError("llama-server exited during startup\n"+log[-4000:])
            try:
                _http_json(self.base_url+"/health", timeout=2)
                return {"pid":self.process.pid,"port":self.port,
                        "parallelism":self.parallelism,"context_size":self.context_size,
                        "command":command}
            except Exception as exc:
                last_error = exc
                time.sleep(.5)
        log = _tail(self.log_path)
        self.stop()
        if _oom_text(log):
            raise RuntimeError("CUDA_OUT_OF_MEMORY during llama-server startup\n"+log[-4000:])
        raise RuntimeError(f"llama-server health timeout ({last_error})\n"+log[-4000:])

    def request(self, message, *, max_tokens=MAX_NEW_TOKENS, seed=2026, timeout=900):
        payload={"model":"qwen3.5-9b-q4km",
            "messages":[{"role":"user","content":message}],
            "max_tokens":int(max_tokens),"temperature":0.0,"top_p":1.0,
            "seed":int(seed),"stream":False}
        result=_http_json(self.base_url+"/v1/chat/completions",payload,timeout=timeout)
        return result["choices"][0]["message"]["content"]

    def stop(self):
        if self.process is not None:
            if self.process.poll() is None:
                self.process.terminate()
                try:self.process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    self.process.kill();self.process.wait(timeout=10)
            self.process=None
        if self._log_handle is not None:
            self._log_handle.close();self._log_handle=None


def _tokenizer():
    """Reuse the cached Qwen3.5 tokenizer to cap long input prompts consistently."""
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained("Qwen/Qwen3.5-2B",
        revision=e19b_reconstruct.MODEL_REVISION,local_files_only=True,padding_side="right")


def _messages(text, tokenizer, max_input_tokens=MAX_INPUT_TOKENS):
    instruction=e19b_reconstruct.RECONSTRUCTION_INSTRUCTION
    content=(instruction+"\n\nAssistant response to analyze:\n<response>\n"+
             str(text)+"\n</response>")
    ids=tokenizer.encode(content,add_special_tokens=False)
    # Reserve the Qwen chat-template tokens and the assistant generation prefix.
    budget=int(max_input_tokens)-96
    if len(ids)>budget:
        content=tokenizer.decode(ids[:budget],skip_special_tokens=False)
    return content


def _load_v0(root):
    frame=_dataset_rows("v0",Path(root)/"data/processed")
    frame=frame.sort_values("id",key=lambda values:values.astype(str),kind="stable")
    return frame.id.astype(str).to_numpy(),frame.text.fillna("").astype(str).to_numpy()


def reconstruct_v0(root=ROOT, *, model_path=None, cache_path=None, output_path=None,
                   batch_sizes=DEFAULT_BATCHES, limit=None, max_new_tokens=MAX_NEW_TOKENS,
                   max_attempts=RETRY_ATTEMPTS, seed=2026, startup_timeout=240,
                   request_timeout=900, log_dir=None):
    if not batch_sizes or tuple(int(x) for x in batch_sizes)!=tuple(sorted(
            set(int(x) for x in batch_sizes),reverse=True)):
        raise ValueError("batch_sizes must be unique and descending, e.g. 10,8,6,4,2")
    if any(x not in DEFAULT_BATCHES for x in batch_sizes):
        raise ValueError(f"batch_sizes may use only the configured ladder {DEFAULT_BATCHES}")
    row_ids,texts=_load_v0(root)
    if len(row_ids)!=len(texts) or len(set(row_ids.tolist()))!=len(row_ids):
        raise ValueError("E19-E V0 IDs and texts must be unique and aligned")
    model_path=_model_path(model_path)
    cache_path=Path(cache_path or OUT/"reconstruction_cache.sqlite3")
    output_path=Path(output_path or OUT/"reconstructed_prompts.parquet")
    log_dir=Path(log_dir or OUT/"server_logs")
    selected=sorted(range(len(row_ids)),key=lambda i:row_ids[i])
    if limit is not None:selected=selected[:max(0,int(limit))]
    tokenizer=_tokenizer()
    contents=[_messages(texts[i],tokenizer) for i in range(len(texts))]
    del tokenizer
    cache=e19b_reconstruct.ReconstructionCache(cache_path)
    model_revision=f"{MODEL_REVISION}:{MODEL_FILE_SHA256}"
    try:
        for index,row_id in enumerate(row_ids):
            key=e19b_reconstruct.cache_key(texts[index],model_revision,PROMPT_VERSION)
            cache.bind_row(row_id,index,key)
        pending=[i for i in selected if not (cache.entry(
            e19b_reconstruct.cache_key(texts[i],model_revision,PROMPT_VERSION)) or {}).get("status")=="ok"]
        if not pending:
            server_history=[]
        else:
            server_history=[]
            batch_pos=0
            server=None
            current_slot=0
            while batch_pos<len(DEFAULT_BATCHES):
                slots=int(DEFAULT_BATCHES[batch_pos])
                if batch_sizes and slots not in batch_sizes:
                    batch_pos+=1;continue
                if server is None:
                    log_path=log_dir/f"llama_server_parallel_{slots}.log"
                    candidate=LlamaServer(model_path=model_path,parallelism=slots,
                        log_path=log_path,max_new_tokens=max_new_tokens)
                    try:
                        info=candidate.start(startup_timeout=startup_timeout)
                        server=candidate;current_slot=slots
                        server_history.append({**info,"log_path":str(log_path),"status":"started"})
                        print(f"E19-E llama-server ready; parallel={slots}, ctx={info['context_size']}",flush=True)
                    except Exception as exc:
                        text=str(exc)
                        if _oom_text(text):
                            server_history.append({"parallelism":slots,"status":"startup_oom",
                                "error":text[-2000:],"log_path":str(log_path)})
                            batch_pos+=1
                            continue
                        raise
                # One wave sends up to `current_slot` independent completions
                # concurrently; llama.cpp batches their active slots internally.
                while batch_pos<len(DEFAULT_BATCHES):
                    active_pending=[]
                    for i in pending:
                        entry=cache.entry(e19b_reconstruct.cache_key(
                            texts[i],model_revision,PROMPT_VERSION)) or {}
                        if entry.get("status")!="ok" and int(entry.get("attempts",0))<int(max_attempts):
                            active_pending.append(i)
                    if not active_pending:break
                    wave=active_pending[:current_slot]
                    errors=[]
                    with concurrent.futures.ThreadPoolExecutor(max_workers=current_slot) as pool:
                        futures={pool.submit(server.request,contents[i],max_tokens=max_new_tokens,seed=seed,
                            timeout=request_timeout):i for i in wave}
                        for future in concurrent.futures.as_completed(futures):
                            index=futures[future]
                            try:
                                raw=future.result()
                                parsed=e19b_reconstruct.parse_reconstruction(raw)
                                key=e19b_reconstruct.cache_key(texts[index],model_revision,PROMPT_VERSION)
                                prior=cache.entry(key);attempt=(int(prior["attempts"]) if prior else 0)+1
                                cache.save_attempt(key,texts[index],MODEL_ID,model_revision,PROMPT_VERSION,
                                    attempt,parsed=parsed,raw_output=raw)
                            except Exception as exc:
                                error=f"{type(exc).__name__}: {exc}"
                                errors.append((index,error))
                                if not _oom_text(error):
                                    key=e19b_reconstruct.cache_key(texts[index],model_revision,PROMPT_VERSION)
                                    prior=cache.entry(key);attempt=(int(prior["attempts"]) if prior else 0)+1
                                    cache.save_attempt(key,texts[index],MODEL_ID,model_revision,
                                        PROMPT_VERSION,attempt,raw_output=None,error=error)
                    for index,error in errors:
                        if _oom_text(error) or server.process.poll() is not None:
                            log=_tail(server.log_path)
                            if _oom_text(error+log):
                                server.stop()
                                server_history.append({"parallelism":current_slot,"status":"runtime_oom",
                                    "error":error,"log_tail":log[-2000:]})
                                server=None;batch_pos+=1
                                break
                    if batch_pos<len(DEFAULT_BATCHES) and server is None:
                        break
                    if errors:
                        # Non-OOM parsing or response errors are retried on the
                        # same server; only CUDA memory errors lower concurrency.
                        exhausted=[]
                        for index,error in errors:
                            key=e19b_reconstruct.cache_key(texts[index],model_revision,PROMPT_VERSION)
                            entry=cache.entry(key)
                            if (entry or {}).get("status")=="failed" and int(entry["attempts"])>=int(max_attempts):
                                exhausted.append(row_ids[index])
                        if exhausted:
                            print(f"E19-E failed {len(exhausted)} rows after retries; continuing",flush=True)
                    n_done=sum(1 for i in selected if (cache.entry(
                        e19b_reconstruct.cache_key(texts[i],model_revision,PROMPT_VERSION)) or {}).get("status")=="ok")
                    print(f"E19-E reconstruction {n_done}/{len(selected)} rows; parallel={current_slot}",flush=True)
                if server is not None and not any(
                        (entry:=cache.entry(e19b_reconstruct.cache_key(
                            texts[i],model_revision,PROMPT_VERSION)) or {}).get("status")!="ok"
                        and int(entry.get("attempts",0))<int(max_attempts) for i in pending):
                    server.stop();server=None
                    break
                if server is None:
                    continue
                break
        frame=cache.export(output_path,row_ids,texts,model_revision,
            model_id=MODEL_ID,prompt_version=PROMPT_VERSION)
    finally:
        if 'server' in locals() and server is not None:server.stop()
        cache.close()
    result={"rows":len(row_ids),"selected_rows":len(selected),"successes":int(frame.status.eq("ok").sum()),
        "failures":int(frame.status.eq("failed").sum()),
        "missing_or_not_yet_selected":int(frame.status.eq("missing").sum()),
        "batch_ladder":list(batch_sizes),"effective_parallelism_history":server_history,
        "max_input_tokens":MAX_INPUT_TOKENS,"max_new_tokens":int(max_new_tokens),
        "temperature":0.0,"top_p":1.0,"seed":int(seed),"thinking_mode":False,
        "model_id":MODEL_ID,"model_revision":model_revision,"prompt_version":PROMPT_VERSION,
        "gguf_path":str(model_path),"llama_server":str(LLAMA_SERVER),
        "cache_path":str(cache_path),"output_path":str(output_path)}
    output_path.parent.mkdir(parents=True,exist_ok=True)
    (output_path.parent/"reconstruction_manifest.json").write_text(json.dumps(result,indent=2))
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=ROOT)
    parser.add_argument("--model-path",type=Path,default=GGUF_PATH)
    parser.add_argument("--batch-sizes",default="10,8,6,4,2")
    parser.add_argument("--limit",type=int)
    parser.add_argument("--max-new-tokens",type=int,default=MAX_NEW_TOKENS)
    parser.add_argument("--max-attempts",type=int,default=RETRY_ATTEMPTS)
    parser.add_argument("--seed",type=int,default=2026)
    parser.add_argument("--startup-timeout",type=int,default=240)
    parser.add_argument("--request-timeout",type=int,default=900)
    parser.add_argument("--cache",type=Path)
    parser.add_argument("--output",type=Path)
    parser.add_argument("--log-dir",type=Path)
    args=parser.parse_args(argv)
    batches=tuple(int(x.strip()) for x in args.batch_sizes.split(",") if x.strip())
    result=reconstruct_v0(args.root,model_path=args.model_path,cache_path=args.cache,
        output_path=args.output,batch_sizes=batches,limit=args.limit,
        max_new_tokens=args.max_new_tokens,max_attempts=args.max_attempts,seed=args.seed,
        startup_timeout=args.startup_timeout,request_timeout=args.request_timeout,log_dir=args.log_dir)
    print(json.dumps(result,indent=2))
    return 0 if result["failures"]==0 else 2


if __name__=="__main__":raise SystemExit(main())
