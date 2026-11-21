"""Resumable Qwen3.5-2B reconstruction of each V0 response's user prompt."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .pipeline import _dataset_rows

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/analysis/e19b"
MODEL_ID = "Qwen/Qwen3.5-2B"
MODEL_REVISION = "15852e8c16360a2fea060d615a32b45270f8a8fc"
RECONSTRUCTION_PROMPT_VERSION = "e19b_prompt_v1"
MAX_NEW_TOKENS = 512
MAX_INPUT_TOKENS = 12288
RETRY_ATTEMPTS = 3

RECONSTRUCTION_INSTRUCTION = """Given an AI assistant's response, infer the most specific original user request that likely produced it.

Identify:
- The central task or question.
- Distinctive subjects, names, entities, numbers and terminology.
- Explicit constraints and requested output format.
- Details that distinguish this request from other requests about the same topic.

Preserve specific evidence from the response. Mark uncertain details rather than inventing them.

Return JSON:
{
  'reconstructed_prompt': '...',
  'topic': '...',
  'distinctive_details': ['...'],
  'constraints': ['...']
}

Output only JSON."""


def response_hash(text: str) -> str:
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def _repair_single_quoted_literal(source):
    """Escape apostrophes inside single-quoted pseudo-JSON string values."""
    output=[]; single=False; double=False; escaped=False; skip=set()
    for index,char in enumerate(source):
        if index in skip: continue
        if escaped:
            output.append(char); escaped=False; continue
        if char=="\\" and (single or double):
            output.append(char); escaped=True; continue
        if char=='"' and not single:
            double=not double; output.append(char); continue
        if char=="," and single:
            look=index+1
            while look<len(source) and source[look].isspace():look+=1
            if look==len(source) or source[look] in ("'",']','}'):
                output.append("'");single=False;output.append(char);continue
        if char=="'" and not double:
            if not single:
                single=True; output.append(char); continue
            look=index+1
            while look<len(source) and source[look].isspace(): look+=1
            if look<len(source) and source[look]=="(":
                # Treat a quote followed by an explanatory parenthetical as a
                # phrase marker, not the end of a list item.
                output.append(" ");continue
            if look==index+1 and look<len(source) and source[look]=='"':
                after=look+1
                while after<len(source) and source[after].isspace(): after+=1
                if after==len(source) or source[after] in ",:]}{":
                    # Some pseudo-JSON outputs reverse the final nested double
                    # quote and outer apostrophe. Restore the intended order.
                    output.extend(['"',"'"]); single=False; skip.add(look); continue
            if look==len(source) or source[look] in ",:]}{\"":
                single=False; output.append(char)
            else:
                output.append("\\'")
            continue
        output.append(char)
    return "".join(output)


def _repair_unescaped_json_quotes(source):
    """Escape quotation marks embedded inside otherwise valid JSON strings."""
    output=[]; in_string=False; escaped=False
    for index,char in enumerate(source):
        if escaped:
            output.append(char);escaped=False;continue
        if in_string:
            if char=="\\":
                output.append(char);escaped=True;continue
            if char=='"':
                look=index+1
                while look<len(source) and source[look].isspace():look+=1
                if look==len(source) or source[look] in ",:]}:":
                    in_string=False;output.append(char)
                else:
                    output.append('\\"')
                continue
            if char in "]},":
                look=index+1
                while look<len(source) and source[look].isspace():look+=1
                if char in "]}" or look==len(source) or source[look] in ('"',"'",']','}',','):
                    output.append('"');in_string=False;output.append(char);continue
            output.append(char);continue
        if char=='"': in_string=True
        output.append(char)
    return "".join(output)


def _parse_partial_json_object(source):
    """Recover completed fields/items from truncated JSON or pseudo-JSON."""
    length=len(source);position=source.find("{")+1;result={}
    def whitespace(pos):
        while pos<length and source[pos].isspace():pos+=1
        return pos
    def string(pos):
        quote=source[pos];pos+=1;value=[]
        while pos<length:
            char=source[pos]
            if char=="\\" and pos+1<length:
                pos+=1;escaped=source[pos]
                value.append({"n":"\n","r":"\r","t":"\t"}.get(escaped,escaped));pos+=1;continue
            if char==quote:
                look=whitespace(pos+1)
                if look==length or source[look] in ",:]}{":
                    return "".join(value),pos+1,True
                value.append(char);pos+=1;continue
            value.append(char);pos+=1
        return "".join(value),pos,False
    def value(pos):
        pos=whitespace(pos)
        if pos>=length:raise ValueError("missing value")
        if source[pos] in ('"',"'"):
            return string(pos)
        if source[pos]=="[":
            items=[];cursor=whitespace(pos+1)
            while cursor<length and source[cursor]!="]":
                try:item,cursor,item_complete=value(cursor)
                except ValueError:break
                if not item_complete:break
                items.append(item);cursor=whitespace(cursor)
                if cursor<length and source[cursor]==",":cursor=whitespace(cursor+1)
                else:break
            complete=cursor<length and source[cursor]=="]"
            return items,cursor+1 if complete else cursor,complete
        end=pos
        while end<length and source[end] not in ",]}\n":end+=1
        return source[pos:end].strip(),end,end<length
    position=whitespace(position)
    while position<length and source[position]!="}":
        try:key,position,key_complete=string(position)
        except Exception:break
        if not key_complete:break
        position=whitespace(position)
        if position>=length or source[position] != ":":break
        position=whitespace(position+1)
        try:parsed_value,position,value_complete=value(position)
        except ValueError:break
        if not value_complete and not isinstance(parsed_value,list):break
        result[key]=parsed_value
        position=whitespace(position)
        if position<length and source[position]==",":position=whitespace(position+1)
        else:break
    required={"reconstructed_prompt","topic","distinctive_details"}
    if not required.issubset(result):
        raise ValueError("truncated reconstruction did not contain its required fields")
    result.setdefault("constraints",[])
    return result


def cache_key(text: str, model_revision: str, prompt_version=RECONSTRUCTION_PROMPT_VERSION) -> str:
    value = "\0".join((response_hash(text), str(model_revision), str(prompt_version)))
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def parse_reconstruction(raw: str) -> dict:
    """Parse JSON-only model output, tolerating a surrounding markdown fence."""
    value = str(raw).strip()
    value = re.sub(r"^```(?:json)?\s*|\s*```$", "", value, flags=re.IGNORECASE)
    start = value.find("{")
    if start < 0:
        raise ValueError("generation contains no JSON object")
    end=value.rfind("}")+1
    if end<=start:
        end=len(value)
    candidate=value[start:end]
    repaired=False
    try:
        json_candidate=_repair_unescaped_json_quotes(candidate)
        repaired=json_candidate!=candidate
        parsed=json.loads(json_candidate)
    except json.JSONDecodeError:
        # The supplied reconstruction prompt illustrates JSON with single
        # quotes; accept that literal form safely, then store normalized JSON.
        try: parsed=ast.literal_eval(_repair_single_quoted_literal(candidate))
        except (ValueError,SyntaxError):
            try:
                parsed=_parse_partial_json_object(_repair_unescaped_json_quotes(candidate))
                repaired=True
            except ValueError as exc:
                raise ValueError("generation contains invalid or unrecoverable JSON") from exc
    if not isinstance(parsed, dict):
        raise ValueError("reconstruction JSON must be an object")
    prompt = parsed.get("reconstructed_prompt")
    topic = parsed.get("topic")
    details = parsed.get("distinctive_details")
    # Older or overlong generations sometimes omit the optional constraints
    # field; the reconstruction remains usable and is normalized to [].
    constraints = parsed.get("constraints", [])
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("reconstructed_prompt must be a nonempty string")
    if not isinstance(topic, str):
        raise ValueError("topic must be a string")
    if not isinstance(details, list) or not all(isinstance(x, str) for x in details):
        raise ValueError("distinctive_details must be a list of strings")
    if isinstance(constraints, str):
        constraints = [constraints]
    if not isinstance(constraints, list) or not all(isinstance(x, str) for x in constraints):
        raise ValueError("constraints must be a list of strings")
    def unique_strings(values):
        return list(dict.fromkeys(x.strip() for x in values if x.strip()))
    return {"reconstructed_prompt": prompt.strip(), "topic": topic.strip(),
            "distinctive_details": unique_strings(details),
            "constraints": unique_strings(constraints),
            "generation_repaired": repaired}


class ReconstructionCache:
    """SQLite cache with transaction-per-result resumability and retry history."""
    def __init__(self, path: Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path, timeout=60)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=FULL")
        self.connection.executescript("""
            CREATE TABLE IF NOT EXISTS cache_entries (
              cache_key TEXT PRIMARY KEY, response_sha256 TEXT NOT NULL,
              model_id TEXT NOT NULL, model_revision TEXT NOT NULL,
              prompt_version TEXT NOT NULL, status TEXT NOT NULL,
              output_json TEXT, raw_output TEXT, attempts INTEGER NOT NULL DEFAULT 0,
              last_error TEXT, updated_at_utc TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS response_rows (
              row_id TEXT PRIMARY KEY, row_index INTEGER NOT NULL,
              cache_key TEXT NOT NULL REFERENCES cache_entries(cache_key)
            );
            CREATE TABLE IF NOT EXISTS attempts (
              id INTEGER PRIMARY KEY AUTOINCREMENT, cache_key TEXT NOT NULL,
              attempt INTEGER NOT NULL, status TEXT NOT NULL, error TEXT,
              raw_output TEXT, created_at_utc TEXT NOT NULL
            );
        """)
        self.connection.commit()

    def close(self):
        self.connection.close()

    def entry(self, key):
        self.connection.row_factory = sqlite3.Row
        row = self.connection.execute("SELECT * FROM cache_entries WHERE cache_key=?", (key,)).fetchone()
        return dict(row) if row else None

    def bind_row(self, row_id, row_index, key):
        self.connection.execute("INSERT OR REPLACE INTO response_rows VALUES (?, ?, ?)",
                                (str(row_id), int(row_index), str(key)))
        self.connection.commit()

    def save_attempt(self, key, text, model_id, model_revision, prompt_version,
                     attempt, *, parsed=None, raw_output=None, error=None):
        status = "ok" if parsed is not None else "failed"
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self.connection.execute("""INSERT INTO cache_entries
            (cache_key,response_sha256,model_id,model_revision,prompt_version,status,
             output_json,raw_output,attempts,last_error,updated_at_utc)
            VALUES (?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(cache_key) DO UPDATE SET status=excluded.status,
              output_json=excluded.output_json,raw_output=excluded.raw_output,
              attempts=excluded.attempts,last_error=excluded.last_error,
              updated_at_utc=excluded.updated_at_utc""",
            (key, response_hash(text), model_id, model_revision, prompt_version,
             status, json.dumps(parsed, ensure_ascii=False) if parsed is not None else None,
             raw_output, int(attempt), error, now))
        self.connection.execute("""INSERT INTO attempts
            (cache_key,attempt,status,error,raw_output,created_at_utc) VALUES (?,?,?,?,?,?)""",
            (key, int(attempt), status, error, raw_output, now))
        self.connection.commit()

    def export(self, output_path: Path, row_ids, texts, model_revision, *,
               model_id=MODEL_ID, prompt_version=RECONSTRUCTION_PROMPT_VERSION):
        records = []
        for index, (row_id, text) in enumerate(zip(row_ids, texts)):
            key = cache_key(text, model_revision, prompt_version)
            entry = self.entry(key)
            payload = json.loads(entry["output_json"]) if entry and entry["output_json"] else {}
            records.append({"id": str(row_id), "row_index": index,
                "response_sha256": response_hash(text), "cache_key": key,
                "model_id": model_id, "model_revision": model_revision,
                "prompt_version": prompt_version,
                "status": entry["status"] if entry else "missing",
                "attempts": int(entry["attempts"]) if entry else 0,
                "last_error": entry["last_error"] if entry else None,
                **payload})
        frame = pd.DataFrame(records)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temp = output_path.with_suffix(output_path.suffix + ".tmp")
        frame.to_parquet(temp, index=False)
        temp.replace(output_path)
        return frame


def _load_qwen(model_id, revision):
    try:
        import torch
        from huggingface_hub import snapshot_download
        from transformers import AutoModelForMultimodalLM, AutoProcessor
    except ImportError as exc:
        raise RuntimeError("Qwen3.5 inference requires current Transformers, PyTorch, and huggingface_hub") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("E19-B prompt reconstruction requires a visible CUDA GPU")
    snapshot = snapshot_download(repo_id=model_id, revision=revision)
    resolved_revision = Path(snapshot).name
    processor = AutoProcessor.from_pretrained(snapshot, local_files_only=True)
    model = AutoModelForMultimodalLM.from_pretrained(
        snapshot, device_map="auto", dtype="auto", local_files_only=True)
    model.eval()
    return torch, processor, model, resolved_revision


def _generation_input(processor, response):
    user_text = (RECONSTRUCTION_INSTRUCTION + "\n\nAssistant response to analyze:\n"
                 "<response>\n" + str(response) + "\n</response>")
    messages = [{"role": "user", "content": [{"type": "text", "text": user_text}]}]
    return processor.apply_chat_template(messages, add_generation_prompt=True,
        tokenize=True, return_dict=True, return_tensors="pt", enable_thinking=False,
        processor_kwargs={"truncation":True,"max_length":MAX_INPUT_TOKENS})


def _generation_inputs(processor, responses):
    conversations=[]
    for response in responses:
        user_text=(RECONSTRUCTION_INSTRUCTION+"\n\nAssistant response to analyze:\n"
                  "<response>\n"+str(response)+"\n</response>")
        conversations.append([{"role":"user","content":[{"type":"text","text":user_text}]}])
    if len(conversations)==1:
        return _generation_input(processor,responses[0])
    processor.tokenizer.padding_side="left"
    return processor.apply_chat_template(conversations,add_generation_prompt=True,
        tokenize=True,return_dict=True,return_tensors="pt",enable_thinking=False,
        processor_kwargs={"padding":True,"truncation":True,"max_length":MAX_INPUT_TOKENS})


def reconstruct_rows(*, row_ids, texts, model_id=MODEL_ID, revision=MODEL_REVISION,
                     cache_path=None, output_path=None, limit=None,
                     max_attempts=RETRY_ATTEMPTS, batch_size=40, max_new_tokens=MAX_NEW_TOKENS,
                     retry_truncated=False, seed=2026,
                     prompt_version=RECONSTRUCTION_PROMPT_VERSION, context=None):
    """Reconstruct selected input rows and export current status for every row."""
    import torch

    if len(row_ids) != len(texts) or len(set(map(str, row_ids))) != len(row_ids):
        raise ValueError("E19-B reconstruction rows need unique, aligned IDs and texts")
    if int(max_attempts) < 1:
        raise ValueError("max_attempts must be positive")
    if int(batch_size)<1:
        raise ValueError("batch_size must be positive")
    if int(max_new_tokens)<1:
        raise ValueError("max_new_tokens must be positive")
    cache_path = Path(cache_path or OUT / "reconstruction_cache.sqlite3")
    output_path = Path(output_path or OUT / "reconstructed_prompts.parquet")
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch_module, processor, model, resolved_revision = _load_qwen(model_id, revision)
    cache = ReconstructionCache(cache_path)
    try:
        ordered = sorted(range(len(row_ids)), key=lambda i: str(row_ids[i]))
        selected = ordered if limit is None else ordered[:max(0, int(limit))]
        for index, row_id in enumerate(row_ids):
            key = cache_key(texts[index], resolved_revision, prompt_version)
            cache.bind_row(row_id, index, key)
        def generate_raw(responses):
            inputs=_generation_inputs(processor,responses)
            device=next(model.parameters()).device
            inputs={name:value.to(device) if hasattr(value,"to") else value for name,value in inputs.items()}
            with torch.inference_mode():
                generated=model.generate(**inputs,do_sample=False,num_beams=1,
                    max_new_tokens=int(max_new_tokens),use_cache=True,
                    pad_token_id=processor.tokenizer.eos_token_id)
            prefix=inputs["input_ids"].shape[1]
            return processor.batch_decode(generated[:,prefix:],skip_special_tokens=True)

        def retry_one(index,retry_budget):
            response=str(texts[index]); row_id=str(row_ids[index])
            key=cache_key(response,resolved_revision,prompt_version)
            prior=cache.entry(key); start_attempt=int(prior["attempts"]) if prior else 0
            last_attempt=start_attempt+int(retry_budget)
            for attempt in range(start_attempt+1,last_attempt+1):
                raw=None
                try:
                    raw=generate_raw([response])[0]
                    parsed=parse_reconstruction(raw)
                    cache.save_attempt(key,response,model_id,resolved_revision,
                        prompt_version,attempt,parsed=parsed,raw_output=raw)
                    return
                except Exception as exc:
                    error=f"{type(exc).__name__}: {exc}"
                    cache.save_attempt(key,response,model_id,resolved_revision,
                        prompt_version,attempt,raw_output=raw,error=error)
                    if torch.cuda.is_available(): torch.cuda.empty_cache()
                    if attempt==last_attempt:
                        print(f"E19-B failed id={row_id} after {attempt} attempts: {error}",flush=True)

        pending=[]
        for index in selected:
            prior=cache.entry(cache_key(texts[index],resolved_revision,prompt_version))
            retry_cutoff=False
            if retry_truncated and prior and prior["status"]=="ok" and prior.get("raw_output"):
                payload=json.loads(prior.get("output_json") or "{}")
                retry_cutoff=bool(payload.get("generation_repaired") and
                                  not str(prior["raw_output"]).rstrip().endswith("}"))
            if not prior or prior["status"]!="ok" or retry_cutoff: pending.append(index)
        completed=0
        def process_batch(indices):
            nonlocal completed
            if not indices:return
            try:
                raws=generate_raw([texts[i] for i in indices])
                if len(raws)!=len(indices):raise RuntimeError("batched generation returned a wrong row count")
                retry=[]
                for index,raw in zip(indices,raws):
                    key=cache_key(texts[index],resolved_revision,prompt_version)
                    prior=cache.entry(key); attempt=(int(prior["attempts"]) if prior else 0)+1
                    try:
                        parsed=parse_reconstruction(raw)
                        cache.save_attempt(key,texts[index],model_id,resolved_revision,
                            prompt_version,attempt,parsed=parsed,raw_output=raw)
                    except Exception as exc:
                        cache.save_attempt(key,texts[index],model_id,resolved_revision,
                            prompt_version,attempt,raw_output=raw,
                            error=f"{type(exc).__name__}: {exc}")
                        retry.append(index)
                completed+=len(indices)
                if completed%10<len(indices) or completed==len(pending):
                    print(f"E19-B reconstruction {completed}/{len(pending)} selected rows",flush=True)
                for index in retry:retry_one(index,max(1,int(max_attempts)-1))
            except Exception:
                if torch.cuda.is_available():torch.cuda.empty_cache()
                if len(indices)>1:
                    middle=len(indices)//2
                    process_batch(indices[:middle]);process_batch(indices[middle:])
                else:
                    completed+=1
                    retry_one(indices[0],max_attempts)
                    if completed%10==0 or completed==len(pending):
                        print(f"E19-B reconstruction {completed}/{len(pending)} selected rows",flush=True)
        batch=int(batch_size)
        for start in range(0,len(pending),batch):process_batch(pending[start:start+batch])
        frame = cache.export(output_path, row_ids, texts, resolved_revision,
                             model_id=model_id,prompt_version=prompt_version)
    finally:
        cache.close()
        del model, processor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    result = {"rows": len(row_ids), "selected_rows": len(selected), "batch_size":int(batch_size),
            "max_input_tokens":MAX_INPUT_TOKENS,"max_new_tokens":int(max_new_tokens),
            "do_sample":False,"num_beams":1,"thinking_mode":False,
            "successes": int(frame.status.eq("ok").sum()),
            "failures": int(frame.status.eq("failed").sum()),
            "missing_or_not_yet_selected": int(frame.status.eq("missing").sum()),
            "model_id": model_id, "model_revision": resolved_revision,
            "prompt_version": prompt_version,
            "cache_path": str(cache_path), "output_path": str(output_path)}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    (output_path.parent / "reconstruction_manifest.json").write_text(json.dumps(result, indent=2))
    return result


def reconstruct_v0(root=ROOT, **kwargs):
    """Use the registered V0 loader; only IDs and response text enter inference."""
    frame = _dataset_rows("v0", Path(root) / "data/processed")
    frame = frame.sort_values("id", key=lambda values: values.astype(str), kind="stable")
    row_ids = frame.id.astype(str).to_numpy()
    texts = frame.text.fillna("").astype(str).to_numpy()
    return reconstruct_rows(row_ids=row_ids, texts=texts, **kwargs)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--revision", default=MODEL_REVISION)
    parser.add_argument("--prompt-version", default=RECONSTRUCTION_PROMPT_VERSION)
    parser.add_argument("--limit", type=int, help="deterministic prefix by sorted response ID")
    parser.add_argument("--max-attempts", type=int, default=RETRY_ATTEMPTS)
    parser.add_argument("--batch-size", type=int, default=40)
    parser.add_argument("--max-new-tokens",type=int,default=MAX_NEW_TOKENS)
    parser.add_argument("--retry-truncated",action="store_true",
        help="retry repaired successful outputs whose raw generation ended before a closing brace")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    result = reconstruct_v0(args.root, model_id=args.model_id, revision=args.revision,
        cache_path=args.cache, output_path=args.output, limit=args.limit,
        max_attempts=args.max_attempts, batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,retry_truncated=args.retry_truncated,seed=args.seed,
        prompt_version=args.prompt_version)
    print(json.dumps(result, indent=2))
    return 0 if result["failures"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
