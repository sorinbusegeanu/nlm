"""Small local model smoke tests; skipped when optional neural packages are absent."""
import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np

AVAILABLE = all(importlib.util.find_spec(name) is not None for name in ("torch", "transformers", "peft", "accelerate", "tokenizers"))


@unittest.skipUnless(AVAILABLE, "install the [neural] extra to run local neural adapter tests")
class NeuralAdapterTests(unittest.TestCase):
    def _tokenizer(self, directory):
        from tokenizers import Tokenizer, models, pre_tokenizers
        from transformers import PreTrainedTokenizerFast
        vocab={"[PAD]":0,"[UNK]":1,"[CLS]":2,"[SEP]":3,"[MASK]":4,"alpha":5,"beta":6,"gamma":7,"common":8}
        backend=Tokenizer(models.WordLevel(vocab=vocab,unk_token="[UNK]"));backend.pre_tokenizer=pre_tokenizers.Whitespace()
        tokenizer=PreTrainedTokenizerFast(tokenizer_object=backend,unk_token="[UNK]",pad_token="[PAD]",cls_token="[CLS]",sep_token="[SEP]",mask_token="[MASK]")
        tokenizer.save_pretrained(directory)
        return tokenizer

    def _rows(self):
        texts=[];labels=[]
        for label,word in zip(("a","b","c"),("alpha","beta","gamma")):
            for i in range(3): texts.append(f"{word} common");labels.append(label)
        return texts,labels

    def test_encoder_adapter_small_local_checkpoint(self):
        import torch
        from transformers import BertConfig, BertForSequenceClassification
        from src.model_factory import train_encoder_fold
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)/"base";root.mkdir();self._tokenizer(root)
            config=BertConfig(vocab_size=9,hidden_size=16,num_hidden_layers=1,num_attention_heads=2,intermediate_size=32,num_labels=3)
            BertForSequenceClassification(config).save_pretrained(root)
            texts,labels=self._rows()
            result=train_encoder_fold(texts,labels,["alpha common"],["a","b","c"],checkpoint=str(root),revision=None,
                output_dir=Path(td)/"out",epochs=1,batch_size=3,max_length=8,learning_rate=1e-3)
            self.assertEqual(result["scores"].shape,(1,3));self.assertTrue(np.isfinite(result["probabilities"]).all())
            reloaded=BertForSequenceClassification.from_pretrained(Path(td)/"out")
            batch=self._tokenizer(root)(["alpha common"],return_tensors="pt",padding=True)
            batch={k:v for k,v in batch.items() if k in {"input_ids","attention_mask"}}
            batch={k:v.to(next(reloaded.parameters()).device) for k,v in batch.items()}
            with torch.no_grad(): self.assertTrue(torch.isfinite(reloaded(**batch).logits).all())

    def test_qwen_lora_readout_gradients_and_saved_model(self):
        import torch
        from transformers import LlamaConfig, LlamaModel
        from src.model_factory import train_qwen_lora_fold
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)/"base";root.mkdir();self._tokenizer(root)
            config=LlamaConfig(vocab_size=9,hidden_size=16,intermediate_size=32,num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=2)
            LlamaModel(config).save_pretrained(root)
            texts,labels=self._rows()
            result=train_qwen_lora_fold(texts,labels,["alpha common"],["a","b","c"],checkpoint=str(root),revision=None,
                output_dir=Path(td)/"out",epochs=1,batch_size=3,max_length=8,learning_rate=1e-3,lora_rank=2,lora_alpha=4)
            self.assertEqual(result["scores"].shape,(1,3));self.assertTrue(np.isfinite(result["probabilities"]).all())
            batch=self._tokenizer(root)(["alpha common"],return_tensors="pt",padding=True)
            batch={k:v for k,v in batch.items() if k in {"input_ids","attention_mask"}}
            device=next(result["model"].parameters()).device;batch={k:v.to(device) for k,v in batch.items()}
            output=result["model"](**batch,labels=torch.tensor([0]));output.loss.backward()
            self.assertTrue(any(p.grad is not None for n,p in result["model"].named_parameters() if "lora_" in n))
            self.assertTrue(any(p.grad is not None for p in result["model"].classifier.parameters()))
            merged_dir=Path(td)/"out"/"merged_backbone"
            self.assertTrue((merged_dir/"model.safetensors").exists())
            loaded=LlamaModel.from_pretrained(merged_dir).to(device).eval()
            classifier=torch.nn.Linear(16,3).to(device)
            classifier.load_state_dict(torch.load(Path(td)/"out"/"classifier.pt",map_location=device,weights_only=True))
            with torch.inference_mode():
                hidden=loaded(input_ids=batch["input_ids"],attention_mask=batch["attention_mask"]).last_hidden_state
                mask=batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                reloaded_logits=classifier(((hidden*mask).sum(1)/mask.sum(1).clamp_min(1)).float())
                self.assertTrue(torch.allclose(output.logits.detach(),reloaded_logits,atol=2e-4,rtol=2e-4))


if __name__ == "__main__": unittest.main()
