import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from src import model_factory


def _write_trainer_checkpoint(output_dir, step, *, epoch, log_history, optimizer_state, scheduler_state):
    checkpoint = Path(output_dir) / f"checkpoint-{step}"
    checkpoint.mkdir(parents=True)
    (checkpoint / "model.safetensors").write_bytes(b"weights")
    (checkpoint / "optimizer.pt").write_text(optimizer_state)
    (checkpoint / "scheduler.pt").write_text(scheduler_state)
    (checkpoint / "rng_state.pth").write_bytes(b"rng")
    (checkpoint / "trainer_state.json").write_text(json.dumps({
        "global_step": step, "epoch": epoch, "log_history": log_history,
    }))
    return checkpoint


class TrainingOomRecoveryTests(unittest.TestCase):
    def test_resumed_early_stopping_replays_validation_history(self):
        class CallbackBase:
            pass

        callback = model_factory._accuracy_early_stopping_callback(
            CallbackBase, patience=2, metric="balanced_accuracy")
        state = SimpleNamespace(log_history=[
            {"epoch": 1.0, "eval_balanced_accuracy": 0.45, "eval_accuracy": 0.5},
            {"epoch": 2.0, "eval_balanced_accuracy": 0.44, "eval_accuracy": 0.49},
        ])
        control = SimpleNamespace(should_training_stop=False)
        callback.on_train_begin(None, state, control)
        self.assertEqual(callback.best_epoch, 1)
        self.assertEqual(callback.bad_epochs, 1)
        callback.on_evaluate(None, SimpleNamespace(epoch=3.0), control,
                             metrics={"eval_balanced_accuracy": 0.43, "eval_accuracy": 0.48})
        self.assertEqual(callback.stopped_epoch, 3)
        self.assertTrue(control.should_training_stop)

    def test_latest_recovery_checkpoint_accepts_adapter_and_classifier_state(self):
        with tempfile.TemporaryDirectory() as temp:
            checkpoint = Path(temp) / "checkpoint-19"
            checkpoint.mkdir()
            for filename in ("adapter_model.safetensors", "classifier.pt", "optimizer.pt",
                             "scheduler.pt", "rng_state.pth", "trainer_state.json"):
                (checkpoint / filename).write_bytes(b"saved")
            self.assertEqual(model_factory._latest_complete_trainer_checkpoint(temp), checkpoint)

    def test_interrupted_run_resumes_configured_latest_trainer_checkpoint(self):
        calls = []
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "prior" / "checkpoint-1873"
            source.parent.mkdir()
            _write_trainer_checkpoint(source.parent, 1873, epoch=0.29,
                log_history=[{"epoch": 0.29, "loss": 1.2}],
                optimizer_state="optimizer", scheduler_state="scheduler")
            source = source.parent / "checkpoint-1873"

            def fake_training(*, train_text, epochs, output_dir, batch_size,
                              gradient_accumulation_steps, trainer_resume_checkpoint,
                              _oom_recovery_context=None, _trainer_resume_checkpoint=None):
                if _trainer_resume_checkpoint is None:
                    _trainer_resume_checkpoint = trainer_resume_checkpoint
                calls.append(_trainer_resume_checkpoint)
                if len(calls) == 1:
                    _oom_recovery_context["phase"] = "training"
                    raise RuntimeError("CUDA out of memory before next periodic checkpoint")
                Path(output_dir).mkdir(parents=True, exist_ok=True)
                (Path(output_dir) / "model.safetensors").write_bytes(b"final")
                return {"selected_checkpoint_path": str(output_dir)}

            recovery = model_factory._with_cuda_oom_recovery(fake_training)
            result = recovery(train_text=list(range(80)), epochs=20,
                output_dir=root / "resumed", batch_size=8,
                gradient_accumulation_steps=2, trainer_resume_checkpoint=str(source))
            self.assertEqual(calls, [str(source), str(source)])
            event = result["oom_recovery"]["events"][0]
            self.assertEqual(event["resume_mode"],
                "resume_configured_model_optimizer_scheduler_rng_and_trainer_state")
            self.assertEqual(event["recovery_global_step"], 1873)

    def test_oom_without_complete_checkpoint_starts_cleanly_at_reduced_batch(self):
        calls = []

        def fake_training(*, train_text, epochs, output_dir, batch_size, gradient_accumulation_steps,
                          _oom_recovery_context=None, _trainer_resume_checkpoint=None):
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            calls.append((_trainer_resume_checkpoint, batch_size, gradient_accumulation_steps))
            if len(calls) == 1:
                _oom_recovery_context["phase"] = "training"
                raise RuntimeError("CUDA out of memory before first optimizer checkpoint")
            (output_dir / "model.safetensors").write_bytes(b"final")
            return {"effective_batch_size": batch_size}

        recovery = model_factory._with_cuda_oom_recovery(fake_training)
        with tempfile.TemporaryDirectory() as temp:
            output_dir = Path(temp) / "checkpoint"
            result = recovery(train_text=list(range(80)), epochs=2,
                              output_dir=output_dir, batch_size=8,
                              gradient_accumulation_steps=1)
            self.assertEqual(calls, [(None, 8, 1), (None, 4, 2)])
            self.assertEqual(result["oom_recovery"]["events"][0]["resume_mode"],
                             "clean_restart_no_complete_trainer_checkpoint")
            self.assertEqual(result["oom_recovery"]["events"][0]["action"],
                             "reduce_batch_and_restart_cleanly_no_recoverable_state")

    def _run_recovery_case(self, *, epoch_boundary):
        calls = []
        log_history = ([{"epoch": 1.0, "eval_accuracy": 0.5,
                         "eval_balanced_accuracy": 0.45, "eval_loss": 1.4}]
                       if epoch_boundary else [{"epoch": 0.4, "loss": 1.8, "learning_rate": 2e-5}])

        def fake_training(*, train_text, epochs, output_dir, batch_size, gradient_accumulation_steps,
                          _oom_recovery_context=None, _trainer_resume_checkpoint=None):
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            if _trainer_resume_checkpoint is None:
                calls.append(("initial", batch_size, gradient_accumulation_steps))
                state_dir = output_dir / "training_state"
                state_dir.mkdir()
                (state_dir / "best_state.json").write_text('{"selected_epoch":1}')
                (state_dir / "validation_epoch_0001.npy").write_bytes(b"saved-validation-output")
                (output_dir / "epoch-01").mkdir()
                _write_trainer_checkpoint(output_dir, 7 if epoch_boundary else 3,
                    epoch=1.0 if epoch_boundary else 0.4, log_history=log_history,
                    optimizer_state="optimizer-at-step", scheduler_state="scheduler-at-step")
                _oom_recovery_context["phase"] = "training"
                raise RuntimeError("CUDA out of memory during optimizer step")

            resume_path = Path(_trainer_resume_checkpoint)
            state = json.loads((resume_path / "trainer_state.json").read_text())
            model_factory._copy_training_recovery_state(resume_path, output_dir)
            calls.append(("resume", batch_size, gradient_accumulation_steps,
                          state["global_step"], state["epoch"],
                          (resume_path / "optimizer.pt").read_text(),
                          (resume_path / "scheduler.pt").read_text(),
                          (output_dir / "training_state" / "best_state.json").exists(),
                          (output_dir / "training_state" / "validation_epoch_0001.npy").exists(),
                          (output_dir / "epoch-01").is_dir()))
            (output_dir / "model.safetensors").write_bytes(b"final")
            return {"effective_batch_size": batch_size,
                    "best_checkpoint_path": str(output_dir),
                    "selected_checkpoint_path": str(output_dir),
                    "epoch_checkpoints": [str(output_dir / "epoch-01")]}

        recovery = model_factory._with_cuda_oom_recovery(fake_training)
        with tempfile.TemporaryDirectory() as temp:
            output_dir = Path(temp) / "checkpoint"
            result = recovery(train_text=list(range(80)), epochs=2,
                              output_dir=output_dir, batch_size=8,
                              gradient_accumulation_steps=1)
            self.assertEqual(calls[0], ("initial", 8, 1))
            resumed = calls[1]
            self.assertEqual(resumed[:3], ("resume", 4, 2))
            self.assertEqual(resumed[3], 7 if epoch_boundary else 3)
            self.assertEqual(resumed[4], 1.0 if epoch_boundary else 0.4)
            self.assertEqual(resumed[5:7], ("optimizer-at-step", "scheduler-at-step"))
            self.assertEqual(resumed[7:], (True, True, True))
            self.assertTrue((output_dir / "model.safetensors").is_file())
            self.assertEqual(result["oom_recovery"]["events"][0]["resume_mode"],
                "resume_model_optimizer_scheduler_rng_and_trainer_state")
            self.assertEqual(result["oom_recovery"]["events"][0]["recovery_global_step"],
                             7 if epoch_boundary else 3)
            schedule = result["oom_recovery"]["events"][0]["recovered_schedule"]
            self.assertEqual(schedule["steps_per_epoch"], 10)
            self.assertEqual(schedule["planned_optimizer_steps"], 20)
            self.assertEqual(schedule["remaining_optimizer_steps"], 13 if epoch_boundary else 17)
            self.assertEqual(result["gradient_accumulation_steps"], 2)

    def test_oom_after_completed_optimizer_steps_resumes_latest_state(self):
        self._run_recovery_case(epoch_boundary=False)

    def test_oom_after_completed_epoch_preserves_validation_selection_history(self):
        self._run_recovery_case(epoch_boundary=True)


if __name__ == "__main__":
    unittest.main()
