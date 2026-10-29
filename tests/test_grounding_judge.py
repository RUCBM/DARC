"""CPU-only checks of constrained requests and the actual reward dispatch path."""

import ast
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
from typing import Dict, List, Optional
import unittest
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]
MODULE_NAME = "examples.reward_function.grounding_judge"
spec = importlib.util.spec_from_file_location(
    MODULE_NAME, ROOT / "examples/reward_function/grounding_judge.py"
)
judge = importlib.util.module_from_spec(spec)
spec.loader.exec_module(judge)


def completion(text, finish_reason="stop"):
    return SimpleNamespace(choices=[SimpleNamespace(text=text, finish_reason=finish_reason)])


class GroundingJudgeTests(unittest.TestCase):
    def setUp(self):
        self.client = Mock()
        self.context = Mock()
        self.context.__enter__ = Mock(return_value=self.client)
        self.context.__exit__ = Mock(return_value=False)
        self.constructor = Mock(return_value=self.context)
        self.create = self.client.completions.create
        for patcher in [
            patch.dict(os.environ, {}, clear=True),
            patch.dict(sys.modules, {"openai": SimpleNamespace(OpenAI=self.constructor), MODULE_NAME: judge}),
            patch.object(judge.time, "sleep"),
        ]:
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_requests_use_base_model_completions_with_full_choice_constraint(self):
        for value, expected in [("yes", True), ("no", False)]:
            with self.subTest(value=value):
                self.create.return_value = completion(value)
                self.assertIs(judge.judge_related("REFERENCE", "QUESTION"), expected)
                kwargs = self.create.call_args.kwargs
                self.assertEqual(kwargs["model"], "Qwen3-4B-Base")
                self.assertEqual(kwargs["extra_body"], {"guided_choice": ["yes", "no"]})
                self.assertEqual(kwargs["temperature"], 0)
                self.assertEqual(kwargs["n"], 1)
                self.assertGreater(kwargs["max_tokens"], 1)
                self.assertIn("`QUESTION` is related to the following text:REFERENCE", kwargs["prompt"])
                self.client.chat.completions.create.assert_not_called()

    def test_rejects_nonliteral_and_truncated_verdicts(self):
        for value in ["", None, "Yes", " yes", "yes\n", "no.", "yes because", "yes/no", "<think>yes"]:
            with self.subTest(value=value):
                self.create.return_value = completion(value)
                with self.assertRaises(judge.GroundingJudgeError):
                    judge.judge_related("document", "question")
        self.create.return_value = completion("yes", "length")
        with self.assertRaises(judge.GroundingJudgeError):
            judge.judge_related("document", "question")

    def test_retries_keep_the_decoding_constraint(self):
        self.create.side_effect = [TimeoutError(), completion("invalid"), completion("no")]
        self.assertFalse(judge.judge_related("document", "question"))
        self.assertEqual(self.create.call_count, 3)
        for call in self.create.call_args_list:
            self.assertEqual(call.kwargs["extra_body"], {"guided_choice": ["yes", "no"]})

    def test_errors_never_become_a_verdict_or_expose_response_body(self):
        self.create.side_effect = RuntimeError("private response body")
        with self.assertRaises(judge.GroundingJudgeError) as error:
            judge.judge_related("document", "question")
        self.assertEqual(self.create.call_count, 3)
        self.assertNotIn("private response body", str(error.exception))
        self.assertIn("no grounding verdict", str(error.exception))

    def test_client_config_is_separate_from_deepseek(self):
        self.create.return_value = completion("yes")
        with patch.dict(os.environ, {
            "API_KEY": "baseline-test-only", "API_BASE_URL": "https://baseline.invalid",
            "GROUNDING_JUDGE_BASE_URL": "http://127.0.0.1:6200/v1",
            "GROUNDING_JUDGE_API_KEY": "local-test-only", "GROUNDING_JUDGE_MODEL": "judge-alias",
        }):
            judge.judge_related("document", "question")
        self.assertEqual(self.constructor.call_args.kwargs["base_url"], "http://127.0.0.1:6200/v1")
        self.assertEqual(self.constructor.call_args.kwargs["api_key"], "local-test-only")
        self.assertEqual(self.create.call_args.kwargs["model"], "judge-alias")

    def test_invalid_retry_count_fails_before_request(self):
        with patch.dict(os.environ, {"GROUNDING_JUDGE_RETRIES": "0"}):
            with self.assertRaises(ValueError):
                judge.judge_related("document", "question")
        self.create.assert_not_called()

    def load_reward_dispatch(self):
        # Load the real dispatch function without GPU/training imports or storage side effects.
        tree = ast.parse((ROOT / "examples/reward_function/difficulty_aware_questioner.py").read_text())
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "generate_relation_results")
        self.deepseek = Mock(return_value=True)
        namespace = dict(os=os, List=List, Dict=Dict, Optional=Optional, _deepseek_judge_related=self.deepseek)
        exec(compile(ast.Module(body=[node], type_ignores=[]), "reward_dispatch", "exec"), namespace)
        return namespace["generate_relation_results"]

    def test_reward_dispatch_preserves_order_and_maps_negative_verdict(self):
        dispatch = self.load_reward_dispatch()
        def create(**kwargs):
            return completion("no" if "`unrelated`" in kwargs["prompt"] else "yes")
        self.create.side_effect = create
        payload = [{"text": "doc", "question": q} for q in ["related", "unrelated", ""]]
        with patch.dict(os.environ, {"GROUNDING_JUDGE_BACKEND": "qwen3_base"}):
            results = dispatch(payload)
        self.assertEqual([r["question"] for r in results], ["related", "unrelated", ""])
        self.assertEqual([r["related"] for r in results], [True, False, False])
        self.assertEqual([r["score"] for r in results], [1.0, 0.0, 0.0])
        self.deepseek.assert_not_called()

    def test_new_reward_path_propagates_failures_instead_of_legacy_fail_open(self):
        dispatch = self.load_reward_dispatch()
        self.create.side_effect = TimeoutError()
        with patch.dict(os.environ, {"GROUNDING_JUDGE_BACKEND": "qwen3_base"}):
            with self.assertRaisesRegex(RuntimeError, "Qwen3 grounding failed"):
                dispatch([{"text": "doc", "question": "q"}])
        self.deepseek.assert_not_called()

    def test_default_keeps_baseline_and_unknown_backend_is_rejected(self):
        dispatch = self.load_reward_dispatch()
        self.deepseek.return_value = None
        payload = [{"text": "doc", "question": "q"}]
        self.assertTrue(dispatch(payload)[0]["related"])
        self.deepseek.assert_called_once_with("doc", "q")
        self.create.assert_not_called()
        with patch.dict(os.environ, {"GROUNDING_JUDGE_BACKEND": "typo"}):
            with self.assertRaises(ValueError):
                dispatch(payload)


class LaunchScriptTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "scripts").mkdir()
        (self.root / "bin").mkdir()
        self.python = self.root / "bin/python3"
        self.python.write_text('#!/bin/bash\nprintf "PYTHON_ARG=%s\\n" "$@"\nexit "${PREFLIGHT_EXIT:-0}"\n')
        self.python.chmod(0o755)
        self.env = {"PATH": str(self.root / "bin") + os.pathsep + os.defpath}

    def test_wrapper_preflight_and_argument_forwarding(self):
        wrapper = self.root / "scripts/difficulty_aware_questioner_train_qwen3_grounding.sh"
        shutil.copyfile(ROOT / "scripts" / wrapper.name, wrapper)
        (self.root / "scripts/difficulty_aware_questioner_train.sh").write_text(
            '#!/bin/bash\nprintf "BACKEND=%s\\nMODEL=%s\\n" "$GROUNDING_JUDGE_BACKEND" "$GROUNDING_JUDGE_MODEL"\n'
            'printf "TRAIN_ARG=%s\\n" "$@"\n'
        )
        args = ["bash", str(wrapper), "/solver model", "/questioner model", "extra experiment", "/train data.parquet"]
        result = subprocess.run(args, env=self.env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("PYTHON_ARG=examples.reward_function.grounding_judge", result.stdout)
        self.assertIn("BACKEND=qwen3_base", result.stdout)
        self.assertIn("MODEL=Qwen3-4B-Base", result.stdout)
        for arg in args[2:]:
            self.assertIn("TRAIN_ARG=" + arg + "\n", result.stdout)
        result = subprocess.run(args, env=dict(self.env, PREFLIGHT_EXIT="7"), capture_output=True, text=True)
        self.assertEqual(result.returncode, 7)
        self.assertNotIn("TRAIN_ARG=", result.stdout)

    def test_judge_server_requires_gpu_and_uses_local_constrained_backend(self):
        args = ["bash", str(ROOT / "vllm_service_init/start_grounding_judge.sh"), "/Qwen3-4B-Base model"]
        result = subprocess.run(args, env=self.env, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn("PYTHON_ARG=", result.stdout)
        result = subprocess.run(args, env=dict(self.env, CUDA_VISIBLE_DEVICES="0"), capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        for option, value in [
            ("--host", "127.0.0.1"), ("--guided-decoding-backend", "xgrammar"),
            ("--model", "/Qwen3-4B-Base model"), ("--served-model-name", "Qwen3-4B-Base"),
            ("--generation-config", "vllm"),
        ]:
            self.assertIn(f"PYTHON_ARG={option}\nPYTHON_ARG={value}\n", result.stdout)


if __name__ == "__main__":
    unittest.main()
