"""Dependency-free regression checks; all judge calls use mocked clients."""

import ast
import importlib.util
import json
import os
from pathlib import Path
import runpy
import sys
import tempfile
from types import SimpleNamespace
from typing import Any, Dict
import unittest
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("recheck_common", ROOT / "evaluation/recheck_common.py")
common = importlib.util.module_from_spec(spec)
spec.loader.exec_module(common)


def completion(content):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


class RecheckTests(unittest.TestCase):
    def setUp(self):
        self.client = Mock()
        self.create = self.client.chat.completions.create
        self.sleep = patch.object(common.time, "sleep")
        self.sleep.start()
        self.addCleanup(self.sleep.stop)

    def test_verdicts_are_strict(self):
        for content, expected in [("Yes", True), (" YES. ", True), ("No", False), ("no!", False)]:
            with self.subTest(content=content):
                self.assertIs(common.parse_verdict(content), expected)
        for content in [None, "", "yesterday", "Yes or No", "Yes, but I am unsure."]:
            with self.subTest(content=content), self.assertRaises(common.JudgeError):
                common.parse_verdict(content)

    def test_sdk_base_url_and_key_configuration(self):
        constructor = Mock(return_value=self.client)
        with patch.dict(sys.modules, {"openai": SimpleNamespace(OpenAI=constructor)}):
            clients = common.create_clients(
                "https://api.openai.com/v1, http://localhost:6000/v1/chat/completions/", "test-key"
            )
        self.assertEqual(len(clients), 2)
        calls = constructor.call_args_list
        self.assertEqual(calls[0].kwargs["base_url"], "https://api.openai.com/v1")
        self.assertEqual(calls[1].kwargs["base_url"], "http://localhost:6000/v1")
        for call in calls:
            self.assertEqual(call.kwargs["api_key"], "test-key")
            self.assertEqual(call.kwargs["max_retries"], 0)
            self.assertEqual(call.kwargs["timeout"], 20)

    def test_configuration_validation(self):
        for urls, keys in [("", "key"), ("https://example.test/v1", ""), ("https://example.test/v1", "one,two")]:
            with self.subTest(urls=urls, keys=keys), self.assertRaises(ValueError):
                common.create_clients(urls, keys)

    def test_prompt_roles_and_negative_verdict(self):
        self.create.return_value = completion("No")
        self.assertFalse(common.judge_answer([self.client], "42", "My final answer is 17."))
        prompt = self.create.call_args.kwargs["messages"][1]["content"]
        self.assertIn("Ground-truth answer:\n42", prompt)
        self.assertIn("Model response to evaluate:\nMy final answer is 17.", prompt)
        self.assertEqual(self.create.call_count, 1)

    def test_retry_recovers_from_request_error_and_invalid_output(self):
        self.create.side_effect = [TimeoutError("temporary"), completion("uncertain"), completion("Yes")]
        self.assertTrue(common.judge_answer([self.client], "42", "42"))
        self.assertEqual(self.create.call_count, 3)

    def test_exhausted_retries_raise_without_leaking_error_payload(self):
        self.create.side_effect = RuntimeError("private gateway error body")
        with self.assertRaises(common.JudgeError) as raised:
            common.judge_answer([self.client], "42", "42", max_attempts=2)
        self.assertEqual(self.create.call_count, 2)
        self.assertIn("no verdict was recorded", str(raised.exception))
        self.assertNotIn("private gateway error body", str(raised.exception))

    def run_entrypoint(self, script, verdict=None, failure=None):
        self.create.reset_mock(side_effect=True, return_value=True)
        if failure is not None:
            self.create.side_effect = failure
        else:
            self.create.return_value = completion(verdict)
        with tempfile.TemporaryDirectory() as temp:
            temp = Path(temp)
            data = temp / "evaluation" / "test-model"
            data.mkdir(parents=True)
            for dataset in ["math", "gsm8k", "amc", "minerva", "olympiad", "aime2024", "aime2025"]:
                (data / f"results_{dataset}.json").write_text(json.dumps([
                    {"answer": "42", "response": "17", "score": 0}, {"average_score": 0}
                ]))
            argv = [script, "--model_name", "test-model", "--api_keys", "test-key"]
            if script.startswith("difficulty"):
                argv += ["--result_path", str(data)]
                output = data / "final_results.json"
            else:
                output = temp / "final_results.jsonl"
            modules = {
                "openai": SimpleNamespace(OpenAI=Mock(return_value=self.client)),
                "tqdm": SimpleNamespace(tqdm=lambda iterable, **kwargs: iterable),
                "recheck_common": common,
            }
            previous_cwd = Path.cwd()
            try:
                os.chdir(temp)
                with patch.dict(sys.modules, modules), patch.object(sys, "argv", argv), patch.dict(os.environ, {"STORAGE_PATH": str(temp)}), patch("builtins.print"):
                    if failure is not None or verdict == "uncertain":
                        with self.assertRaises(common.JudgeError) as raised:
                            runpy.run_path(str(ROOT / "evaluation" / script), run_name="__main__")
                        self.assertIn("math", str(raised.exception))
                        self.assertIn("sample 0", str(raised.exception))
                        self.assertFalse(output.exists(), "Failed judging must not produce a normal dataset score")
                    else:
                        runpy.run_path(str(ROOT / "evaluation" / script), run_name="__main__")
                        records = [json.loads(line) for line in output.read_text().splitlines()]
                        self.assertEqual(len(records), 7)
                        self.assertTrue(all(row["score"] == (100 if verdict == "Yes" else 0) for row in records))
            finally:
                os.chdir(previous_cwd)

    def test_both_entrypoints_keep_yes_no_and_failure_distinct(self):
        for script in ["results_recheck.py", "difficulty_aware_results_recheck.py"]:
            for verdict in ["Yes", "No", "uncertain"]:
                with self.subTest(script=script, verdict=verdict):
                    self.run_entrypoint(script, verdict=verdict)
            with self.subTest(script=script, failure=True):
                self.run_entrypoint(script, failure=TimeoutError("temporary"))


class LengthFilterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Execute the actual filter without importing the CUDA training stack.
        tree = ast.parse((ROOT / "verl/utils/dataset.py").read_text())
        dataset_class = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "RLHFDataset")
        method = next(node for node in dataset_class.body if isinstance(node, ast.FunctionDef) and node.name == "_filter_overlong_prompts")
        namespace = {"Dict": Dict, "Any": Any}
        exec(compile(ast.Module(body=[method], type_ignores=[]), "dataset.py", "exec"), namespace)
        cls.filter_method = staticmethod(namespace["_filter_overlong_prompts"])

    def subject(self, prompt_tokens, teacher_tokens=None, chat_template=None):
        counts = {"prompt": prompt_tokens, "teacher": teacher_tokens}
        tokenizer = Mock()
        tokenizer.chat_template = chat_template
        tokenizer.encode.side_effect = lambda text, **kwargs: [1] * counts["teacher" if text.startswith("teacher") else "prompt"]
        tokenizer.apply_chat_template.side_effect = lambda messages, **kwargs: [1] * counts[messages]
        return SimpleNamespace(
            tokenizer=tokenizer, processor=None, max_prompt_length=8192,
            label_prompt_key="text_prompt", _build_messages=lambda example: "prompt",
            _chat_to_prompt_fallback=lambda messages: messages + "x" * 9000,
            _normalize_prompt_data=lambda value: value,
        )

    def test_fallback_filters_by_tokens_including_teacher_and_boundary(self):
        for prompt, teacher, expected in [(3000, None, True), (8192, None, True), (8193, None, False), (3000, 8192, True), (3000, 8193, False)]:
            with self.subTest(prompt=prompt, teacher=teacher):
                obj = self.subject(prompt, teacher)
                example = {} if teacher is None else {"text_prompt": "teacher"}
                self.assertEqual(self.filter_method(obj, example), expected)
                for call in obj.tokenizer.encode.call_args_list:
                    self.assertIs(call.kwargs["add_special_tokens"], False)

    def test_chat_template_path_remains_token_based(self):
        for prompt, teacher, expected in [(8192, 8192, True), (8193, 100, False), (100, 8193, False)]:
            with self.subTest(prompt=prompt, teacher=teacher):
                obj = self.subject(prompt, teacher, chat_template="template")
                self.assertEqual(self.filter_method(obj, {"text_prompt": "teacher"}), expected)
                obj.tokenizer.encode.assert_not_called()


if __name__ == "__main__":
    unittest.main()
