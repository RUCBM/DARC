"""Check real teacher rollout padding code without loading the CUDA stack."""
from pathlib import Path
from types import SimpleNamespace
import textwrap
import unittest


class Batch:
    def __init__(self, count, n):
        self.count = count
        self.meta_info = {'n': n}

    def __len__(self):
        return self.count


class Output:
    def __init__(self, rows):
        self.batch = {'responses': rows}

    def __len__(self):
        return len(self.batch['responses'])


class TeacherPaddingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = (Path(__file__).resolve().parents[1] / 'verl/trainer/ray_trainer.py').read_text()
        start = source.index('            n = int(label_batch.meta_info.get(')
        end = source.index('            resp_ids =', start)
        cls.code = compile(textwrap.dedent(source[start:end]), 'teacher_padding', 'exec')

    def run_rollout(self, count, n=8, missing_output=False):
        def pad(batch, divisor):
            missing = (-len(batch)) % divisor
            return Batch(len(batch) + missing, n), missing

        def generate(batch):
            rows = [question for question in range(len(batch)) for _ in range(n)]
            return Output(rows[:-1] if missing_output else rows)

        def unpad(output, pad_size):
            rows = output.batch['responses']
            return Output(rows[:-pad_size] if pad_size else rows)

        namespace = {
            'label_batch': Batch(count, n),
            'self': SimpleNamespace(
                config=SimpleNamespace(worker=SimpleNamespace(rollout=SimpleNamespace(n=n))),
                actor_rollout_wg=SimpleNamespace(world_size=8, generate_sequences=generate)),
            'pad_dataproto_to_divisor': pad, 'unpad_dataproto': unpad,
        }
        exec(self.code, namespace)
        return namespace['label_out'].batch['responses']

    def test_five_prompts_keep_exactly_eight_answers_each(self):
        rows = self.run_rollout(5)
        self.assertEqual(rows, [question for question in range(5) for _ in range(8)])

    def test_full_batch_keeps_all_answers(self):
        self.assertEqual(len(self.run_rollout(8)), 64)

    def test_wrong_response_count_and_invalid_sampling_fail(self):
        with self.assertRaises(ValueError):
            self.run_rollout(5, missing_output=True)
        with self.assertRaises(ValueError):
            self.run_rollout(5, n=0)


if __name__ == '__main__':
    unittest.main()
