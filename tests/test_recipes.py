import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class Recipes(unittest.TestCase):
    def test_runtime_profile_matrix(self):
        for recipe in (ROOT / 'recipes').iterdir():
            for profile in ('4xa100', '2xa6000'):
                for runtime in ('docker', 'singularity', 'apptainer'):
                    with self.subTest(recipe=recipe.name, profile=profile, runtime=runtime):
                        env = dict(os.environ, DRY_RUN='1', LLM_HW_PROFILE=profile,
                                   LLM_RUNTIME=runtime, VLLM_MODEL='/weights/model with spaces',
                                   VLLM_IMAGE='vllm/vllm-openai:test', MAX_MODEL_LEN='4096')
                        output = subprocess.check_output([str(recipe / 'serve.sh'), '--max-num-seqs', '2'], env=env, text=True)
                        self.assertIn('--max-model-len 4096', output)
                        self.assertIn('--tensor-parallel-size ' + ('4' if profile == '4xa100' else '2'), output)
                        self.assertIn('--max-num-seqs 2', output)
                        self.assertIn('--served-model-name ' + recipe.name, output)
                        self.assertIn('vllm.entrypoints.openai.api_server', output)

    def test_invalid_profile(self):
        env = dict(os.environ, DRY_RUN='1', LLM_HW_PROFILE='missing')
        run = subprocess.run([str(ROOT / 'recipes/Qwen3.6-27B/serve.sh')], env=env, capture_output=True)
        self.assertEqual(run.returncode, 2)

    def test_summary_preserves_missing_metrics(self):
        spec = importlib.util.spec_from_file_location('summary', ROOT / 'benchmarks/summarize.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = root / 'results/4xa100/Qwen3.6-27B/docker/run'
            run.mkdir(parents=True)
            (run / 'metadata.json').write_text(json.dumps(dict(profile='4xa100', recipe='Qwen3.6-27B', runtime='docker')))
            (run / 'result.json').write_text(json.dumps(dict(completed=10, failed=1, output_throughput=42)))
            rows = module.summarize(root)
            self.assertEqual(rows[0]['output_throughput'], 42)
            self.assertEqual(rows[0]['failed'], 1)
            self.assertEqual(rows[0]['mean_ttft_ms'], '')

    def test_shell_syntax(self):
        for script in [*ROOT.glob('*.sh'), *ROOT.glob('scripts/*.sh'),
                       *ROOT.glob('recipes/*/*.sh'), *ROOT.glob('benchmarks/*.sh')]:
            subprocess.run(['bash', '-n', str(script)], check=True)
