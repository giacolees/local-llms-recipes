import importlib.util
import json
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SINGLES = [p for p in (ROOT / 'recipes').iterdir() if p.name != 'Qwen3.8-27B-Cluster']


def invoke(script, *args, **env):
    return subprocess.run([str(ROOT / script), *args], env=dict(os.environ, **env),
                          text=True, capture_output=True)


class Recipes(unittest.TestCase):
    def test_runtime_profile_matrix(self):
        for recipe in SINGLES:
            for profile in ('4xa100', '2xa6000', '4xa6000'):
                for runtime in ('docker', 'singularity', 'apptainer'):
                    with self.subTest(recipe=recipe.name, profile=profile, runtime=runtime):
                        result = invoke(recipe / 'serve.sh', '--max-num-seqs', '2',
                                        DRY_RUN='1', LLM_HW_PROFILE=profile, LLM_RUNTIME=runtime,
                                        VLLM_MODEL='/weights/model with spaces',
                                        VLLM_IMAGE='vllm/vllm-openai:test', MAX_MODEL_LEN='4096')
                        self.assertEqual(result.returncode, 0, result.stderr)
                        args = shlex.split(result.stdout)
                        self.assertEqual(args[args.index('--max-model-len') + 1], '4096')
                        self.assertIn('/weights/model with spaces', args)
                        self.assertEqual(args[-2:], ['--max-num-seqs', '2'])
                        self.assertIn('vllm.entrypoints.openai.api_server', args)

    def test_replica_placement_and_defaults(self):
        for runtime in ('docker', 'singularity', 'apptainer'):
            result = invoke('recipes/qwen3.5-2b/run.sh', DRY_RUN='1',
                            LLM_RUNTIME=runtime, LLM_HW_PROFILE='2xa6000')
            self.assertEqual(result.returncode, 0, result.stderr)
            for i in range(4):
                self.assertIn(f'vllm-qwen3.5-2b-{i}', result.stdout)
                self.assertIn(str(8900 + i), result.stdout)
            if runtime == 'docker':
                commands = [shlex.split(line.removeprefix('DRY-RUN:')) for line in result.stdout.splitlines()]
                self.assertEqual([cmd[cmd.index('--gpus')+1] for cmd in commands],
                                 ['"device=0"', '"device=1"', '"device=0"', '"device=1"'])

    def test_tensor_parallel_gpu_selection(self):
        result = invoke('recipes/qwen3.6-35b-a3b/run.sh', DRY_RUN='1',
                        LLM_RUNTIME='docker', LLM_HW_PROFILE='2xa6000')
        self.assertEqual(result.returncode, 0, result.stderr)
        args = shlex.split(result.stdout.removeprefix('DRY-RUN:'))
        self.assertEqual(args[args.index('--gpus')+1], '"device=0,1"')
        self.assertEqual(args[args.index('--tensor-parallel-size')+1], '2')

    def test_overrides_and_budget_validation(self):
        result = invoke('recipes/qwen3.5-2b/run.sh', '20', DRY_RUN='1', LLM_HW_PROFILE='2xa6000')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('exceeds', result.stderr)
        result = invoke('recipes/qwen3.5-2b/serve.sh', DRY_RUN='1', LLM_HW_PROFILE='2xa6000',
                        MODEL='custom/checkpoint', TP='2', REPLICAS='1', GPU_MEM_UTIL='0.6')
        self.assertEqual(result.returncode, 0, result.stderr)
        args = shlex.split(result.stdout)
        self.assertEqual(args[args.index('--model')+1], 'custom/checkpoint')
        self.assertEqual(args[args.index('--gpu-memory-utilization')+1], '0.6')

    def test_alias_and_discovery(self):
        command = 'source scripts/vllm-docker/lib.sh; recipe_keys'
        keys = subprocess.check_output(['bash', '-c', command], cwd=ROOT, text=True).splitlines()
        self.assertEqual(len(keys), 10)
        self.assertIn('Qwen3.6-27B', keys)
        result = invoke('vllm-docker/run-model.sh', 'qwen3.6-27b', '1', DRY_RUN='1', LLM_HW_PROFILE='2xa6000')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('Qwen/Qwen3.6-27B', result.stdout)

    def test_cluster_matrix(self):
        for runtime in ('docker', 'singularity', 'apptainer'):
            for script in ('cluster_up.sh', 'serve.sh', 'status.sh', 'stop.sh'):
                with self.subTest(runtime=runtime, script=script):
                    result = invoke(f'recipes/Qwen3.8-27B-Cluster/{script}', DRY_RUN='1',
                                    LLM_RUNTIME=runtime, LLM_HW_PROFILE='4xa6000')
                    self.assertEqual(result.returncode, 0, result.stderr)
                    if script == 'serve.sh':
                        self.assertIn('pipeline-parallel-size', result.stdout)
                        self.assertIn('qwen3.8-27b', result.stdout)
                    elif script == 'cluster_up.sh':
                        self.assertIn('10.0.0.2', result.stdout)
                        self.assertIn('10.0.0.3', result.stdout)
                        self.assertIn('--num-gpus', result.stdout)

    def test_benchmark_endpoint_resolution(self):
        for recipe, profile, model, port in [
                ('qwen3.5-2b', '2xa6000', 'Qwen/Qwen3.5-2B', '8900'),
                ('Qwen3.6-27B', '4xa100', 'Qwen/Qwen3.6-27B', '8600'),
                ('Qwen3.8-27B-Cluster', '4xa6000', 'qwen3.8-27b', '8000')]:
            result = invoke('benchmarks/run.sh', recipe, DRY_RUN='1', LLM_HW_PROFILE=profile,
                            MODEL_REVISION='test-revision', VLLM_IMAGE='test/image:tag')
            self.assertEqual(result.returncode, 0, result.stderr)
            args = shlex.split(result.stdout)
            self.assertEqual(args[args.index('--model')+1], model)
            self.assertTrue(args[args.index('--base-url')+1].endswith(':'+port))

    def test_runtime_execution_without_gpus(self):
        # Exercise the real lifecycle branch with instrumented runtime binaries.
        # No daemon, instance, network, or GPU is used.
        with tempfile.TemporaryDirectory() as tmp:
            temp = Path(tmp)
            stub = temp / 'docker'
            stub.write_text('''#!/usr/bin/env python3
import json, os, sys
with open(os.environ['RUNTIME_TRACE'], 'a') as stream:
    stream.write(json.dumps({'argv':sys.argv, 'gpu':os.environ.get('APPTAINERENV_CUDA_VISIBLE_DEVICES')}) + "\\n")
if sys.argv[1:2] == ['ps']: print('fake-container')
''')
            stub.chmod(0o755)
            for name in ('singularity', 'apptainer'):
                (temp / name).symlink_to(stub)
            sif = temp / 'test.sif'
            sif.touch()
            trace = temp / 'trace.jsonl'
            for runtime in ('docker', 'singularity', 'apptainer'):
                with self.subTest(runtime=runtime):
                    trace.write_text('')
                    env = dict(PATH=str(temp)+os.pathsep+os.environ['PATH'],
                               RUNTIME_TRACE=str(trace), LLM_RUNTIME=runtime,
                               LLM_HW_PROFILE='2xa6000', VLLM_SIF=str(sif),
                               NO_WAIT='1', HF_HOME=str(temp / 'hf'), LOG_DIR=str(temp / 'logs'),
                               DRY_RUN='0')
                    start = invoke('recipes/qwen3.5-2b/run.sh', '2', **env)
                    self.assertEqual(start.returncode, 0, start.stderr)
                    records = [json.loads(line) for line in trace.read_text().splitlines()]
                    starts = [r for r in records if r['argv'][1:2] == ['run'] or r['argv'][1:3] == ['instance', 'start']]
                    self.assertEqual(len(starts), 2)
                    if runtime != 'docker':
                        self.assertEqual([r['gpu'] for r in starts], ['0', '1'])
                    stop = invoke('recipes/qwen3.5-2b/stop.sh', **env)
                    self.assertEqual(stop.returncode, 0, stop.stderr)

    def test_invalid_profile(self):
        result = invoke('recipes/Qwen3.6-27B/serve.sh', DRY_RUN='1', LLM_HW_PROFILE='missing')
        self.assertEqual(result.returncode, 2)

    def test_summary_preserves_missing_metrics(self):
        spec = importlib.util.spec_from_file_location('summary', ROOT / 'benchmarks/summarize.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = root / 'results/4xa6000/Qwen3.8-27B-Cluster/singularity/run'
            run.mkdir(parents=True)
            (run / 'metadata.json').write_text(json.dumps(dict(profile='4xa6000', recipe='Qwen3.8-27B-Cluster', runtime='singularity')))
            (run / 'result.json').write_text(json.dumps(dict(completed=10, failed=1, output_throughput=42)))
            rows = module.summarize(root)
            self.assertEqual(rows[0]['output_throughput'], 42)
            self.assertEqual(rows[0]['failed'], 1)
            self.assertEqual(rows[0]['mean_ttft_ms'], '')

    def test_shell_syntax(self):
        for script in ROOT.rglob('*.sh'):
            if '.git' not in script.parts and '.agent' not in script.parts:
                subprocess.run(['bash', '-n', str(script)], check=True)
