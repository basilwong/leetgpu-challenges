# LeetGPU

This is the challenge set for [LeetGPU.com](https://leetgpu.com). We welcome contributions and bug reports!

## Overview

Each challenge includes problem descriptions, reference implementation, test cases, and starter templates for multiple GPU programming frameworks.

## Local CPU Demo

The repository includes a small browser playground for exploring the challenges in the
currently checked-out branch. It runs a challenge's example case through its PyTorch
reference implementation on your CPU, and can validate a PyTorch `solve(...)`
function against the functional test suite. It does not submit code, contact the
LeetGPU service, or run the large performance tests. Submitted code runs locally, so
only validate code you trust.

```bash
python -m pip install -r scripts/requirements.txt
python scripts/serve_cpu_demo.py
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000). The playground is intentionally
limited to the combined branch's four RL challenges: GAE, PPO, DPO, and GRPO.
No additional Python packages are required for the editor: the browser loads MathJax
and CodeMirror from their CDNs, so an internet connection is required for equation
rendering and the enhanced code editor.

## Challenge Structure

Each challenge contains:

- **`challenge.html`**: Detailed problem description, examples, and constraints
- **`challenge.py`**: Reference implementation, test cases, and challenge metadata
- **`starter/`**: Template files for each supported framework

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on contributing new challenges or improvements.

## License

This problem set is licensed under [CC BY‑NC‑ND 4.0 license](LICENSE).

© 2025 AlphaGPU, LLC. Commercial use, redistribution, or derivative use is prohibited.

## Local Testing

Start the CPU playground from the repository root:

```bash
python -m pip install -r scripts/requirements.txt
python scripts/serve_cpu_demo.py
```

In another terminal, run a smoke test for PPO:

```bash
curl -X POST http://127.0.0.1:8000/api/challenges/easy%2F107_ppo_clipped_surrogate_loss/run-example
```
