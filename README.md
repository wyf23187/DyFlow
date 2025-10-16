# DyFlow: Dynamic Workflow Framework for Agentic Reasoning

[![NeurIPS 2025](https://img.shields.io/badge/NeurIPS-2025-blue.svg)](https://nips.cc/Conferences/2025)
[![arXiv](https://img.shields.io/badge/arXiv-2509.26062-b31b1b.svg)](https://arxiv.org/abs/2509.26062)
[![Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-DyPlanner-yellow)](https://huggingface.co/wyf23187/DyPlanner)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**TL;DR:** DyFlow introduces a two-level Designer–Executor architecture with dynamic operators that adaptively re-plan subgoals during execution based on intermediate feedback. This enables more generalizable and robust reasoning across diverse domains and tasks.

## Highlights

- **Execution-adaptive workflows**: Dynamically adjust reasoning processes and subgoals according to intermediate feedback
- **Two-core components**:
  - **Designer** — performs high-level task decomposition and planning
  - **Executor** — carries out low-level execution and tool invocation
- **Cross-domain evaluation**: Demonstrated effectiveness across multiple domains

## Installation

```bash
git clone https://github.com/wyf23187/DyFlow.git
cd DyFlow
pip install -r requirements.txt
```

### Environment Setup

Create a `.env` file with your API keys:

```bash
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
DEEPINFRA_API_KEY=your_deepinfra_key
```

## Deploying DyPlanner with vLLM

DyPlanner uses a locally deployed model via vLLM. First, deploy the DyPlanner model:

```bash
# Download and deploy the DyPlanner model from Hugging Face
# Model: https://huggingface.co/wyf23187/DyPlanner

vllm serve wyf23187/DyPlanner \
    --port 8000 \
```

The `ModelService.local()` will automatically connect to this vLLM endpoint at `http://localhost:8000` to get responses from DyPlanner.

## Quick Start

For basic usage and benchmark evaluation examples, please refer to:
- `scripts/run_workflow.py` - Single problem workflow execution
- `scripts/run_dataset.py` - Batch benchmark evaluation

Available benchmarks: HumanEval, MATH, LiveBench, SocialMaze, PubMedQA

## Training Data Generation

For generating training data from DyFlow execution traces, see `train/`.


## Citation

If you find our work useful, please cite:

```bibtex
@inproceedings{wang2025dyflow,
  title={DyFlow: Dynamic Workflow Framework for Agentic Reasoning},
  author={Wang, Yanbo and Xu, Zixiang and Huang, Yue and Wang, Xiangqi and Song, Zirui and Gao, Lang and Wang, Chenxi and Tang, Xiangru and Zhao, Yue and Cohan, Arman and others},
  booktitle={Advances in Neural Information Processing Systems},
  year={2025}
}
```