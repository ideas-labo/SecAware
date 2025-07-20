# SecAware

This repository contains the data and code for the following paper:

> Awareness Reinforced Security Testing for LLM-based Code Generation

## Introduction

> Large Language Models (LLMs) have become essential parts of software development due to their ability to generate code from natural instructions written in a task prompt. While highly efficient, the code they generate can, unfortunately, contain severe security vulnerabilities. Without understanding the true capability of LLMs in generating secure code, it leaves developers with great doubt in choosing the appropriate LLM and/or designing a good quality task prompt. Yet, existing approaches to evaluate a LLM to that end mostly rely on static benchmarks, which are limited to only certain tasks and the vulnerability types of the code generated.

> In this paper, we propose SecAware, an automated testing tool that perturbs the test cases (task prompts), specifically designed for revealing the capability of secure code generation by LLMs. SecAware is special such that it relies on three levels of awareness with reinforcement: context, weakness, and diversity. The weakness awareness is reinforced, supported by the other two awareness levels, to strike the task prompts and vulnerability types that are the most likely to reveal issues in the target LLM. Experiment on six popular LLMs and five other approaches reveals that SecAware achieves significantly better results in terms of testing effectiveness and efficiency with up to 5.63× improvements on testing success rate. It also reveals previously unknown patterns in the capability of the LLMs in secure code generation.

## Code Structure

- datasets => This directory includes extracted knowledge used in our method, and initial seed pool to start with.<br>
- fuzzer => Main components of method.<br>
- utils => Utils for our method, including oracle, prompt templates.<br>
- requirements.txt => Essential requirments need to be installed <br>

## <a name='quick-start'></a> Quick Start

- Python 3.8+

To run the code, and install the essential requirements:

```
pip install -r requirements.txt
```

And you can run the below code to have a quick start:

```
cd ./code
python3 run.py
```

## <a name='Sotas'></a>Compared Methods

Below are the repositories of the SOTA bechmarks and automated tools evaluated and compared with SecAware.

### Benchmarks

- [LLMSecEval](https://github.com/tuhh-softsec/LLMSecEval):
- [CyberSecEval](https://github.com/meta-llama/PurpleLlama/tree/main/CybersecurityBenchmarks):

### Automated Tools

- Random Search: a simple baseline that merely selects an operator to mutate the task prompts in a random manner.
- [LLM-fuzzer](https://github.com/sherdencooper/GPTFuzz):
- [EvoPrompt](https://github.com/beeevita/EvoPrompt):

## RQ Reproduction

- **RQ1 Effectiveness**: To measure the effectiveness of our method, you can directly run [Quick Start](#quick-start). The other SOTA methods being compared are described in [Compared Methods](#Sotas).

- **RQ2 Ablation**: Compare the differences when the key component rules are uesd or not used:

- **RQ3 Prompt Pattern Study**: Analyze the sensitivity of the key parameter $l$, and set it to 5, 10, 15 or 20:

- **RQ4 Explainability Case Study**:

## RQs

Experiment results of our paper.
