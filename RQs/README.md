# RQ Experiment Artifacts

The raw experiment outputs for RQ1-RQ4 are hosted on Zenodo. They are intentionally not versioned in this Git repository, because the raw CSV outputs are large and include filenames that are inconvenient on some platforms.

## Download

The raw experiment artifact record is available here:

- Zenodo record: [Zenodo](https://zenodo.org/records/20582932)

Download the released archives from Zenodo and extract them into this directory.

PowerShell example from the repository root:

```powershell
Expand-Archive .\SecAware-RQ1.zip -DestinationPath .\RQs -Force
Expand-Archive .\SecAware-RQ2.zip -DestinationPath .\RQs -Force
Expand-Archive .\SecAware-RQ3.zip -DestinationPath .\RQs -Force
Expand-Archive .\SecAware-RQ4.zip -DestinationPath .\RQs -Force
```

Unix shell example from the repository root:

```bash
unzip -o SecAware-RQ1.zip -d ./RQs
unzip -o SecAware-RQ2.zip -d ./RQs
unzip -o SecAware-RQ3.zip -d ./RQs
unzip -o SecAware-RQ4.zip -d ./RQs
```

After extraction, the raw CSV files should be available under `RQs/RQ1`, `RQs/RQ2`, `RQs/RQ3`, and `RQs/RQ4`.

## Expected Layout

```text
RQs/
  README.md
  RQ1/
    CyberSecEval/
    LLMSecEval/
    RandomSearch/
    SecAware/
    gptfuzz/
    EvoPrompt/
  RQ2/
    MCTSexplore/
    without-C/
    without-D/
  RQ3/
    from_rq1_secaware/
  RQ4/
    from_rq1_secaware/
```

## Contents

- `RQ1`: raw effectiveness-comparison outputs for SecAware, static benchmarks, and automated baselines.
- `RQ2`: raw ablation-study outputs for the awareness components.
- `RQ3`: record-level data reused from SecAware RQ1 runs for CWE sensitivity analysis.
- `RQ4`: record-level SecAware prompts and responses reused as the basis for repair-guidance analysis.

Each CSV follows the original experiment result format and includes the core `prompt` and `response` columns.
