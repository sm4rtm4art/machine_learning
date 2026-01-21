# Project Template

This is a template for new ML projects in the portfolio. Copy this directory to create a new project.

## Usage

```bash
# Copy template
cp -r projects/_template projects/my_new_project

# Update README and configs
# Implement project-specific code in project/
```

## Structure

```
projects/<name>/
├── README.md           # This file - update with project details
├── configs/
│   └── default.yaml    # Default configuration
├── notebooks/
│   └── 00_report.ipynb # Results visualization
├── scripts/
│   ├── download_data.py
│   ├── train.py
│   ├── evaluate.py
│   ├── export.py
│   └── serve.py
├── project/
│   ├── __init__.py
│   ├── data.py         # Dataset handling
│   ├── model.py        # Model definition
│   ├── train.py        # Training logic
│   └── eval.py         # Evaluation logic
└── tests/
    └── test_smoke.py   # Basic tests
```

## Checklist for New Projects

- [ ] Update README with project description
- [ ] Define config schema in `configs/default.yaml`
- [ ] Implement dataset loading in `project/data.py`
- [ ] Implement model in `project/model.py`
- [ ] Implement training in `project/train.py`
- [ ] Implement evaluation in `project/eval.py`
- [ ] Write smoke tests
- [ ] Add project to main README.md project table
