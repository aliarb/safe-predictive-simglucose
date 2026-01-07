# Quick Start Guide for Validation Scripts

## Running Validations

All validation scripts should be run from the **project root** directory.

### Setup
```bash
cd /path/to/safe-predictive-simglucose
source venv/bin/activate
```

### 1. Full Simulator Validation
```bash
python tests/validation/validate_nmpc_simulator.py
```
**What it tests:**
- Patient model response
- Worst-case safety checking
- Cost function computation
- Safety constraints
- Controller behavior
- Numerical stability
- Consistency
- Full simulation

**Time:** ~4-7 minutes

### 2. Solver Validation
```bash
python tests/validation/test_nmpc_solver.py
```
**What it tests:**
- Solver convergence
- Solution quality
- Gradient computation
- Constraint handling
- Numerical stability
- Performance metrics

**Time:** ~1-2 minutes

### 3. Multi-Start Comparison
```bash
python tests/validation/test_multi_start_solver.py
```
**What it tests:**
- Single-start vs multi-start performance
- Solution quality comparison
- Time overhead analysis

**Time:** ~1-2 minutes

### 4. Debug Script
```bash
python tests/validation/debug_nmpc.py
```
**What it does:**
- Step-by-step NMPC execution
- Detailed optimization logging
- Cost function evaluation

**Time:** ~30 seconds

## Results

Results are saved to:
- `results/validation/` - Simulation results
- Console output - Test summaries

## Troubleshooting

### Import Errors
If you get import errors, make sure you're in the project root:
```bash
cd /path/to/safe-predictive-simglucose
export PYTHONPATH=$PWD:$PYTHONPATH
python tests/validation/validate_nmpc_simulator.py
```

### Path Issues
All scripts use relative paths from project root. If paths fail:
1. Check you're in project root
2. Verify `results/` directory exists
3. Check file permissions

### Virtual Environment
Always activate the virtual environment:
```bash
source venv/bin/activate
```

## See Also

- `README.md` - Detailed documentation
- `HOW_TO_VALIDATE_NMPC.md` - Comprehensive validation guide
- `VALIDATION_RESULTS_ANALYSIS.md` - Analysis of results

