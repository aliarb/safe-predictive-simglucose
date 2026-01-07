# Validation and Testing Scripts

This folder contains scripts and documentation for validating and testing the NMPC controller and simulator.

## Scripts

### `validate_nmpc_simulator.py`
**Purpose**: Comprehensive validation of NMPC controller with simulator

**Tests**:
1. Patient model response (insulin/meal effects)
2. Worst-case safety checking (realistic predictions)
3. Cost function computation
4. Safety constraints (barrier functions)
5. Controller behavior
6. Numerical stability
7. Consistency and reproducibility
8. Full simulation

**Usage**:
```bash
cd /path/to/project
source venv/bin/activate
python tests/validation/validate_nmpc_simulator.py
```

### `test_nmpc_solver.py`
**Purpose**: Validate NMPC optimization solver

**Tests**:
1. Solver convergence
2. Solution quality
3. Gradient computation accuracy
4. Constraint handling
5. Numerical stability
6. Performance metrics
7. Convergence rate

**Usage**:
```bash
python tests/validation/test_nmpc_solver.py
```

### `test_multi_start_solver.py`
**Purpose**: Compare single-start vs multi-start optimization

**Tests**:
1. Solution quality comparison
2. Convergence rate
3. Local minima escape
4. Performance (time vs quality trade-off)

**Usage**:
```bash
python tests/validation/test_multi_start_solver.py
```

### `debug_nmpc.py`
**Purpose**: Detailed debugging of NMPC controller

**Features**:
- Step-by-step NMPC execution
- Detailed logging of optimization process
- Cost function evaluation
- Prediction analysis

**Usage**:
```bash
python tests/validation/debug_nmpc.py
```

## Documentation

### `HOW_TO_VALIDATE_NMPC.md`
Guide for validating the NMPC simulator, including manual validation steps and common issues.

### `VALIDATION_RESULTS_ANALYSIS.md`
Analysis of validation results, including test outcomes and recommendations.

### `NMPC_SOLVER_VALIDATION.md`
Detailed validation report for the NMPC optimization solver.

### `MULTI_START_SOLVER_ANALYSIS.md`
Analysis of multi-start optimization heuristic, including test results and recommendations.

### `validation_results.txt`
Output from validation runs (for reference).

## Running All Validations

To run all validation tests:

```bash
cd /path/to/project
source venv/bin/activate

# Run simulator validation
python tests/validation/validate_nmpc_simulator.py

# Run solver validation
python tests/validation/test_nmpc_solver.py

# Run multi-start comparison
python tests/validation/test_multi_start_solver.py
```

## Notes

- All scripts assume the project root is in `PYTHONPATH` or run from project root
- Scripts use the virtual environment (`venv`)
- Results are saved to `results/validation/` folder
- Some scripts may take several minutes to complete

## Project Structure

```
tests/
├── validation/          # This folder
│   ├── validate_nmpc_simulator.py
│   ├── test_nmpc_solver.py
│   ├── test_multi_start_solver.py
│   ├── debug_nmpc.py
│   ├── *.md            # Documentation
│   └── README.md        # This file
├── test_*.py            # Unit tests (pytest)
└── sim_results.csv      # Test data
```

