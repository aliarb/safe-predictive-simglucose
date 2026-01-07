# Project Organization

This document describes the organization of the simglucose project, particularly the testing and validation structure.

## Directory Structure

```
safe-predictive-simglucose/
├── simglucose/              # Main package
│   ├── controller/         # Controllers (NMPC, PID, Basal-Bolus)
│   ├── patient/            # Patient models
│   ├── sensor/             # CGM sensors
│   ├── actuator/          # Insulin pumps
│   ├── simulation/        # Simulation engine
│   └── ...
├── examples/               # Example scripts and tutorials
│   ├── compare_nmpc_for_paper.py
│   ├── run_nmpc_controller.py
│   └── ...
├── tests/                  # All testing and validation
│   ├── README.md          # Tests overview
│   ├── validation/        # Validation scripts and docs
│   │   ├── README.md
│   │   ├── validate_nmpc_simulator.py
│   │   ├── test_nmpc_solver.py
│   │   ├── test_multi_start_solver.py
│   │   ├── debug_nmpc.py
│   │   └── *.md           # Validation documentation
│   ├── test_*.py          # Unit tests (pytest)
│   └── sim_results.csv    # Test data
├── results/                # Simulation results
│   ├── paper_comparison/  # Paper comparison results
│   └── validation/        # Validation results
├── Paper/                  # Paper materials (gitignored)
├── docs/                   # Documentation (if exists)
└── README.md              # Main project README
```

## Testing Structure

### Unit Tests (`tests/test_*.py`)
Standard pytest-compatible unit tests for individual components:
- `test_gym.py` - Gym environment tests
- `test_pid_controller.py` - PID controller tests
- `test_sim_engine.py` - Simulation engine tests
- And more...

**Run with:**
```bash
pytest tests/
```

### Validation Scripts (`tests/validation/`)
Comprehensive validation and testing scripts:

#### Scripts
- **`validate_nmpc_simulator.py`** - Full NMPC simulator validation (8 tests)
- **`test_nmpc_solver.py`** - NMPC optimization solver validation
- **`test_multi_start_solver.py`** - Multi-start vs single-start comparison
- **`debug_nmpc.py`** - Detailed NMPC debugging script

#### Documentation
- **`HOW_TO_VALIDATE_NMPC.md`** - Validation guide
- **`VALIDATION_RESULTS_ANALYSIS.md`** - Validation results analysis
- **`NMPC_SOLVER_VALIDATION.md`** - Solver validation report
- **`MULTI_START_SOLVER_ANALYSIS.md`** - Multi-start analysis

**Run from project root:**
```bash
source venv/bin/activate
python tests/validation/validate_nmpc_simulator.py
python tests/validation/test_nmpc_solver.py
python tests/validation/test_multi_start_solver.py
```

## Running Validations

### Quick Validation
```bash
cd /path/to/project
source venv/bin/activate
python tests/validation/validate_nmpc_simulator.py
```

### Full Test Suite
```bash
# Unit tests
pytest tests/

# Validation scripts
python tests/validation/validate_nmpc_simulator.py
python tests/validation/test_nmpc_solver.py
python tests/validation/test_multi_start_solver.py
```

## File Locations

### Moved to `tests/validation/`
- ✅ `validate_nmpc_simulator.py`
- ✅ `test_nmpc_solver.py`
- ✅ `test_multi_start_solver.py`
- ✅ `debug_nmpc.py`
- ✅ `HOW_TO_VALIDATE_NMPC.md`
- ✅ `VALIDATION_RESULTS_ANALYSIS.md`
- ✅ `NMPC_SOLVER_VALIDATION.md`
- ✅ `MULTI_START_SOLVER_ANALYSIS.md`
- ✅ `validation_results.txt`

### Remaining in Root
- `NMPC_COST_FUNCTION.md` - Main documentation (not validation-specific)
- `NMPC_CONVERSION_GUIDE.md` - Main documentation
- `definitions_of_vpatient_parameters.md` - Reference documentation

## Notes

- All validation scripts assume project root is in `PYTHONPATH` or run from project root
- Scripts use relative paths that work from project root
- Results are saved to `results/validation/` folder
- Some scripts may take several minutes to complete

## Benefits of Organization

1. **Clear Separation**: Validation scripts separate from unit tests
2. **Easy Discovery**: All validation scripts in one place
3. **Documentation**: Validation docs co-located with scripts
4. **Maintainability**: Easier to find and update tests
5. **Clean Root**: Root directory less cluttered

