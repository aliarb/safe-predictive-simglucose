# Tests Directory

This directory contains all testing and validation scripts for the simglucose project.

## Structure

```
tests/
├── README.md                    # This file
├── validation/                 # Validation and testing scripts
│   ├── README.md              # Validation scripts documentation
│   ├── validate_nmpc_simulator.py
│   ├── test_nmpc_solver.py
│   ├── test_multi_start_solver.py
│   ├── debug_nmpc.py
│   └── *.md                   # Validation documentation
├── test_*.py                   # Unit tests (pytest compatible)
└── sim_results.csv             # Test data
```

## Test Categories

### Unit Tests (`test_*.py`)
Standard unit tests using pytest framework:
- `test_gym.py` - Gym environment tests
- `test_gymnasium.py` - Gymnasium environment tests
- `test_pid_controller.py` - PID controller tests
- `test_sim_engine.py` - Simulation engine tests
- `test_ui.py` - User interface tests
- And more...

**Run unit tests:**
```bash
pytest tests/
```

### Validation Scripts (`validation/`)
Comprehensive validation and testing scripts:
- `validate_nmpc_simulator.py` - Full simulator validation
- `test_nmpc_solver.py` - Solver validation
- `test_multi_start_solver.py` - Multi-start comparison
- `debug_nmpc.py` - Debugging script

**Run validation:**
```bash
cd /path/to/project
source venv/bin/activate
python tests/validation/validate_nmpc_simulator.py
```

See `validation/README.md` for detailed documentation.

## Running Tests

### All Unit Tests
```bash
pytest tests/
```

### Specific Test File
```bash
pytest tests/test_pid_controller.py
```

### Validation Scripts
```bash
# From project root
source venv/bin/activate
python tests/validation/validate_nmpc_simulator.py
python tests/validation/test_nmpc_solver.py
python tests/validation/test_multi_start_solver.py
```

## Notes

- All scripts assume the project root is in `PYTHONPATH` or run from project root
- Scripts use the virtual environment (`venv`)
- Results are saved to `results/` folder
- Some validation scripts may take several minutes to complete

