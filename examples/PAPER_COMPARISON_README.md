# NMPC Controller Comparison for Paper Publication

This script compares the NMPC controller with constraints against other baseline controllers (PID, Basal-Bolus) and saves results in multiple formats suitable for academic paper publication.

## Quick Start

```bash
python examples/compare_nmpc_for_paper.py
```

## What It Does

1. **Runs simulations** with 4 controllers:
   - Basal-Bolus Controller (baseline)
   - PID Controller (baseline)
   - NMPC with Constraints (barrier_weight=10.0)
   - NMPC with Strict Constraints (barrier_weight=50.0)

2. **Calculates comprehensive metrics**:
   - Blood glucose statistics (mean, std, min, max, CV)
   - Time in range metrics (70-180 mg/dL, 70-140 mg/dL)
   - Hypoglycemia/hyperglycemia rates
   - Constraint violation rates
   - Risk indices (LBGI, HBGI, Risk Index)
   - Insulin usage statistics
   - Glucose variability (MAGE)

3. **Saves results in multiple formats**:
   - CSV tables (for Excel/analysis)
   - LaTeX tables (for paper inclusion)
   - JSON summary (for programmatic access)
   - Publication-quality figures (PNG 300 DPI and PDF)
   - Individual controller CSV files

## Output Files

All files are saved to `./results/paper_comparison/`:

### Tables
- `controller_comparison_table.csv` - Excel-compatible CSV with all metrics
- `controller_comparison_table.tex` - LaTeX table (auto-generated)
- `controller_comparison_table_formatted.tex` - Formatted LaTeX table with proper styling

### Figures
- `controller_comparison.png` - High-resolution figure (300 DPI) for papers
- `controller_comparison.pdf` - Vector figure for papers

### Data
- `controller_comparison_summary.json` - JSON summary of all metrics
- `[Controller]_detailed.csv` - Individual controller time-series data

## Using Results in Your Paper

### LaTeX Table

The formatted LaTeX table can be directly included in your paper:

```latex
\input{results/paper_comparison/controller_comparison_table_formatted.tex}
```

Or copy the contents into your LaTeX document. Make sure to include the `booktabs` package:

```latex
\usepackage{booktabs}
```

### Figures

The PDF figure can be included directly:

```latex
\begin{figure}[htbp]
    \centering
    \includegraphics[width=\textwidth]{results/paper_comparison/controller_comparison.pdf}
    \caption{Performance comparison of controllers}
    \label{fig:controller_comparison}
\end{figure}
```

### CSV Data

Use the CSV file for:
- Further statistical analysis
- Creating custom visualizations
- Reproducing results
- Sharing data with reviewers

## Metrics Explained

### Time in Range (TIR)
- **70-180 mg/dL**: Standard clinical target range
- **70-140 mg/dL**: Tight control range
- Higher is better (target: >70%)

### Constraint Violation Rate
- Percentage of time outside safe bounds (70-180 mg/dL)
- Lower is better (target: <5%)

### Risk Indices
- **LBGI**: Low Blood Glucose Index (hypoglycemia risk)
- **HBGI**: High Blood Glucose Index (hyperglycemia risk)
- **Risk Index**: Combined risk metric
- Lower is better

### Glucose Variability
- **CV**: Coefficient of variation (std/mean × 100)
- **MAGE**: Mean Absolute Glucose Excursion
- Lower indicates more stable glucose control

## Customization

### Change Constraint Bounds

Edit the script to modify constraint bounds:

```python
constraint_bounds = (80.0, 200.0)  # Custom bounds
bg_min, bg_max = constraint_bounds
```

### Add More Controllers

Add additional controllers in the script:

```python
# Add your custom controller
controller5 = YourCustomController(...)
results5 = sim(sim_obj5)
all_results['Your Controller'] = results5
```

### Modify Metrics

Add custom metrics in the statistics calculation section:

```python
# Add your custom metric
custom_metric = calculate_your_metric(bg_data)
all_stats[name]['Your Metric'] = custom_metric
```

## Example Output

```
================================================================================
PERFORMANCE COMPARISON SUMMARY
================================================================================

Key Metrics:
                        Mean BG (mg/dL)  Time in Range 70-180 (%)  ...
Controller                                                          
Basal-Bolus                       142.87                     89.0  ...
PID                               145.23                     85.5  ...
NMPC (Constrained)                140.12                     92.3  ...
NMPC (Strict Constraints)         138.45                     94.1  ...
```

## Tips for Paper Writing

1. **Use the formatted LaTeX table** - It's already styled for papers
2. **Reference the PDF figure** - Vector graphics scale perfectly
3. **Cite specific metrics** - Use the CSV for exact numbers
4. **Include constraint analysis** - Highlight constraint satisfaction
5. **Compare statistically** - Use the CSV data for statistical tests

## Troubleshooting

### Missing Dependencies
```bash
pip install pandas matplotlib numpy scipy
```

### LaTeX Table Issues
- Ensure `booktabs` package is included
- Check for special characters in controller names
- Verify column alignment

### Figure Quality
- PNG is 300 DPI (suitable for most journals)
- PDF is vector (best quality, preferred by journals)

## Citation

If using this comparison script in your paper, please cite:
- The simglucose simulator
- Your NMPC controller implementation
- Any relevant controller comparison literature


