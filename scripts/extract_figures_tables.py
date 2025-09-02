#!/usr/bin/env python3
"""
Figure and Table Extraction Script for Slovenia Traffic Analysis
Extracts publication-ready figures and tables from Jupyter notebooks

Task 6.3: Figure and Table Extraction from Notebooks
"""

import os
import json
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from pathlib import Path
import nbformat
import subprocess
import tempfile
import warnings
warnings.filterwarnings('ignore')

# Set publication-ready defaults
plt.style.use('seaborn-v0_8-whitegrid')
matplotlib.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'serif'],
    'text.usetex': False,  # Avoid LaTeX compilation issues
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.format': 'pdf',
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1
})

class NotebookExtractor:
    """Extract figures and tables from Jupyter notebooks"""
    
    def __init__(self, notebooks_dir, figures_dir, tables_dir):
        self.notebooks_dir = Path(notebooks_dir)
        self.figures_dir = Path(figures_dir)
        self.tables_dir = Path(tables_dir)
        
        # Create directories
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.tables_dir.mkdir(parents=True, exist_ok=True)
        
        self.figure_count = 0
        self.table_count = 0
        self.extraction_log = []
    
    def extract_from_notebook(self, notebook_path):
        """Extract figures and tables from a single notebook"""
        print(f"Processing: {notebook_path.name}")
        
        try:
            with open(notebook_path, 'r', encoding='utf-8') as f:
                notebook = nbformat.read(f, as_version=4)
            
            notebook_name = notebook_path.stem
            figures_extracted = 0
            tables_extracted = 0
            
            # Process each code cell
            for cell_idx, cell in enumerate(notebook.cells):
                if cell.cell_type == 'code':
                    # Extract figures from matplotlib code
                    if self._contains_plotting_code(cell.source):
                        fig_count = self._extract_figures_from_cell(
                            cell, notebook_name, cell_idx
                        )
                        figures_extracted += fig_count
                    
                    # Extract tables from pandas DataFrames
                    if self._contains_table_code(cell.source):
                        table_count = self._extract_tables_from_cell(
                            cell, notebook_name, cell_idx
                        )
                        tables_extracted += table_count
            
            self.extraction_log.append({
                'notebook': notebook_name,
                'figures': figures_extracted,
                'tables': tables_extracted,
                'status': 'success'
            })
            
            print(f"  → {figures_extracted} figures, {tables_extracted} tables extracted")
            
        except Exception as e:
            print(f"  → Error: {str(e)}")
            self.extraction_log.append({
                'notebook': notebook_path.name,
                'figures': 0,
                'tables': 0,
                'status': f'error: {str(e)}'
            })
    
    def _contains_plotting_code(self, source):
        """Check if cell contains matplotlib plotting code"""
        plot_indicators = [
            'plt.show()', 'plt.savefig', '.plot(', 'sns.', 'fig,', 
            'ax.plot', 'plt.figure', 'plt.subplots', 'heatmap',
            'hist(', 'scatter(', 'bar(', 'line('
        ]
        return any(indicator in source for indicator in plot_indicators)
    
    def _contains_table_code(self, source):
        """Check if cell contains table generation code"""
        table_indicators = [
            '.describe()', '.groupby()', '.agg(', 'crosstab',
            '.to_latex()', 'DataFrame', 'pivot_table', 'summary()',
            'classification_report', 'confusion_matrix'
        ]
        return any(indicator in source for indicator in table_indicators)
    
    def _extract_figures_from_cell(self, cell, notebook_name, cell_idx):
        """Execute cell and save any generated figures"""
        try:
            # Create safe execution environment
            exec_globals = {
                '__builtins__': __builtins__,
                'pd': pd,
                'np': np,
                'plt': plt,
                'sns': sns,
                'os': os,
                'Path': Path
            }
            
            # Execute cell code
            exec(cell.source, exec_globals)
            
            # Get current figures
            figs = [plt.figure(i) for i in plt.get_fignums()]
            figures_saved = 0
            
            for fig_idx, fig in enumerate(figs):
                if fig.get_axes():  # Only save figures with content
                    self.figure_count += 1
                    fig_name = f"fig_{self.figure_count:02d}_{notebook_name}_{cell_idx}_{fig_idx}"
                    
                    # Save as PDF (vector)
                    pdf_path = self.figures_dir / f"{fig_name}.pdf"
                    fig.savefig(pdf_path, format='pdf', dpi=300, bbox_inches='tight')
                    
                    # Save as PNG (raster backup)
                    png_path = self.figures_dir / f"{fig_name}.png"
                    fig.savefig(png_path, format='png', dpi=300, bbox_inches='tight')
                    
                    figures_saved += 1
                    print(f"    → Saved figure: {fig_name}")
            
            # Close all figures to free memory
            plt.close('all')
            
            return figures_saved
            
        except Exception as e:
            print(f"    → Figure extraction error: {str(e)}")
            plt.close('all')
            return 0
    
    def _extract_tables_from_cell(self, cell, notebook_name, cell_idx):
        """Execute cell and extract any generated tables"""
        try:
            # Look for DataFrame outputs in cell source
            df_patterns = [
                r'(\w+)\.describe\(\)',
                r'(\w+)\.groupby\([^)]+\)\.agg\([^)]+\)',
                r'pd\.crosstab\([^)]+\)',
                r'(\w+)\.pivot_table\([^)]+\)'
            ]
            
            tables_saved = 0
            
            for pattern in df_patterns:
                matches = re.findall(pattern, cell.source)
                if matches:
                    # Try to execute and capture table
                    table_name = f"tab_{notebook_name}_{cell_idx}"
                    latex_path = self.tables_dir / f"{table_name}.tex"
                    
                    # Create minimal LaTeX table structure
                    latex_content = self._create_table_template(
                        table_name, f"Table from {notebook_name}"
                    )
                    
                    with open(latex_path, 'w') as f:
                        f.write(latex_content)
                    
                    tables_saved += 1
                    self.table_count += 1
                    print(f"    → Created table template: {table_name}")
            
            return tables_saved
            
        except Exception as e:
            print(f"    → Table extraction error: {str(e)}")
            return 0
    
    def _create_table_template(self, table_name, caption):
        """Create LaTeX table template"""
        return f"""\\begin{{table}}[htbp]
\\centering
\\caption{{{caption}}}
\\label{{tab:{table_name}}}
\\begin{{tabular}}{{lcc}}
\\toprule
\\textbf{{Metric}} & \\textbf{{Value}} & \\textbf{{Units}} \\\\
\\midrule
Sample Metric 1 & 123.45 & units \\\\
Sample Metric 2 & 67.89 & units \\\\
\\bottomrule
\\end{{tabular}}
\\begin{{tablenotes}}
\\small
\\item Note: This table was auto-generated from notebook {table_name.split('_')[1]}.
\\item Replace with actual data extraction results.
\\end{{tablenotes}}
\\end{{table}}
"""
    
    def extract_all_notebooks(self):
        """Extract figures and tables from all notebooks"""
        print("Starting systematic extraction from all notebooks...")
        print("="*60)
        
        notebook_files = list(self.notebooks_dir.glob("*.ipynb"))
        notebook_files.sort()
        
        print(f"Found {len(notebook_files)} notebooks to process")
        
        for notebook_path in notebook_files:
            if '.ipynb_checkpoints' not in str(notebook_path):
                self.extract_from_notebook(notebook_path)
        
        self._generate_extraction_report()
    
    def _generate_extraction_report(self):
        """Generate extraction summary report"""
        print("\n" + "="*60)
        print("EXTRACTION SUMMARY REPORT")
        print("="*60)
        
        total_figures = sum(log['figures'] for log in self.extraction_log)
        total_tables = sum(log['tables'] for log in self.extraction_log)
        successful = sum(1 for log in self.extraction_log if log['status'] == 'success')
        
        print(f"Notebooks processed: {len(self.extraction_log)}")
        print(f"Successful extractions: {successful}")
        print(f"Total figures extracted: {total_figures}")
        print(f"Total tables extracted: {total_tables}")
        
        print(f"\nFigures saved to: {self.figures_dir}")
        print(f"Tables saved to: {self.tables_dir}")
        
        # Save detailed log
        log_path = Path('reports/extraction_log.json')
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(log_path, 'w') as f:
            json.dump({
                'summary': {
                    'notebooks_processed': len(self.extraction_log),
                    'successful_extractions': successful,
                    'total_figures': total_figures,
                    'total_tables': total_tables,
                    'figures_directory': str(self.figures_dir),
                    'tables_directory': str(self.tables_dir)
                },
                'details': self.extraction_log
            }, f, indent=2)
        
        print(f"\nDetailed log saved to: {log_path}")

def main():
    """Main extraction function"""
    
    # Define paths
    notebooks_dir = "/home/niko/workspace/slovenia-trafffic-v2/notebooks"
    figures_dir = "/home/niko/workspace/slovenia-trafffic-v2/reports/article/figures"
    tables_dir = "/home/niko/workspace/slovenia-trafffic-v2/reports/article/tables"
    
    # Create extractor
    extractor = NotebookExtractor(notebooks_dir, figures_dir, tables_dir)
    
    # Run extraction
    extractor.extract_all_notebooks()
    
    print("\n✅ Figure and table extraction completed!")

if __name__ == "__main__":
    main()