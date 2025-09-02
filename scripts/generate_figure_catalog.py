#!/usr/bin/env python3
"""
Generate Comprehensive Figure and Table Catalog
Creates complete inventory of all extracted figures and tables

Task 6.3: Catalog Generation
"""

import json
import pandas as pd
from pathlib import Path
from datetime import datetime

class FigureTableCatalog:
    """Generate comprehensive catalog of figures and tables"""
    
    def __init__(self):
        self.figures_dir = Path('/home/niko/workspace/slovenia-trafffic-v2/reports/article/figures')
        self.tables_dir = Path('/home/niko/workspace/slovenia-trafffic-v2/reports/article/tables')
        self.reports_dir = Path('/home/niko/workspace/slovenia-trafffic-v2/reports')
        
    def catalog_figures(self):
        """Create comprehensive figure catalog"""
        print("Cataloging figures...")
        
        # Get all PDF figures
        pdf_files = sorted(self.figures_dir.glob("*.pdf"))
        
        figure_catalog = []
        for pdf_file in pdf_files:
            # Extract figure info from filename
            stem = pdf_file.stem
            parts = stem.split('_', 2)
            
            if len(parts) >= 3:
                fig_num = parts[1]
                fig_name = '_'.join(parts[2:])
            else:
                fig_num = "unknown"
                fig_name = stem
            
            # Categorize figure
            category = self._categorize_figure(fig_name)
            
            figure_info = {
                'figure_number': fig_num,
                'filename': pdf_file.name,
                'name': fig_name,
                'category': category,
                'size_kb': pdf_file.stat().st_size // 1024,
                'path_relative': f"../figures/{pdf_file.name}"
            }
            
            figure_catalog.append(figure_info)
        
        print(f"  → Found {len(figure_catalog)} figures")
        return figure_catalog
    
    def catalog_tables(self):
        """Create comprehensive table catalog"""
        print("Cataloging tables...")
        
        # Get all TEX tables
        tex_files = sorted(self.tables_dir.glob("*.tex"))
        
        table_catalog = []
        for tex_file in tex_files:
            stem = tex_file.stem
            
            # Extract table info
            if stem.startswith('tab_'):
                table_name = stem[4:]  # Remove 'tab_' prefix
            else:
                table_name = stem
            
            # Categorize table
            category = self._categorize_table(table_name)
            
            table_info = {
                'table_name': table_name,
                'filename': tex_file.name,
                'category': category,
                'size_kb': tex_file.stat().st_size // 1024,
                'path_relative': f"../tables/{tex_file.name}"
            }
            
            table_catalog.append(table_info)
        
        print(f"  → Found {len(table_catalog)} tables")
        return table_catalog
    
    def _categorize_figure(self, fig_name):
        """Categorize figure by content"""
        name_lower = fig_name.lower()
        
        if any(term in name_lower for term in ['fundamental', 'speed', 'density', 'relationship']):
            return 'Traffic Analysis'
        elif any(term in name_lower for term in ['economic', 'cost', 'benefit', 'waterfall']):
            return 'Economic Analysis'
        elif any(term in name_lower for term in ['roc', 'accident', 'risk', 'threshold']):
            return 'Safety Analysis'
        elif any(term in name_lower for term in ['capacity', 'utilization', 'projection']):
            return 'Capacity Analysis'
        elif any(term in name_lower for term in ['temporal', 'daily', 'weekly', 'trend']):
            return 'Temporal Analysis'
        elif any(term in name_lower for term in ['network', 'resilience', 'cascade']):
            return 'Network Analysis'
        elif any(term in name_lower for term in ['distribution', 'histogram']):
            return 'Statistical Distribution'
        else:
            return 'General Analysis'
    
    def _categorize_table(self, table_name):
        """Categorize table by content"""
        name_lower = table_name.lower()
        
        if any(term in name_lower for term in ['dataset', 'statistics', 'descriptive']):
            return 'Dataset Description'
        elif any(term in name_lower for term in ['hypothesis', 'test', 'statistical']):
            return 'Statistical Tests'
        elif any(term in name_lower for term in ['performance', 'model', 'metrics']):
            return 'Model Performance'
        elif any(term in name_lower for term in ['economic', 'impact', 'cost']):
            return 'Economic Analysis'
        elif any(term in name_lower for term in ['threshold', 'risk']):
            return 'Risk Analysis'
        else:
            return 'General Results'
    
    def generate_latex_includes(self, figure_catalog, table_catalog):
        """Generate LaTeX include files"""
        print("Generating LaTeX include files...")
        
        # Generate figure includes
        fig_include_path = self.figures_dir / "all_figures.tex"
        with open(fig_include_path, 'w') as f:
            f.write("% Auto-generated figure includes for Slovenia Traffic Analysis\n")
            f.write("% Generated by generate_figure_catalog.py\n")
            f.write(f"% Generated on: {datetime.now().isoformat()}\n\n")
            
            current_category = None
            for fig in figure_catalog:
                if fig['category'] != current_category:
                    f.write(f"\n% {fig['category']} Figures\n")
                    current_category = fig['category']
                
                f.write(f"% Figure {fig['figure_number']}: {fig['name']}\n")
                f.write(f"% \\includegraphics[width=0.8\\textwidth]{{{fig['path_relative']}}}\n\n")
        
        # Generate table includes  
        tab_include_path = self.tables_dir / "all_tables_includes.tex"
        with open(tab_include_path, 'w') as f:
            f.write("% Auto-generated table includes for Slovenia Traffic Analysis\n")
            f.write("% Generated by generate_figure_catalog.py\n")
            f.write(f"% Generated on: {datetime.now().isoformat()}\n\n")
            
            current_category = None
            for table in table_catalog:
                if table['category'] != current_category:
                    f.write(f"\n% {table['category']} Tables\n")
                    current_category = table['category']
                
                f.write(f"% Table: {table['table_name']}\n")
                f.write(f"\\input{{{table['path_relative']}}}\n\\clearpage\n\n")
        
        print(f"  → LaTeX includes saved:")
        print(f"    Figures: {fig_include_path}")
        print(f"    Tables: {tab_include_path}")
    
    def create_appendix_content(self, figure_catalog, table_catalog):
        """Create appendix content for the article"""
        print("Creating appendix content...")
        
        appendix_path = self.reports_dir / "article" / "tex" / "appendix_figures_tables.tex"
        
        with open(appendix_path, 'w') as f:
            f.write("% Figures and Tables Appendix\n")
            f.write("% Auto-generated by generate_figure_catalog.py\n\n")
            
            f.write("\\section{Complete Figure and Table Catalog}\n\n")
            
            # Figure catalog table
            f.write("\\subsection{Figure Catalog}\n\n")
            f.write("\\begin{longtable}{|l|p{4cm}|l|r|}\n")
            f.write("\\hline\n")
            f.write("\\textbf{Figure} & \\textbf{Description} & \\textbf{Category} & \\textbf{Size (KB)} \\\\\n")
            f.write("\\hline\n")
            f.write("\\endhead\n")
            
            for fig in figure_catalog:
                desc = fig['name'].replace('_', '\\_')
                f.write(f"Fig. {fig['figure_number']} & {desc} & {fig['category']} & {fig['size_kb']} \\\\\n")
                f.write("\\hline\n")
            
            f.write("\\end{longtable}\n\n")
            
            # Table catalog table
            f.write("\\subsection{Table Catalog}\n\n")
            f.write("\\begin{longtable}{|l|p{4cm}|l|r|}\n")
            f.write("\\hline\n")
            f.write("\\textbf{Table} & \\textbf{Description} & \\textbf{Category} & \\textbf{Size (KB)} \\\\\n")
            f.write("\\hline\n")
            f.write("\\endhead\n")
            
            for table in table_catalog:
                desc = table['table_name'].replace('_', '\\_')
                f.write(f"Tab. {desc} & {table['filename']} & {table['category']} & {table['size_kb']} \\\\\n")
                f.write("\\hline\n")
            
            f.write("\\end{longtable}\n\n")
            
            # Summary statistics
            f.write("\\subsection{Summary Statistics}\n\n")
            f.write("\\begin{itemize}\n")
            f.write(f"\\item Total figures extracted: {len(figure_catalog)}\n")
            f.write(f"\\item Total tables extracted: {len(table_catalog)}\n")
            
            # Figure categories
            fig_categories = pd.Series([fig['category'] for fig in figure_catalog]).value_counts()
            f.write("\\item Figure categories:\n")
            f.write("\\begin{itemize}\n")
            for cat, count in fig_categories.items():
                f.write(f"\\item {cat}: {count}\n")
            f.write("\\end{itemize}\n")
            
            # Table categories
            table_categories = pd.Series([table['category'] for table in table_catalog]).value_counts()
            f.write("\\item Table categories:\n")
            f.write("\\begin{itemize}\n")
            for cat, count in table_categories.items():
                f.write(f"\\item {cat}: {count}\n")
            f.write("\\end{itemize}\n")
            
            f.write("\\end{itemize}\n\n")
            
            # Reproducibility note
            f.write("\\subsection{Reproducibility}\n\n")
            f.write("All figures and tables in this article were programmatically extracted from ")
            f.write("Jupyter notebooks using standardized scripts. The complete extraction process ")
            f.write("is documented and reproducible. Source notebooks and extraction scripts are ")
            f.write("available in the project repository.\n\n")
            
            f.write(f"\\textbf{{Extraction completed:}} {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        print(f"  → Appendix content saved: {appendix_path}")
    
    def generate_json_catalog(self, figure_catalog, table_catalog):
        """Generate JSON catalog for programmatic access"""
        print("Generating JSON catalog...")
        
        full_catalog = {
            'metadata': {
                'generation_date': datetime.now().isoformat(),
                'total_figures': len(figure_catalog),
                'total_tables': len(table_catalog),
                'extraction_method': 'automated_notebook_processing'
            },
            'figures': figure_catalog,
            'tables': table_catalog,
            'statistics': {
                'figure_categories': pd.Series([fig['category'] for fig in figure_catalog]).value_counts().to_dict(),
                'table_categories': pd.Series([table['category'] for table in table_catalog]).value_counts().to_dict(),
                'total_size_kb': sum(fig['size_kb'] for fig in figure_catalog) + sum(table['size_kb'] for table in table_catalog)
            }
        }
        
        catalog_path = self.reports_dir / "complete_figure_table_catalog.json"
        with open(catalog_path, 'w') as f:
            json.dump(full_catalog, f, indent=2)
        
        print(f"  → JSON catalog saved: {catalog_path}")
        return full_catalog
    
    def print_summary(self, catalog):
        """Print summary of extraction"""
        print("\n" + "="*60)
        print("FIGURE AND TABLE EXTRACTION SUMMARY")
        print("="*60)
        
        print(f"📊 Total figures: {catalog['metadata']['total_figures']}")
        print(f"📋 Total tables: {catalog['metadata']['total_tables']}")
        print(f"💾 Total size: {catalog['statistics']['total_size_kb']:,} KB")
        
        print(f"\nFigure categories:")
        for category, count in catalog['statistics']['figure_categories'].items():
            print(f"  • {category}: {count}")
        
        print(f"\nTable categories:")
        for category, count in catalog['statistics']['table_categories'].items():
            print(f"  • {category}: {count}")
        
        print(f"\nOutput locations:")
        print(f"  • Figures: {self.figures_dir}")
        print(f"  • Tables: {self.tables_dir}")
        print(f"  • Catalog: {self.reports_dir}")

def main():
    """Generate complete catalog"""
    print("📚 FIGURE AND TABLE CATALOG GENERATION")
    print("="*50)
    
    cataloger = FigureTableCatalog()
    
    # Create catalogs
    figure_catalog = cataloger.catalog_figures()
    table_catalog = cataloger.catalog_tables()
    
    # Generate outputs
    cataloger.generate_latex_includes(figure_catalog, table_catalog)
    cataloger.create_appendix_content(figure_catalog, table_catalog)
    full_catalog = cataloger.generate_json_catalog(figure_catalog, table_catalog)
    
    # Print summary
    cataloger.print_summary(full_catalog)
    
    print("\n✅ Complete catalog generation finished!")

if __name__ == "__main__":
    main()