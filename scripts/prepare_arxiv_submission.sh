#!/bin/bash

# prepare_arxiv_submission.sh
# Script to prepare arXiv submission package for "Statistical Evidence for Highway Expansion Necessity"
# Author: Niko Gamulin
# Date: January 2025

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}arXiv Submission Package Preparation${NC}"
echo -e "${GREEN}========================================${NC}"

# Configuration
BASE_DIR="/home/niko/workspace/slovenia-trafffic-v2"
SOURCE_TEX="$BASE_DIR/reports/article/tex"
SOURCE_FIGURES="$BASE_DIR/reports/article/figures"
SOURCE_TABLES="$BASE_DIR/reports/article/tables"
SUBMISSION_DIR="$BASE_DIR/reports/arxiv_submission"
ARCHIVE_NAME="gamulin_highway_expansion_arxiv.tar.gz"

# Step 1: Clean and create submission directory
echo -e "\n${YELLOW}Step 1: Preparing submission directory...${NC}"
if [ -d "$SUBMISSION_DIR" ]; then
    echo "  Cleaning existing submission directory..."
    rm -rf "$SUBMISSION_DIR"/*
else
    echo "  Creating submission directory..."
    mkdir -p "$SUBMISSION_DIR"
fi
mkdir -p "$SUBMISSION_DIR/figures"
mkdir -p "$SUBMISSION_DIR/tables"
echo -e "  ${GREEN}✓${NC} Directory structure created"

# Step 2: Copy and fix main.tex
echo -e "\n${YELLOW}Step 2: Processing main.tex...${NC}"
cp "$SOURCE_TEX/main.tex" "$SUBMISSION_DIR/"
# Fix figure paths
sed -i 's|../figures/|figures/|g' "$SUBMISSION_DIR/main.tex"
# Fix table paths
sed -i 's|../tables/|tables/|g' "$SUBMISSION_DIR/main.tex"
echo -e "  ${GREEN}✓${NC} main.tex copied and paths fixed"

# Step 3: Copy bibliography
echo -e "\n${YELLOW}Step 3: Copying bibliography...${NC}"
cp "$SOURCE_TEX/references.bib" "$SUBMISSION_DIR/"
echo -e "  ${GREEN}✓${NC} references.bib copied"

# Step 4: Copy figures used in the document
echo -e "\n${YELLOW}Step 4: Copying figures...${NC}"
FIGURES=(
    "fig_24_speed_density_relationship.pdf"
    "fig_30_roc_curve_accident_prediction.pdf"
    "fig_21_traffic_volume_distribution.pdf"
    "fig_09_capacity_utilization_projection.pdf"
    "fig_34_economic_cost_waterfall.pdf"
    "fig_15_failure_probability_distribution.pdf"
    "fig_17_network_graph_centrality.pdf"
)

for fig in "${FIGURES[@]}"; do
    if [ -f "$SOURCE_FIGURES/$fig" ]; then
        cp "$SOURCE_FIGURES/$fig" "$SUBMISSION_DIR/figures/"
        echo -e "  ${GREEN}✓${NC} Copied $fig"
    else
        echo -e "  ${RED}✗${NC} Warning: $fig not found!"
    fi
done

# Step 5: Copy tables
echo -e "\n${YELLOW}Step 5: Copying tables...${NC}"
if [ -f "$SOURCE_TABLES/tab_model_performance_metrics.tex" ]; then
    cp "$SOURCE_TABLES/tab_model_performance_metrics.tex" "$SUBMISSION_DIR/tables/"
    echo -e "  ${GREEN}✓${NC} Copied tab_model_performance_metrics.tex"
else
    echo -e "  ${RED}✗${NC} Warning: Table file not found!"
fi

# Step 6: Test compilation
echo -e "\n${YELLOW}Step 6: Testing LaTeX compilation...${NC}"
cd "$SUBMISSION_DIR"

# First pass
echo "  Running pdflatex (first pass)..."
if pdflatex -interaction=nonstopmode main.tex > /dev/null 2>&1; then
    echo -e "  ${GREEN}✓${NC} First pass successful"
else
    echo -e "  ${RED}✗${NC} First pass failed!"
    exit 1
fi

# BibTeX
echo "  Running bibtex..."
if bibtex main > /dev/null 2>&1; then
    echo -e "  ${GREEN}✓${NC} BibTeX successful"
else
    echo -e "  ${YELLOW}!${NC} BibTeX warnings (usually normal)"
fi

# Second pass
echo "  Running pdflatex (second pass)..."
if pdflatex -interaction=nonstopmode main.tex > /dev/null 2>&1; then
    echo -e "  ${GREEN}✓${NC} Second pass successful"
else
    echo -e "  ${RED}✗${NC} Second pass failed!"
    exit 1
fi

# Third pass (for references)
echo "  Running pdflatex (final pass)..."
if pdflatex -interaction=nonstopmode main.tex > /dev/null 2>&1; then
    echo -e "  ${GREEN}✓${NC} Final pass successful"
else
    echo -e "  ${RED}✗${NC} Final pass failed!"
    exit 1
fi

# Check PDF output
if [ -f "main.pdf" ]; then
    PDF_SIZE=$(du -h main.pdf | cut -f1)
    PDF_PAGES=$(pdfinfo main.pdf 2>/dev/null | grep Pages | awk '{print $2}')
    echo -e "  ${GREEN}✓${NC} PDF generated: $PDF_SIZE, $PDF_PAGES pages"
else
    echo -e "  ${RED}✗${NC} PDF generation failed!"
    exit 1
fi

# Step 7: Clean auxiliary files
echo -e "\n${YELLOW}Step 7: Cleaning auxiliary files...${NC}"
rm -f *.aux *.log *.out *.bbl *.blg *.toc *.lof *.lot
echo -e "  ${GREEN}✓${NC} Auxiliary files removed"

# Step 8: Create archive
echo -e "\n${YELLOW}Step 8: Creating submission archive...${NC}"
cd "$BASE_DIR/reports"
tar -czf "$ARCHIVE_NAME" arxiv_submission/
ARCHIVE_SIZE=$(du -h "$ARCHIVE_NAME" | cut -f1)
echo -e "  ${GREEN}✓${NC} Archive created: $ARCHIVE_NAME ($ARCHIVE_SIZE)"

# Step 9: Verify archive contents
echo -e "\n${YELLOW}Step 9: Verifying archive contents...${NC}"
FILE_COUNT=$(tar -tzf "$ARCHIVE_NAME" | wc -l)
echo -e "  ${GREEN}✓${NC} Archive contains $FILE_COUNT files"

# Step 10: Generate file list
echo -e "\n${YELLOW}Step 10: Archive contents:${NC}"
tar -tzf "$ARCHIVE_NAME" | head -20

# Final summary
echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}Submission Package Ready!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Package location: $BASE_DIR/reports/$ARCHIVE_NAME"
echo "Submission directory: $SUBMISSION_DIR"
echo ""
echo "Next steps:"
echo "1. Review the generated PDF in $SUBMISSION_DIR/main.pdf"
echo "2. Upload $ARCHIVE_NAME to arXiv"
echo "3. Select category: stat.AP (Statistics - Applications)"
echo "4. Add secondary categories: econ.GN, physics.soc-ph"
echo ""
echo -e "${GREEN}Good luck with your submission!${NC}"