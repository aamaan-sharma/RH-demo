#!/bin/bash
# Test runner script for the modular Flask application

set -e

echo "=========================================="
echo "Running Test Suite for Modular Flask App"
echo "=========================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if pytest is installed
if ! command -v pytest &> /dev/null; then
    echo -e "${YELLOW}Warning: pytest not found. Installing test dependencies...${NC}"
    pip install -r requirements-test.txt
fi

# Run tests with coverage
echo -e "${GREEN}Running unit tests...${NC}"
pytest tests/test_services/ tests/test_utils/ -v --cov=app --cov-report=term-missing

echo ""
echo -e "${GREEN}Running route tests...${NC}"
pytest tests/test_routes/ -v --cov=app --cov-report=term-missing

echo ""
echo -e "${GREEN}Running integration tests...${NC}"
pytest tests/test_integration/ -v --cov=app --cov-report=term-missing

echo ""
echo -e "${GREEN}Running all tests with coverage report...${NC}"
pytest tests/ -v --cov=app --cov-report=html --cov-report=term-missing

echo ""
echo -e "${GREEN}=========================================="
echo "Test Suite Complete!"
echo "==========================================${NC}"
echo ""
echo "Coverage report generated in htmlcov/index.html"
echo ""

# Check for any imports from app.py
echo -e "${YELLOW}Checking for remaining app.py imports...${NC}"
if grep -r "from app import\|import app" . --exclude-dir=node_modules --exclude-dir=__pycache__ --exclude-dir=.git --exclude="app.py" --exclude="*.md" --exclude="*.sh" &> /dev/null; then
    echo -e "${RED}WARNING: Found imports from app.py in:${NC}"
    grep -r "from app import\|import app" . --exclude-dir=node_modules --exclude-dir=__pycache__ --exclude-dir=.git --exclude="app.py" --exclude="*.md" --exclude="*.sh"
else
    echo -e "${GREEN}✓ No remaining app.py imports found${NC}"
fi

echo ""
echo -e "${YELLOW}Checking for references to old functions...${NC}"
OLD_FUNCTIONS=("process_single_transcript_question" "generate_claim_decision_from_chunks" "_normalize_chunks_with_names" "_generate_chunk_name")
for func in "${OLD_FUNCTIONS[@]}"; do
    if grep -r "$func" . --exclude-dir=node_modules --exclude-dir=__pycache__ --exclude-dir=.git --exclude="app.py" --exclude="*.md" --exclude="*.sh" &> /dev/null; then
        echo -e "${RED}WARNING: Found reference to old function: $func${NC}"
    else
        echo -e "${GREEN}✓ No references to $func found${NC}"
    fi
done

echo ""
echo -e "${GREEN}All checks complete!${NC}"
