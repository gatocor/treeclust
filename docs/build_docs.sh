#!/bin/bash
# Quick documentation build script for treeclust

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Building TreeClust Documentation${NC}"

# Check if we're in the right directory
if [ ! -f "conf.py" ]; then
    echo -e "${RED}Error: conf.py not found. Please run this script from the docs/ directory.${NC}"
    exit 1
fi

# Check if sphinx is installed
if ! command -v sphinx-build &> /dev/null; then
    echo -e "${RED}Error: sphinx-build not found. Please install with:${NC}"
    echo "  pip install \"treeclust[docs]\""
    exit 1
fi

# Clean previous build
echo -e "${YELLOW}Cleaning previous build...${NC}"
rm -rf _build

# Build HTML documentation
echo -e "${YELLOW}Building HTML documentation...${NC}"
sphinx-build -b html . _build/html

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Documentation built successfully!${NC}"
    echo -e "${GREEN}Open docs/_build/html/index.html in your browser${NC}"
    
    # Try to open automatically on macOS
    if [[ "$OSTYPE" == "darwin"* ]]; then
        echo -e "${YELLOW}Opening documentation in default browser...${NC}"
        open _build/html/index.html
    fi
else
    echo -e "${RED}✗ Documentation build failed${NC}"
    exit 1
fi