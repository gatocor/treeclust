# Building TreeClust Documentation

This directory contains the Sphinx documentation for treeclust.

## Requirements

To build the documentation, you'll need:

```bash
# Install documentation dependencies
pip install "treeclust[docs]"

# Or install manually
pip install sphinx sphinx-rtd-theme
```

## Building the Documentation

### Method 1: Using Make (Recommended)

```bash
# From the docs/ directory
cd docs
make html

# Clean build directory
make clean

# Build other formats
make latexpdf  # PDF (requires LaTeX)
make epub      # ePub format
```

### Method 2: Using sphinx-build directly

```bash
# From the docs/ directory  
cd docs
sphinx-build -b html . _build/html

# Clean build
rm -rf _build

# Build PDF
sphinx-build -b latex . _build/latex
cd _build/latex && make
```

### Method 3: From project root

```bash
# From the treeclust/ root directory
sphinx-build -b html docs docs/_build/html
```

## Viewing the Documentation

After building, open `_build/html/index.html` in your browser.

## Documentation Structure

- `index.rst` - Main documentation page
- `installation.rst` - Installation guide
- `quickstart.rst` - Quick start tutorial
- `api/` - API reference documentation
- `examples/` - Example gallery
- `tutorials/` - In-depth tutorials
- `conf.py` - Sphinx configuration

## Auto-documentation

The documentation uses Sphinx autodoc to automatically generate API documentation from docstrings in the source code. Make sure your docstrings follow the Google or NumPy style guide.

## Contributing to Documentation

1. Edit the relevant `.rst` files
2. Build the documentation locally to test
3. Submit a pull request with your changes

## Troubleshooting

**Import errors during build:**
- Make sure treeclust is installed in your environment
- Check that all dependencies are available

**Missing modules:**
- Ensure all treeclust modules are properly imported in `__init__.py` files

**Build errors:**
- Check syntax of `.rst` files
- Verify all referenced files exist