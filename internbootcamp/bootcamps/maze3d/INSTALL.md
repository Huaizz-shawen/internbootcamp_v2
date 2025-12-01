# Installation Guide - 3D Maze RLVR Bootcamp

## Prerequisites

- Python 3.10 or higher
- conda (recommended) or pip

## Method 1: Using Conda (Recommended)

### Option A: Create New Environment

```bash
# From the maze3d directory
cd /path/to/internbootcamp_v2/internbootcamp/bootcamps/maze3d

# Create environment from file
conda env create -f environment.yml

# Activate environment
conda activate maze3d
```

### Option B: Update Existing Environment

```bash
# Activate your environment
conda activate maze3d

# Install/update packages
conda env update -f environment.yml
```

## Method 2: Using Pip

### From requirements.txt

```bash
# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Manual Installation

```bash
pip install numpy matplotlib pyyaml tqdm pillow datasets \
            starlette pydantic openai urllib3 click requests
```

## Verify Installation

Run the test script to verify everything is working:

```bash
cd /path/to/internbootcamp_v2

# Using conda environment
/home/user/miniconda3/envs/maze3d/bin/python \
  internbootcamp/bootcamps/maze3d/examples/simple_test.py

# Or if environment is activated
python internbootcamp/bootcamps/maze3d/examples/simple_test.py
```

Expected output:
```
============================================================
SIMPLE MAZE3D RLVR TEST
============================================================
...
ALL TESTS PASSED! ✓
============================================================
```

## Package Summary

### Core Dependencies (Required)
- **numpy**: Numerical computations
- **matplotlib**: 3D maze visualization
- **pyyaml**: Configuration file parsing
- **tqdm**: Progress bars
- **pillow**: Image processing
- **datasets**: Data format conversion

### API Dependencies (Required for Evaluation)
- **starlette**: Web framework utilities
- **pydantic**: Data validation
- **openai**: LLM API client
- **urllib3, click, requests**: HTTP and CLI utilities

### Optional Dependencies

#### For Tool Server (not needed for basic usage):
```bash
pip install uvicorn
```

#### For RL Training (install separately):
Follow VERL installation instructions from the main repository.

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'internbootcamp'"

**Solution:** Make sure you're running scripts from the repository root:

```bash
cd /path/to/internbootcamp_v2
python internbootcamp/bootcamps/maze3d/examples/generate_data.py
```

### Issue: "ModuleNotFoundError: No module named 'verl'"

**Solution:** VERL is only needed for RL training, not for data generation or evaluation. The warnings about missing VERL can be safely ignored for basic usage.

### Issue: Package conflicts with conda

**Solution:** Create a fresh environment:

```bash
# Remove old environment
conda env remove -n maze3d

# Create new one
conda env create -f environment.yml
```

### Issue: Missing packages

**Solution:** Install manually:

```bash
conda activate maze3d
pip install <package-name>
```

## Environment Comparison

| Feature | Conda | Pip + venv |
|---------|-------|------------|
| Isolation | ✓ Better | ✓ Good |
| Scientific packages | ✓ Faster | Slower |
| Reproducibility | ✓ Better | Good |
| Size | Larger | Smaller |
| Speed | Faster installs | Slower installs |

**Recommendation:** Use conda for scientific computing projects (recommended for this bootcamp).

## Next Steps

After successful installation:

1. **Test core functionality:**
   ```bash
   python internbootcamp/bootcamps/maze3d/examples/simple_test.py
   ```

2. **Generate sample data:**
   ```bash
   python internbootcamp/bootcamps/maze3d/examples/generate_data.py
   ```

3. **Read the quick start guide:**
   - See `QUICK_START.md` for usage examples
   - See `USAGE_GUIDE.md` for detailed workflow

## Support

- Check `README.md` for feature documentation
- Check `USAGE_GUIDE.md` for complete workflow
- Check `IMPLEMENTATION_SUMMARY.md` for technical details

## Uninstall

### Conda Environment

```bash
conda env remove -n maze3d
```

### Pip Virtual Environment

```bash
# Deactivate first
deactivate

# Remove directory
rm -rf venv/
```
