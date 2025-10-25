# Linear Regression Project Setup Guide

This guide helps you set up a **completely isolated Python environment** for the Linear Regression project. All dependencies (587MB+) will be contained within the project folder.

## 🎯 **Quick Setup Commands**

### 1. **Create Virtual Environment**

```bash
# Navigate to project directory
cd /path/to/your/LinearRegression

# Create isolated virtual environment
python3 -m venv .venv
```

### 2. **Activate Environment**

```bash
# On macOS/Linux
source .venv/bin/activate

# On Windows
.venv\Scripts\activate
```

### 3. **Install Required Packages**

```bash
# Upgrade pip first
pip install --upgrade pip

# Install all data science packages
pip install pandas numpy matplotlib seaborn scikit-learn jupyter ipykernel
```

### 4. **Configure Jupyter for This Environment**

```bash
# Install IPython kernel for this environment
python -m ipykernel install --user --name=linearregression --display-name "Linear Regression Project"
```

## 📋 **Complete Setup Script**

Copy and paste this entire block to set up everything at once:

```bash
#!/bin/bash
# Linear Regression Project Setup Script

# Create virtual environment
python3 -m venv .venv

# Activate environment
source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install required packages
pip install pandas numpy matplotlib seaborn scikit-learn jupyter ipykernel

# Configure Jupyter kernel
python -m ipykernel install --user --name=linearregression --display-name "Linear Regression Project"

echo "✅ Setup complete! Environment size:"
du -sh .venv
echo "🎉 Your isolated environment is ready!"
```

## 🔧 **Daily Usage**

### **Start Working**

```bash
# Navigate to project
cd /path/to/your/LinearRegression

# Activate environment
source .venv/bin/activate

# Launch Jupyter
jupyter notebook
```

### **Stop Working**

```bash
# Deactivate environment
deactivate
```

## 📦 **Installed Packages**

Your isolated environment includes:

- **pandas** - Data manipulation and analysis
- **numpy** - Numerical computing foundation
- **matplotlib** - Basic plotting and visualization
- **seaborn** - Statistical data visualization
- **scikit-learn** - Machine learning algorithms
- **jupyter** - Interactive notebook environment
- **ipykernel** - Jupyter kernel support

**Total Size**: ~587MB (all contained in `.venv/` folder)

## 🗑️ **Clean Removal**

To completely remove all dependencies:

```bash
# Simply delete the entire project folder
rm -rf /path/to/your/LinearRegression

# Or just remove the virtual environment
rm -rf .venv
```

## 📁 **Project Structure**

```
LinearRegression/
├── .venv/                 # 587MB - All dependencies live here
├── housing.csv           # Dataset
├── main.ipynb            # Main notebook
├── maian.py              # Python script
├── SETUP.md              # This setup guide
└── HousePrice_LinearRegression/
    ├── .git/             # Git repository
    ├── .venv/            # Another isolated environment (if needed)
    └── Repository_Tracker.txt
```

## 🚀 **Verification Commands**

Check everything is working:

```bash
# Verify Python version
python --version

# List installed packages
pip list

# Check environment size
du -sh .venv

# Test imports
python -c "import pandas, numpy, matplotlib, seaborn, sklearn; print('✅ All packages working!')"
```

## 🔄 **Re-setup from Scratch**

If you need to recreate the environment:

```bash
# Remove old environment
rm -rf .venv

# Run setup script again
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install pandas numpy matplotlib seaborn scikit-learn jupyter ipykernel
python -m ipykernel install --user --name=linearregression --display-name "Linear Regression Project"
```

## 💡 **Benefits of This Setup**

✅ **Complete Isolation** - No global Python pollution  
✅ **Easy Cleanup** - Delete folder = delete all dependencies  
✅ **Reproducible** - Same environment every time  
✅ **Version Control** - Each project has its own versions  
✅ **Portable** - Works on any machine with Python 3

---

**Remember**: Always activate the environment (`source .venv/bin/activate`) before working on this project!
