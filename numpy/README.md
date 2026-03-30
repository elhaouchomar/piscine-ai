# Python Exercises (NumPy + JupyterLab)

This repository contains exercises (`ex00`–`ex09`) using **Python**, **NumPy**, and **JupyterLab**.  
Follow these instructions to set up the environment, install dependencies, and run the exercises.

---

## 1. Requirements

- Python 3.9 or higher
- `pip` (Python package manager)
- Optional: `virtualenv` (recommended)

---

## 2. Setting up a virtual environment (recommended)

It is best to use a **single global environment** for all exercises:

### Linux / macOS
```bash
# Create virtual environment (once)
python3 -m venv ~/py_env

# Activate environment
source ~/py_env/bin/activate
```

## 3. Install dependencies
All required libraries are listed in requirements.txt:

```bash
pip install -r requirements.txt
```

## 4. Running JupyterLab
After activating the environment:

```bash
jupyter lab
```

Open the notebook for the exercise you want, e.g., Notebook_ex00.ipynb.

You can also run Python scripts directly:

```bash
python ex01/ex01.py
```

## 5. Workflow Tips
Always activate the virtual environment before installing or running code:

```bash
source ~/py_env/bin/activate
```

If you add new libraries, update **requirements.txt**:

```bash
pip freeze > requirements.txt
```

This ensures anyone who receives your project can install all dependencies easily.

## 6. Optional: Automatically detect dependencies
You can generate a minimal **requirements.txt** based on imports using **pipreqs**:

```bash
pip install pipreqs
pipreqs /path/to/project
```

This scans your code and creates **requirements.txt** automatically.

## 7. Deactivating the environment
When done, you can deactivate the environment:

```bash
deactivate
```