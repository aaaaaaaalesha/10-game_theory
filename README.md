# Game Theory. 10 semester (5 year).

## Quick Start

1. Installing Poetry to manage dependencies.
```bash
pip3 install poetry=1.7.1
export PATH="$HOME/.local/bin:$PATH"
```

2. All dependencies have already locked in poetry.lock, you just need to install it. 
```bash
poetry install
```

3. And now you're ready to run jupyter notebook using env based on poetry dependencies.
```bash
poetry run jupyter-notebook
```