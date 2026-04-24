# Internal

This page contains internal instructions for maintaining and releasing the project.



## Documentation

These pages are generated with mkdocs.

To set things up, perform the following steps (substitute name of venv if needed).

```
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -e .[docs]
```

To start a local server, enter:
```
mkdocs serve
```
 

# PyPI release

Install the packages build and twine. Then

```bash
python -m build
python -m twine upload dist/*
```