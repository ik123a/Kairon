# Publishing Kairon to PyPI

The package `kairon-cache` (PyPI name: `kairon_cache`) is built and
ready in `dist/`. To publish:

## Prerequisites

1. Have a PyPI account at https://pypi.org/
2. Create an API token at https://pypi.org/manage/account/token/
3. Configure the token (one-time):
   ```
   python -m keyring set https://upload.pypi.org/legacy/ __token__
   # Paste your API token (starts with `pypi-...`) when prompted
   ```

## Test publish (TestPyPI)

```bash
cd "C:/Users/SKV/Desktop/projects/kairon"
python -m twine upload --repository testpypi dist/*
# Browse: https://test.pypi.org/project/kairon-cache/
```

## Production publish (PyPI)

```bash
cd "C:/Users/SKV/Desktop/projects/kairon"
python -m twine upload dist/*
# Browse: https://pypi.org/project/kairon-cache/
```

## Verify install

```bash
# After publication:
python -m venv /tmp/kairon-test
/tmp/kairon-test/bin/pip install kairon-cache
python -c "from kairon import CausalRouter; print('installed OK')"
```

## CI/CD (recommended)

Add this to `.github/workflows/release.yml`:

```yaml
name: Publish to PyPI
on:
  push:
    tags: ['v*']
jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - run: pip install build
      - run: python -m build
      - uses: pypa/gh-action-pypi-publish@release/v1
        with:
          packages_dir: dist/
```

Configure trusted publishing on PyPI:
- Project URL: `https://github.com/ik123a/Kairon`
- Workflow filename: `release.yml`
