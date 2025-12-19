# mic-hackathon-2025

<p align="left">
  <a href="https://opensource.org/licenses/MIT">
    <img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-yellow.svg" />
  </a>
  <img alt="Python Version" src="https://img.shields.io/badge/python-3.8%2B-blue" />
  <img alt="Repo Size" src="https://img.shields.io/github/repo-size/ashriva16/mic-hackathon-2025" />
  <img alt="Last Commit" src="https://img.shields.io/github/last-commit/ashriva16/mic-hackathon-2025" />
  <a href="https://github.com/ashriva16/mic-hackathon-2025/issues">
    <img alt="Issues" src="https://img.shields.io/github/issues/ashriva16/mic-hackathon-2025" />
  </a>
  <a href="https://github.com/ashriva16/mic-hackathon-2025/pulls">
    <img alt="Pull Requests" src="https://img.shields.io/github/issues-pr/ashriva16/mic-hackathon-2025" />
  </a>
  <a href="https://github.com/ashriva16/mic-hackathon-2025/actions/workflows/ci.yml">
    <img alt="CI" src="https://github.com/ashriva16/mic-hackathon-2025/actions/workflows/ci.yml/badge.svg" />
  </a>
  <a href="https://github.com/ashriva16/mic-hackathon-2025/actions/workflows/doc.yml">
    <img alt="CI" src="https://github.com/ashriva16/mic-hackathon-2025/actions/workflows/doc.yml/badge.svg" />
  </a>
</p>

## 📌 Description

A short description of the research project

---

## 📁 Repository Setup

```sh
git clone https://github.com/ashriva16/mic-hackathon-2025.git
cd mic-hackathon-2025
```

---

## 🚀 End-User Setup & Usage

- **Use the Makefile to create a .venv and install user-level dependencies.**

    ```bash
    make env
    ```

    This creates `.venv/` and installs packages from `requirements.txt` (if present).

- **For refreshing and installing updated dependencies run**

    ```bash
    git pull        # get latest code + updated requirements.txt
    make install    # refresh dependencies inside .venv
    ```

- **To manually install packages or missing dependency in the venv**

    ```sh
    source .venv/bin/activate
    pip install <package>
    ```

- **Clean build/cache files**

    ```sh
    make clean
    ```

- **Usage**

    ```sh
    .venv/bin/python -m codes.main
    ```

---

## 🤝 Contributing

Contributions are encouraged and appreciated. To maintain a clean history, all changes must be made on feature branches (direct pushes to main may be restricted).

### Setup Development Environment

  ```sh
  python3 -m venv .venv
  pip install --upgrade pip
  source .venv/bin/activate
  pip install -e ".[dev]"
  ```

This installs:

- the project in editable mode
- dev tools (pytest, black, isort, flake8, pylint, mypy)
- docs tools (sphinx, myst-parser, nbsphinx, etc.)

### Workflow

- Follow the method to push/pull your transformations

- **Getting started**

  ```sh
  source .venv/bin/activate
  # Make sure main is up to date
  git pull origin main
  ```

- **Create your branch for your task (choose a clear name) and follow the workflow locally**

  ```sh
  git checkout -b my-task-branch

  git add .
  git commit -m "Describe what you did."
  ```

- **push to remote for the first time**

  ```sh
  git push -u origin my-task-branch
  ```

- **after that, push usually**

  ```sh
  git push
  ```

- **If you want to get updates from main into your branch**

  ```sh
  # Update your branch with the newest main changes
  git pull origin main --rebase
  ```

PR will be reviewed by admin as soon as possible.

---

## 👤 Maintainer

**Ankit Shrivastava**
Feel free to open an issue or discussion for support.

---

## 📜 License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT). See the `LICENSE` file for full details.

---

## 📈 Project Status

> Status: 🚧 In Development — Not ready for production use.

---

## 📘 References

- [Cookiecutter Docs](https://cookiecutter.readthedocs.io)
- [PEP 621](https://peps.python.org/pep-0621/)
- [GitHub Actions](https://docs.github.com/en/actions)
