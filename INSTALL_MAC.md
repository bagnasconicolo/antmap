# macOS Installation and Usage

This guide explains how to download, install, and run AntMap on macOS.

## Prerequisites

- [Homebrew](https://brew.sh/) for package management
- Git
- Python 3 (install with Homebrew if needed: `brew install python`)

## Download the source

Open Terminal and clone the repository:

```bash
git clone https://github.com/OWNER/antmap.git
cd antmap
```

Replace `OWNER` with the appropriate GitHub username.

## Automatic setup and launch

Use the provided script to create a virtual environment, install dependencies, and start the application:

```bash
chmod +x run_mac.sh
./run_mac.sh
```

The script will:

- Create a `.venv` directory if one does not exist
- Upgrade `pip`
- Install PyQt5 when missing
- Launch the AntMap editor

## Manual setup (optional)

If you prefer to manage dependencies yourself, run the following commands:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install PyQt5
python main.py
```

An example concept map is available in `examples/Untit.cxl`.
