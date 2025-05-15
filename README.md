# AMAS - an mobility ai assistant

## Requirements
- Python 3.8+
- uv (https://github.com/urpylka/uv)

## Installation
1. Create and activate virtual environment:
   ```bash
   uv venv
   uv source
   ```
2. Install dependencies:
   ```bash
   uv install
   ```

## Usage
Run the application:
```bash
uv run main.py
```

## Todo
- [] train yolo-seg or other segmentation model with cityscrapes dataset
- [] use cool math to split screen to a reasonable block size, detects walking path
- [] video
