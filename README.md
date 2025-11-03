## ResumeBuilder

Minimal Python CLI that selects relevant resume content via OpenAI and renders a LaTeX PDF.

### Structure
- `main.py`: Orchestrates selection and rendering
- `generate.py`: Renders LaTeX and compiles PDF
- `latex.txt`: LaTeX template with placeholders
- `data/jobs.json`, `data/projects.json`: Source content
- `data/personal.json`: Personal info (name, contact, education) used across all resumes

### Install
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Prepare data
1. Edit `data/jobs.json` and `data/projects.json` to include your roles and projects. IDs must be unique.
2. Edit `data/personal.json` with your name, contact information, and education. This will be used consistently across all generated resumes.

### Run
```bash
python main.py \
  --output resume.pdf \
  --input job_description.txt \
  --key $OPENAI_API_KEY
```

Personal info (name, contact, education) is automatically loaded from `data/personal.json`. You can override with `--name`, `--contact`, or `--education` flags if needed.

The system will use `tectonic` (if installed) or fall back to `pdflatex` for PDF generation. If neither is available, a `.tex` file will be created.

### Template placeholders
- `%%NAME%%`
- `%%CONTACT_LINE%%`
- `%%SKILLS_BLOCKS%%`
- `%%EXPERIENCE_BLOCKS%%`
- `%%PROJECT_BLOCKS%%`
- `%%EDUCATION_BLOCKS%%`



