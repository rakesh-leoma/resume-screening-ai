# Resume Screening AI Web App

This Streamlit app adapts the pipeline in `main.ipynb` into a local web application.

## Features
- Upload a job description as TXT, PDF, DOCX, or paste text
- Upload multiple resumes as PDF, DOCX, or TXT
- Extract and preprocess resume text
- Split resume/JD text into sections
- Generate normalized BGE embeddings with `BAAI/bge-large-en-v1.5`
- Calculate section-wise semantic similarity and show matching evidence
- Rank candidates and export CSV results
- Extract email, phone, LinkedIn, GitHub, and a heuristic name
- No model training is performed by this app

## Setup on Windows (VS Code)

1. Install Python 3.10 or 3.11.
2. Extract the ZIP into a project folder.
3. Open that folder in VS Code.
4. Open Terminal and run:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, use Command Prompt activation instead:

```bat
.venv\Scripts\activate.bat
```

5. Start the app:

```powershell
streamlit run app.py
```

6. Open the local URL shown in the terminal, usually `http://localhost:8501`.

## First run
The app downloads the BGE embedding model the first time it runs. This can take time and requires internet access. Subsequent runs normally use the local model cache.

## Important notes
- Upload only resumes you are authorized to process.
- Scanned/image-only PDFs may produce no text and may need OCR.
- Relative scores are min-max scaled within the current batch. They are not absolute probabilities of suitability.
- Review evidence and candidate details manually. Do not use the score as the sole hiring decision.
- This app uses the BGE embedding model from the notebook. It does not include the optional NER model; name extraction is a simple header heuristic to keep deployment lighter and more reliable.
