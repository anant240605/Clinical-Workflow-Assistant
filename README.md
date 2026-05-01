# Clinical Workflow Assistant

An MVP healthcare AI agent pipeline for processing raw clinical notes into structured extraction, reasoning recommendations, and readable clinical reports.

## Features

- FastAPI backend with modular agent pipeline
- SQLite persistence for analyzed reports
- React/Vite demo UI
- Editable prompts in `backend/prompts.json`
- Groq-compatible LLM provider with mock fallback
- Mock medical reference API and local file report storage
- Docker and Docker Compose support

## Architecture

```text
Doctor Note
   |
   v
Input Agent
   |
   v
Processing Agent
   |
   v
Medical Extraction Agent ----> Mock Medical API
   |
   v
Reasoning Agent -------------> Groq/OpenAI-compatible LLM or Mock LLM
   |
   v
Report Generation Agent
   |
   +--> SQLite database
   +--> backend/storage/reports/{id}.json
   +--> REST API response
```

## API Flow

1. `POST /analyze-note` accepts raw clinical note text.
2. The backend loads prompts dynamically from `backend/prompts.json`.
3. Agents run in order: input validation, text cleanup, extraction, reasoning, and report generation.
4. The final report is saved to SQLite and as a JSON file.
5. `GET /report/{id}` returns a previously generated report.

## Run Locally

### Backend

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Edit `.env` and set:

```text
GROQ_API_KEY=your_free_groq_api_key
LLM_PROVIDER=groq
```

Then start:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

If you do not set `GROQ_API_KEY`, the app automatically uses the deterministic mock LLM.

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Open the Vite URL, usually `http://localhost:5173`.

## Run With Docker

```bash
copy backend\.env.example backend\.env
```

Set your Groq key in `backend/.env`, then:

```bash
docker compose up --build
```

Open:

- Frontend: `http://localhost:5173`
- Backend docs: `http://localhost:8000/docs`

## Sample Request

```bash
curl -X POST http://localhost:8000/analyze-note ^
  -H "Content-Type: application/json" ^
  -d "{\"note\":\"Patient reports fever for 3 days, mild cough, fatigue. Temperature 101 F. No prior conditions.\"}"
```

## Example Output JSON

```json
{
  "id": "f2d10b9d-93a4-4b03-9d82-44e9b4dd9f6a",
  "summary": "Patient presents with 3 days of fever, mild cough, fatigue, and temperature of 101 F with no known prior conditions.",
  "extracted_info": {
    "symptoms": ["fever for 3 days", "mild cough", "fatigue"],
    "diagnoses": [],
    "medications": [],
    "vital_signs": [{"name": "temperature", "value": "101 F"}]
  },
  "recommendations": {
    "possible_conditions": ["viral upper respiratory infection", "influenza-like illness", "COVID-19 or other respiratory infection"],
    "recommended_tests": ["COVID-19 antigen/PCR", "influenza test", "CBC if symptoms worsen or persist"],
    "medications": ["acetaminophen or ibuprofen if clinically appropriate", "hydration and rest"],
    "follow_ups": ["follow up in 48-72 hours if fever persists", "seek urgent care for shortness of breath, chest pain, confusion, or worsening symptoms"],
    "safety_notes": ["AI recommendations require clinician review and are not a final diagnosis."]
  },
  "report": "Clinical Workflow Assistant Report..."
}
```

## Prompt Transparency

All agent prompts live in:

```text
backend/prompts.json
```

Update this file to tune extraction, reasoning, and report wording without changing Python code.

## Important Clinical Note

This MVP is decision-support software for demonstration. It does not replace licensed clinical judgment, local protocols, medication reconciliation, allergy checks, or emergency evaluation.
