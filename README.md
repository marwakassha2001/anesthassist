# AnesthAssist — Setup Guide


## Step 1 — Open in VS Code

```bash
code anesthassist
```

---

## Step 2 — Start the backend

Open a terminal in VS Code and run:

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
```

Backend runs at: http://127.0.0.1:8000
API docs at:     http://127.0.0.1:8000/docs

---

## Step 3 — Start the frontend

Open a second terminal in VS Code and run:

```bash
cd frontend
npm install
npm run dev
```

Frontend runs at: http://localhost:3000

---

## How memory works

Every time you assess a patient:
- Their result is saved to backend/patients.json
- They appear in the left sidebar
- Click any patient to see their full visit history
- Use the same Patient ID to add a new visit to an existing patient
- Click "New Patient" to reset the form with a fresh auto-generated ID

---

## API Endpoints

| Method | Endpoint          | Description                   |
|--------|-------------------|-------------------------------|
| POST   | /predict          | Run prediction + save result  |
| GET    | /patients         | List all patients             |
| GET    | /patients/{id}    | Full history for one patient  |
| DELETE | /patients/{id}    | Delete a patient record       |
| GET    | /health           | Server health check           |

---

## Retrain the model (optional)

```bash
cd backend
python train_and_save.py --data "final data ag  3.xlsx"
```
