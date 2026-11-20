# Deploying the Student Early-Warning model (FastAPI + Render + Postman)

Final model: Logistic Regression (L2), one model per checkpoint (A admission, B after sem 1, C after sem 2).
Files: main.py (API), engine.py (prediction logic), model_bundle.joblib (trained models),
requirements.txt, train_model.py + features.py + split_indices.json (only needed to retrain).

## Part A - Local (do this first)
1. Put the folder somewhere simple, e.g. C:\dropout_api, open it in VS Code, open a terminal (PowerShell).
2. Install:   python -m pip install fastapi uvicorn pydantic pandas numpy joblib scikit-learn
3. (Recommended, 10 seconds) Rebuild the model with YOUR library versions. Copy data.csv in too, then:
      python train_model.py --data data.csv --split split_indices.json
   Expect:  split fingerprint: 598f3aad...   and   Saved model_bundle.joblib
4. Start the server:   python -m uvicorn main:app --reload
   Expect:  Uvicorn running on http://127.0.0.1:8000
5. Browser checks:  http://127.0.0.1:8000/health  and  http://127.0.0.1:8000/docs
6. Postman:  POST  http://127.0.0.1:8000/predict   Body -> raw -> JSON:
{
  "previous_qualification_percent": 72, "admission_percent": 68, "hostel": 1,
  "first_generation": 1, "scholarship": 0, "fee_dues": 1,
  "semester1": {"registered": 6, "passed": 3, "sgpa": 5.5},
  "semester2": {"registered": 6, "passed": 2, "sgpa": 4.8}
}
   Expect checkpoint "C", predicted_outcome "Dropout", dropout_probability about 0.78, flag_for_counsellor_review true.
   Leave out semester2 -> checkpoint B.  Leave out semester1 and semester2 -> checkpoint A.
   A strong student (passed 6/6, SGPA 8.5, no fee dues) -> Graduate, not flagged.
   Bad input (passed > registered, percent 150) -> 422 error: that is correct behaviour.

## Part B - Public URL with Render (what your faculty chose)
1. Create a GitHub repo and upload ONLY: main.py, engine.py, model_bundle.joblib, requirements.txt
   (do not upload data.csv). On the repo page: Add file -> Upload files.
2. render.com -> sign in with GitHub -> New + -> Web Service -> pick the repo.
3. Settings:  Language Python 3 | Build Command:  pip install -r requirements.txt
   Start Command:  uvicorn main:app --host 0.0.0.0 --port $PORT      | Instance type: Free
   Environment variable:  PYTHON_VERSION = 3.12.10
4. Create Web Service. Wait for "Your service is live". Your URL: https://<name>.onrender.com
5. Test: https://<name>.onrender.com/docs  then Postman POST https://<name>.onrender.com/predict (same JSON).
   Free tier sleeps after ~15 min idle; the first request can take about a minute. Open /health before a demo.

## Part B2 - ngrok backup (if Render fails; laptop must stay on)
1. ngrok.com -> free account -> download -> ngrok config add-authtoken YOUR_TOKEN (never share the token)
2. Keep uvicorn running (Part A), in a second terminal:  ngrok http 8000
3. Use the https://....ngrok-free.dev address + /docs or /predict.

## Troubleshooting
- No module named 'engine' / 'main': run commands from inside the project folder.
- Build fails on Render with a scikit-learn version error: change requirements.txt to your local
  version (python -m pip show scikit-learn) and re-run train_model.py locally, then re-upload model_bundle.joblib.
- Pickle/unpickle error on Render: model was trained with a different scikit-learn than requirements.txt. Same fix.
- 502 on ngrok: uvicorn is not running on port 8000.
- 422 in Postman: body must be raw JSON (Content-Type application/json) and values within range.
- Port in use: python -m uvicorn main:app --reload --port 8001
