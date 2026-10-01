# ML API QA: accuracy, consistency and edge cases

An executable AI-testing portfolio project connecting postgraduate AI study
with software QA. A real scikit-learn LogisticRegression pipeline predicts Iris
species through FastAPI. No paid API, cloud credentials or LLM service required.

## Run
Python 3.12 recommended:
```
python -m venv .venv
# activate .venv (Windows: .venv\Scripts\activate; POSIX: source .venv/bin/activate)
pip install -r requirements-lock.txt
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```
Open http://127.0.0.1:8000/docs. POST /predict:
`{"features":[5.1,3.5,1.4,0.2]}` (sepal length/width, petal length/width in cm).
The response includes label, model version, confidence and all class probabilities.

## Test and evaluate
```
python -m pytest -n 2 --html=reports/pytest.html --self-contained-html --junitxml=reports/junit.xml
python -m scripts.evaluate
```
The tests call the actual ASGI endpoint in-process via TestClient, not a mocked
prediction function. A 70/30 stratified split with seed 42 produces 45 held-out
samples; both the scaler and classifier fit only on training rows. Gates:
accuracy >=90%, recall >=80% for each species. Other checks cover response
schema, probability sum/ranges, label-confidence alignment, repeatability,
concurrent requests, small perturbations, shape/type/range validation,
extra fields, malformed JSON and NaN/Infinity. HTML/JUnit reports and a metrics
JSON with confusion matrix make failures reviewable.

## Docker / GitHub Actions
```
docker build -t ml-api-qa .
docker run --rm -p 8000:8000 ml-api-qa
docker run --rm -v "$PWD/reports:/app/reports" ml-api-qa python -m pytest -n 2 --html=reports/pytest.html --self-contained-html --junitxml=reports/junit.xml
```
Actions builds the image, runs two pytest workers and uploads reports on failure.

## Limits and release decision
Educational benchmark, not a production ML service. The 0..20 cm input bounds
are an API contract, not an out-of-distribution detector. High confidence does
not prove calibration. The holdout is small; no demographic fairness claim,
load-test result, AWS deployment or LLM evaluation is implied. Contract and
quality gate success supports a demo release only; a production release needs
representative external data, drift monitoring, authentication, load tests and
business-defined harm thresholds. Repeatedly tuning to this holdout would turn
it into validation data; reserve a fresh external test set for future tuning.

## Sources
- https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html
- https://scikit-learn.org/stable/common_pitfalls.html#data-leakage
- https://fastapi.tiangolo.com/tutorial/testing/

requirements.txt lists direct dependencies; requirements-lock.txt records the complete tested environment.

requirements.txt lists direct dependencies; requirements-lock.txt records the complete tested environment.
