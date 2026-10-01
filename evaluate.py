from pathlib import Path
import json
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix
from app.model import get_model, split_data, VERSION, SEED

_, x_test, _, y_test = split_data()
actual=get_model().predict(x_test)
metrics={'model_version':VERSION,'split_seed':SEED,'holdout_samples':len(y_test),
 'accuracy':float(accuracy_score(y_test,actual)),
 'per_class_recall':recall_score(y_test,actual,average=None).tolist(),
 'confusion_matrix':confusion_matrix(y_test,actual).tolist()}
Path('reports').mkdir(exist_ok=True)
Path('reports/metrics.json').write_text(json.dumps(metrics,indent=2))
print(json.dumps(metrics,indent=2))
