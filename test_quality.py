from concurrent.futures import ThreadPoolExecutor
import pytest
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix
from app.model import split_data, LABELS

def test_heldout_accuracy_and_class_recall(client):
    _, features, _, expected = split_data()
    actual=[]
    for row in features:
        response=client.post('/predict',json={'features':row.tolist()})
        assert response.status_code == 200
        actual.append(LABELS.index(response.json()['label']))
    accuracy=accuracy_score(expected,actual)
    recalls=recall_score(expected,actual,labels=[0,1,2],average=None)
    print(f'holdout n={len(expected)} accuracy={accuracy:.4f} recalls={recalls.tolist()} confusion={confusion_matrix(expected,actual).tolist()}')
    assert accuracy >= 0.90
    assert all(value >= 0.80 for value in recalls)

@pytest.mark.parametrize('row',[[5.1,3.5,1.4,0.2],[6.7,3.1,5.6,2.4]])
def test_small_perturbations_preserve_clear_examples(client,row):
    baseline=client.post('/predict',json={'features':row}).json()['label']
    for feature in range(4):
        changed=row.copy(); changed[feature]+=0.01
        assert client.post('/predict',json={'features':changed}).json()['label'] == baseline

def test_concurrent_predictions_do_not_mix_inputs(client):
    rows=[[5.1,3.5,1.4,0.2],[6.7,3.1,5.6,2.4]]*6
    baseline=[client.post('/predict',json={'features':row}).json() for row in rows]
    with ThreadPoolExecutor(max_workers=4) as pool:
        actual=list(pool.map(lambda row:client.post('/predict',json={'features':row}).json(),rows))
    assert actual == baseline
