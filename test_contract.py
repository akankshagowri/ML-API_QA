import pytest
from app.model import LABELS, VERSION

def test_health(client):
    assert client.get('/health').json() == {'status':'ok','model_version':VERSION}

@pytest.mark.parametrize('features', [[5.1,3.5,1.4,0.2],[6,2.9,4.5,1.5],[6.7,3.1,5.6,2.4],[0,0,0,0],[20,20,20,20]])
def test_schema_probabilities_and_consistency(client,features):
    response=client.post('/predict',json={'features':features})
    assert response.status_code == 200
    data=response.json()
    assert set(data) == {'model_version','label','confidence','probabilities'}
    assert data['model_version'] == VERSION
    assert data['label'] in LABELS
    probabilities=data['probabilities']
    assert set(probabilities) == set(LABELS)
    assert all(0 <= value <= 1 for value in probabilities.values())
    assert sum(probabilities.values()) == pytest.approx(1,abs=1e-8)
    assert data['confidence'] == pytest.approx(max(probabilities.values()))
    assert data['label'] == max(probabilities,key=probabilities.get)
    for _ in range(3):
        assert client.post('/predict',json={'features':features}).json() == data

@pytest.mark.parametrize('payload',[
 {},{'features':[]},{'features':[1,2,3]},{'features':[1,2,3,4,5]},
 {'features':[-1,2,3,4]},{'features':[21,2,3,4]},
 {'features':['5',2,3,4]},{'features':[True,2,3,4]},
 {'features':[None,2,3,4]},{'features':'wrong'},
 {'features':[1,2,3,4],'unexpected':'field'},
])
def test_invalid_requests(client,payload):
    assert client.post('/predict',json=payload).status_code == 422

@pytest.mark.parametrize('body',['{bad json','{"features":[NaN,2,3,4]}','{"features":[Infinity,2,3,4]}'])
def test_non_json_and_non_finite_values(client,body):
    assert client.post('/predict',content=body,headers={'Content-Type':'application/json'}).status_code == 422
