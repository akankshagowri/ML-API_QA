"""Seeded educational classifier. Holdout rows never reach fit()."""
from functools import lru_cache
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

SEED = 42
VERSION = 'iris-logistic-v1'

def split_data():
    data = load_iris()
    return train_test_split(data.data, data.target, test_size=0.3,
                            stratify=data.target, random_state=SEED)

@lru_cache(maxsize=1)
def get_model():
    x_train, _, y_train, _ = split_data()
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=300, random_state=SEED))
    model.fit(x_train, y_train)
    return model

LABELS = ['setosa', 'versicolor', 'virginica']
