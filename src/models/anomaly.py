from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
import pandas as pd

def run_isolation_forest(
    X,
    contamination=0.032,
    random_state=42,
    n_estimators=300
):
    model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )
    
    model.fit(X)
    
    score = -model.score_samples(X)
    
    prediction = model.predict(X)
    
    return {
        "model": model,
        "score": score,
        "prediction": prediction,
    }
    
def run_lof(
    X,
    contamination=0.032,
    n_neighbors=50,
):
    model = LocalOutlierFactor(
        n_neighbors=n_neighbors,
        contamination=contamination,
        novelty=False,
        n_jobs=-1,
    )

    prediction = model.fit_predict(X)

    score = -model.negative_outlier_factor_

    return {
        "model": model,
        "score": score,
        "prediction": prediction,
    }