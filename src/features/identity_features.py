import pandas as pd

def build_identity_features(kyc_records: pd.DataFrame) -> pd.DataFrame:
    df = kyc_records.copy()
    
    kyc_map = {
        "none": 0,
        "basic": 1,
        "full": 2,
    }
    
    df['kyc_level_ord'] = df["KYCLevel"].map(kyc_map)
    
    df['has_face_score'] = df["FaceMatchScore"].notna().astype(int)
    df['has_iddoc_score'] = df["IDDocMatchScore"].notna().astype(int)
    
    df["face_match_score"] = df["FaceMatchScore"].fillna(0)
    df["id_doc_match_score"] = df["IDDocMatchScore"].fillna(0)
    
    return df[
    [
        "CustomerID",
        "kyc_level_ord",
        "has_face_score",
        "has_iddoc_score",
        "face_match_score",
        "id_doc_match_score",
    ]
]