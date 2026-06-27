import numpy as np
import pandas as pd


def generate_reason_codes_df(
    customer_ids: list | np.ndarray,
    X_explain: pd.DataFrame | None,
    shap_array: np.ndarray | None,
    top_n_reasons: int = 5,
    reason_code_max_rows: int | None = 1000,
    fallback_message: str | None = None,
) -> pd.DataFrame:
    """
    Generates a DataFrame of reason codes for customer IDs based on SHAP values.
    
    Parameters:
    -----------
    customer_ids : array-like
        The full list of CustomerIDs.
    X_explain : pd.DataFrame or None
        The feature DataFrame (head of holdout features up to reason_code_max_rows).
        Can be None if SHAP is not computed or failed.
    shap_array : np.ndarray or None
        The corresponding SHAP values array. Can be None if SHAP is not computed or failed.
    top_n_reasons : int
        Number of top reasons to generate per customer.
    reason_code_max_rows : int or None
        The maximum number of rows for which to calculate reasons. Use None for all rows.
    fallback_message : str or None
        If provided, uses this message for all columns instead of attempting SHAP extraction.
    """
    customer_ids = list(customer_ids)
    reason_rows = []
    
    n_total = len(customer_ids)
    n_explain = n_total if reason_code_max_rows is None else min(n_total, reason_code_max_rows)
    
    if fallback_message is not None or shap_array is None or X_explain is None:
        msg = fallback_message or "N/A (SHAP unavailable)"
        for cid in customer_ids[:n_explain]:
            row = {"CustomerID": cid}
            for i in range(top_n_reasons):
                row[f"reason_{i+1}"] = msg
            reason_rows.append(row)
    else:
        feature_names = X_explain.columns.tolist()
        for idx, cid in enumerate(customer_ids[:n_explain]):
            row_shap = shap_array[idx]
            sorted_indices = np.argsort(np.abs(row_shap))[::-1]
            
            row_reasons = {"CustomerID": cid}
            for rank in range(top_n_reasons):
                if rank < len(sorted_indices):
                    f_idx = sorted_indices[rank]
                    f_name = feature_names[f_idx]
                    f_val = X_explain.iloc[idx, f_idx]
                    f_shap = row_shap[f_idx]
                    row_reasons[f"reason_{rank+1}"] = f"{f_name} (val: {f_val:.2f}, shap: {f_shap:+.4f})"
                else:
                    row_reasons[f"reason_{rank+1}"] = "N/A"
            reason_rows.append(row_reasons)
            
    reasons_df = pd.DataFrame(reason_rows)

    if reason_code_max_rows is not None and n_total > n_explain:
        remaining_cids = customer_ids[n_explain:]
        rem_rows = []
        for cid in remaining_cids:
            row = {"CustomerID": cid}
            for i in range(top_n_reasons):
                row[f"reason_{i+1}"] = "N/A (Bounded by REASON_CODE_MAX_ROWS)"
            rem_rows.append(row)
        rem_df = pd.DataFrame(rem_rows)
        reasons_df = pd.concat([reasons_df, rem_df], ignore_index=True)

    return reasons_df


def select_explaining_pipeline(
    model_name: str,
    final_models: dict,
    ranked_model_names: list[str] | None = None,
):
    explaining_pipeline = final_models.get(model_name)
    explaining_model_name = model_name

    if model_name == "stacking_tuned":
        ranked_model_names = ranked_model_names or list(final_models)
        non_stack_ranked = [
            candidate
            for candidate in ranked_model_names
            if candidate != "stacking_tuned" and candidate in final_models
        ]
        if non_stack_ranked:
            explaining_model_name = non_stack_ranked[0]
            explaining_pipeline = final_models[explaining_model_name]
        else:
            explaining_pipeline = None

    return explaining_model_name, explaining_pipeline


def generate_reason_codes_for_models(
    model_names: list[str],
    final_models: dict,
    customer_ids: list | np.ndarray,
    X_input: pd.DataFrame,
    compute_shap_values_fn,
    ranked_model_names: list[str] | None = None,
    top_n_reasons: int = 5,
    reason_code_max_rows: int | None = 1000,
) -> tuple[pd.DataFrame, dict]:
    reason_frames = []
    shap_artifacts = {}

    for model_name in model_names:
        explaining_model_name, explaining_pipeline = select_explaining_pipeline(
            model_name=model_name,
            final_models=final_models,
            ranked_model_names=ranked_model_names,
        )

        if explaining_pipeline is None or explaining_model_name == "stacking_tuned":
            reasons_model_df = generate_reason_codes_df(
                customer_ids=customer_ids,
                X_explain=None,
                shap_array=None,
                top_n_reasons=top_n_reasons,
                reason_code_max_rows=reason_code_max_rows,
                fallback_message="N/A (Stacking model only; SHAP unavailable)",
            )
        else:
            try:
                X_transformed = explaining_pipeline._transform_features(X_input)
                X_explain = (
                    X_transformed
                    if reason_code_max_rows is None
                    else X_transformed.head(reason_code_max_rows)
                )
                shap_array = compute_shap_values_fn(explaining_pipeline, X_explain)
                reasons_model_df = generate_reason_codes_df(
                    customer_ids=customer_ids,
                    X_explain=X_explain,
                    shap_array=shap_array,
                    top_n_reasons=top_n_reasons,
                    reason_code_max_rows=reason_code_max_rows,
                )
                shap_artifacts[model_name] = {
                    "explaining_model_name": explaining_model_name,
                    "X_explain": X_explain,
                    "shap_array": shap_array,
                }
            except Exception as exc:
                reasons_model_df = generate_reason_codes_df(
                    customer_ids=customer_ids,
                    X_explain=None,
                    shap_array=None,
                    top_n_reasons=top_n_reasons,
                    reason_code_max_rows=reason_code_max_rows,
                    fallback_message=f"Error: SHAP failed ({str(exc)[:50]})",
                )
                shap_artifacts[model_name] = {
                    "explaining_model_name": explaining_model_name,
                    "error": str(exc),
                }

        reasons_model_df.insert(1, "model_name", model_name)
        reasons_model_df.insert(2, "explaining_model_name", explaining_model_name)
        reason_frames.append(reasons_model_df)

    if not reason_frames:
        raise RuntimeError("No reason codes were produced. Check final model availability.")

    return pd.concat(reason_frames, ignore_index=True), shap_artifacts
