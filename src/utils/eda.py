import pandas as pd
import matplotlib.pyplot as plt

def dataset_overview(datasets: dict) -> pd.DataFrame:
    rows = []
    for name, df in datasets.items():
        rows.append({
            "table": name,
            "rows": len(df),
            "columns": df.shape[1],
            "duplicated_rows": df.duplicated().sum()
        })
        
    return pd.DataFrame(rows)

def missing_report(df: pd.DataFrame) -> pd.DataFrame:
    report = pd.DataFrame({
        "missing_count": df.isna().sum(),
        "missing_pct": df.isna().mean() * 100,
        "dtype": df.dtypes.astype(str),
        "nunique": df.nunique(dropna=True)
    })
    
    return report.sort_values("missing_pct", ascending=False)

def plot_missing_top(df: pd.DataFrame, table_name: str, top_n: int = 20):
    missing = missing_report(df).query("missing_pct > 0").head(top_n)

    if missing.empty:
        print(f"No missing values in {table_name}")
        return

    ax = missing["missing_pct"].sort_values().plot(kind="barh", figsize=(8, 5))
    ax.set_title(f"Top missing columns - {table_name}")
    ax.set_xlabel("Missing percentage (%)")
    ax.set_ylabel("Column")
    plt.tight_layout()
    
def plot_value_counts(
    df: pd.DataFrame,
    column: str,
    title: str = None,
    normalize: bool = False,
    top_n: int = 20
):
    vc = df[column].value_counts(normalize=normalize).head(top_n)

    ax = vc.sort_values().plot(kind="barh", figsize=(8, 5))
    ax.set_title(title or f"Distribution of {column}")
    ax.set_xlabel("Proportion" if normalize else "Count")
    ax.set_ylabel(column)
    plt.tight_layout()
    
def plot_numeric_distribution(
    df: pd.DataFrame,
    column: str,
    title: str = None,
    bins: int = 50
):
    ax = df[column].dropna().plot(kind="hist", bins=bins, figsize=(8, 5))
    ax.set_title(title or f"Distribution of {column}")
    ax.set_xlabel(column)
    plt.tight_layout()
    
def train_vs_holdout(dts_train: pd.DataFrame, dts_holdout: pd.DataFrame, col: str):
    import matplotlib.pyplot as plt

    print("Count:")
    plt.figure(figsize=(8,5))

    dts_train[col].hist(
        bins=30,
        alpha=0.5,
        label="train"
    )

    dts_holdout[col].hist(
        bins=30,
        alpha=0.5,
        label="holdout"
    )

    plt.legend()
    plt.ylabel("Count")
    plt.title(col)
    plt.show()
    
    print("Density:")
    plt.figure(figsize=(8,5))

    dts_train[col].hist(
        bins=30,
        density=True,
        alpha=0.5,
        label="train"
    )

    dts_holdout[col].hist(
        bins=30,
        density=True,
        alpha=0.5,
        label="holdout"
    )

    plt.legend()
    plt.title(col)
    plt.ylabel("Density")

    plt.show()
    
    print("Boxplot:")
    
    compare_df = pd.concat([
        pd.DataFrame({
            col: dts_train[col],
            "dataset": "train"
        }),
        pd.DataFrame({
            col: dts_holdout[col],
            "dataset": "holdout"
        })
    ])

    compare_df.boxplot(
        column=col,
        by="dataset"
    )

    plt.suptitle("")
    plt.show()
    
    print("Descriptive Statistic:")
    print(    
        pd.concat([
            dts_train["MonthsInService"].describe(),
            dts_holdout["MonthsInService"].describe()
        ], axis=1)
    )
    
    
    print("KS test:")
    from scipy.stats import ks_2samp

    print(    
        ks_2samp(
            dts_train["MonthsInService"],
            dts_holdout["MonthsInService"]
        )
    )