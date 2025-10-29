import pandas as pd 
from lazypredict.Supervised import LazyClassifier
from sklearn.model_selection import train_test_split

def write_cols_to_excel(cols_csv_path, excel_path):
    cols_csv = pd.read_csv(cols_csv_path, names=["NAME", "LABEL"])
    
    # write cols_csv to cols.xlsx 
    with pd.ExcelWriter(excel_path, engine="openpyxl", mode="w") as writer:
        cols_csv.to_excel(writer, index=False)
       
def one_hot_encode(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """One hot encode a column in a dataframe."""
    one_hot = pd.get_dummies(df[column], prefix=column, dummy_na=True)
    df = df.drop(column, axis=1)
    df = df.join(one_hot)
    return df

def clean_hstexas_excel():
    df = pd.read_csv("data/hstexas_full.csv")
    # drop columns if the name does not start with 'qn' 
    df = df[[col for col in df.columns if not col.startswith('qn')]]
    # drop columns with more than 50% na values
    df = df.dropna(subset=['q35', 'q36', 'q40'], how='any')
    df = df.dropna(axis=1, thresh=len(df)*0.5)
    df.to_csv("data/cleaned_hstexas_full.csv", index=False)
     

def analyze_unique_values(df):
    for col in df.columns:
        unique_values = df[col].value_counts(dropna=False)
        with open("output/hstexas_unique_values.txt", "a") as f:
            f.write(str(unique_values)+"\n\n")

def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns based on variable_names.xlsx file."""
    columns_excel = "variable_names.xlsx"
    column_info = pd.read_excel(columns_excel)
    cols_to_keep = column_info[column_info['keep'] == 'keep']['name'].to_list()
    print("keeping {} columns".format(len(cols_to_keep)))
    # filter df to only keep cols in cols_to_keep
    df = df[[col for col in df.columns if col in cols_to_keep]]
    df = df[[col for col in df.columns if col.startswith('q')]]

    # rename cols if column_info["newname"] is not empty for col in df.columns
    for col in df.columns:
        newname = column_info[column_info['name'] == col]['newname'].values
        if len(newname) > 0 and newname != '' and pd.notna(newname[0]):
            df = df.rename(columns={col: newname[0]})
    return df



def prepare_data_for_causal_analysis():
    """Prepare data for causal analysis by creating target variables."""
    df = pd.read_csv("data/cleaned_hstexas_full.csv")
    
    # Create target variables from survey questions
    # q35: Ever used electronic vapor product
    # q36: Current electronic vapor use
    if 'q35' in df.columns:
        df['ever_vaped'] = (df['q35'] == 1).astype(int)
    
    if 'q36' in df.columns:
        df['currently_vape'] = (df['q36'] == 1).astype(int)
        
    # Create quit smoking variable if we have tobacco data
    if 'q32' in df.columns and 'q33' in df.columns:
        # Ever smoked but not currently smoking
        df['quit_smoking'] = ((df['q32'] == 1) & (df['q33'] != 1)).astype(int)
    
    # Save processed data
    df.to_csv("data/processed_for_causal.csv", index=False)
    return df

def run_predictive_analysis():
    """Run the original predictive analysis."""
    df = prepare_data_for_causal_analysis()
    
    # Check if we have target variables
    target_variables = ["ever_vaped"]
    available_targets = [tv for tv in target_variables if tv in df.columns]
    
    if not available_targets:
        print("No target variables found. Creating from survey questions...")
        return df
    
    # Drop columns with more than 5% na values and then drop rows with any na values
    df = df.dropna(axis=1, thresh=len(df)*0.95)
    df = df.dropna(axis=0, how='any')
    
    print("after dropping na there are {} rows and {} columns".format(len(df), len(df.columns)))  
    
    # Prepare features and target
    feature_cols = [col for col in df.columns if col not in available_targets]
    X = df[feature_cols]
    y = df['ever_vaped'] if 'ever_vaped' in df.columns else None
    
    if y is not None:
        print(f"Target variable distribution:")
        print(y.value_counts())
        print(f"Vaping rate: {y.mean():.2%}")
        
        # Optional: Run LazyClassifier if needed
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
        # clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
        # models, predictions = clf.fit(X_train, X_test, y_train, y_test)
        # print(models)
    
    return df

if __name__ == "__main__":
    df = run_predictive_analysis()