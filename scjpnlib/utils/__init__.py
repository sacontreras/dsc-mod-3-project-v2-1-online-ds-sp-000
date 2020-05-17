from IPython.core.display import HTML, Markdown
import pandas as pd
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype
import numpy as np
import matplotlib.pyplot as plt

def yes_no_prompt(prompt):
    display(HTML("<h3>{}</h3>".format(prompt)))
    response = input().strip().lower()
    return response[0] == "y" if len(response) > 0 else False

def analyze_values(df, df_name, normalize_lcase=False, display_is_numeric_str=False, n_display_unique_threshold=150, sort_by_unique_count=False, suppress_header=False):
    l = len(df)
    if not suppress_header:
        display(HTML(f"<h3>{df_name} DataFrame Feature-Value Analysis (out of {l} total observations{', strings lcase normalized' if normalize_lcase else ''})</h3>"))
    
    feats_with_unique_count = {}
    for col in df.columns:
        unique_values = df[col].str.lower().unique() if normalize_lcase and is_string_dtype(df[col]) else df[col].unique()
        n_unique = len(unique_values)
        feats_with_unique_count[col] = (n_unique, unique_values) 
    cols = ['feature', 'dtype', 'n_unique', 'n_unique_ratio', 'unique_vals', 'p_cat', 'n_null', 'n_null_ratio', 'null_index']
    if display_is_numeric_str:
        cols.append('n_is_numeric_str')
        cols.append('is_numeric_str_index')
    df_unique_analysis = pd.DataFrame(columns=cols)
    feats = [t[0] for t in sorted(feats_with_unique_count.items(), key=lambda t: t[1][0])] if sort_by_unique_count else list(df.columns)
    for feat in feats:
        unique_values = feats_with_unique_count[feat][1]
        n_unique = feats_with_unique_count[feat][0]
        n_unique_ratio = n_unique/l
        p_cat = 1 - n_unique_ratio
        n_null = df[feat].isnull().sum()
        n_null_ratio = n_null/l
        p_null = 1 - n_null_ratio
        null_index = df[df[feat].isnull()==True].index if n_null > 0 else None
        data = [{
            'feature': feat, 
            'dtype': df[feat].dtypes, 
            'n_unique': n_unique, 
            'n_unique_ratio': n_unique_ratio, 
            'unique_vals': unique_values,
            'p_cat': round(p_cat,4)*100, 
            'n_null': n_null,
            'n_null_ratio': n_null_ratio,
            'null_index': null_index
        }]

        if display_is_numeric_str:
            str_cat_feat_numeric_string_vals = df[feat].loc[df[feat].str.isnumeric()==True] if df[feat].dtype==object else None
            n_is_numeric_str = len(str_cat_feat_numeric_string_vals) if str_cat_feat_numeric_string_vals is not None else None
            is_numeric_str_index = str_cat_feat_numeric_string_vals.index if n_is_numeric_str is not None and n_is_numeric_str>0 else None
            data[0]['n_is_numeric_str'] = n_is_numeric_str
            data[0]['is_numeric_str_index'] = is_numeric_str_index

        df_unique_analysis = df_unique_analysis.append(data, ignore_index=True, sort=False)
            
    return df_unique_analysis

def get_outliers(df, feat, iqr_factor=1.5):
    q1, q3 = np.percentile(df.sort_values(by=feat)[feat], [25, 75])
    delta = q3 - q1
    iqr_q_lb = q1 - (iqr_factor * delta) 
    iqr_q_ub = q3 + (iqr_factor * delta)
    return (df.query(f"{feat}<{iqr_q_lb} or {feat}>{iqr_q_ub}").index, iqr_q_lb, iqr_q_ub, q1, q3, iqr_factor)

def analyze_outliers(df, df_name, iqr_factor=1.5, feats=None, display_plots=True, plot_edge=2):
    l = len(df)
    display(HTML(f"<h3>{df_name} DataFrame Outlier (iqr_factor=={iqr_factor}) Analysis</h3>"))

    if feats is None:
        feats = list(df.columns)

    if display_plots:
        n_feats = len(feats)
        r_w = 4*plot_edge if n_feats > plot_edge else (n_feats*4 if n_feats > 1 else plot_edge)
        r_h = plot_edge if n_feats > 4 else (plot_edge if n_feats > 1 else plot_edge)
        c_n = 4 if n_feats > 4 else n_feats
        r_n = n_feats/c_n
        r_n = int(r_n) + (1 if n_feats > 4 and r_n % int(r_n) != 0 else 0)        
        fig = plt.figure(figsize=(r_w, r_h*r_n))

    df_outlier_analysis = pd.DataFrame(columns=['feature', 'dtype', 'q1', 'q3', 'IQR_lower_bound', 'IQR_upper_bound', 'n_outliers', 'n_outliers_ratio', 'outliers_index'])
    for idx, feat in enumerate(feats):
        if is_numeric_dtype(df[feat]):
            feat_outliers_index, iqr_q_lb, iqr_q_ub, q1, q3, iqr_factor = get_outliers(df, feat, iqr_factor=iqr_factor)
            n_outliers = len(feat_outliers_index)
            n_outliers_ratio = n_outliers/l
            data = [{
                'feature': feat, 
                'dtype': df[feat].dtype,
                'q1': q1,
                'q3': q3,
                'IQR_lower_bound': iqr_q_lb,
                'IQR_upper_bound': iqr_q_ub,
                'n_outliers': n_outliers,
                'n_outliers_ratio': n_outliers_ratio,
                'outliers_index': feat_outliers_index
            }]
            df_outlier_analysis = df_outlier_analysis.append(data, ignore_index=True, sort=False)
            if display_plots:
                ax = fig.add_subplot(r_n, c_n, idx+1)
                df[[feat]].boxplot()
        else:
            display(HTML(f"<h4>*** WARNING: outlier analysis is not applicable for <i>{feat}</i> since it is not numeric ({df[feat].dtype}) ***</h4>"))

    if display_plots:
        fig.tight_layout()
        plt.show()
            
    return df_outlier_analysis

def analyze_numeric_strings(df, df_name):
    l = len(df)
    display(HTML(f"<h3>{df_name} DataFrame Feature-Numeric-String Value Analysis (out of {l} total observations)</h3>"))

    df_analysis = analyze_values(df, df_name, display_is_numeric_str=True, suppress_header=True)
    df_analysis = df_analysis.loc[df_analysis['dtype']==object][['feature', 'n_unique', 'n_is_numeric_str', 'is_numeric_str_index']]
    display(HTML(df_analysis.to_html()))
    display(HTML("<p><br>"))

    str_feat_unique_numeric_string_vals = {}
    for str_feat in list(df_analysis.feature.unique()):
        feat_numeric_index = df_analysis.loc[df_analysis['feature']==str_feat]['is_numeric_str_index'].values[0]
        if feat_numeric_index is not None:
            unique_numeric_strings = df.loc[feat_numeric_index][str_feat].unique()
            str_feat_unique_numeric_string_vals[str_feat] = df.loc[feat_numeric_index][str_feat].unique()
            display(HTML(f"in {df_name}, string-type feature <b>{str_feat}</b> has unique numeric values: {unique_numeric_strings}"))
    
    display(HTML("<p><br><br>"))

    return (str_feat_unique_numeric_string_vals, df_analysis)