from IPython.core.display import HTML, Markdown
import pandas as pd
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import itertools as it
from .skl_transformers import SimpleValueTransformer, LambdaTransformer
import scipy.stats as st
import math
from itertools import combinations
import json
import hashlib
import re
import inspect
from copy import copy, deepcopy
import pprint

sns.set_style('whitegrid')




def yes_no_prompt(prompt):
    display(HTML("<h3>{}</h3>".format(prompt)))
    response = input().strip().lower()
    return response[0] == "y" if len(response) > 0 else False


def helper__HTML_tabs(n_tabs, nbsp_count=4):
    return ('&nbsp;'*nbsp_count)*n_tabs


def analyze_values(
    df, 
    df_name, 
    standard_options_kargs={'sort_unique_vals':False,'normalize_lcase':False,'compute_non_alphanumeric_str':False,'sort_by_unique_count':False},
    compute_top__kargs=None,
    compute_duplicates__kargs=None,
    suppress_output=False
):
    """
        standard_options_kargs: 
            options for controlling columns/output of: 
                sorting unique-values analysis
                normalizing to lower-case when before identifying unique values
                computing whether string-type values are purely numeric

            keys:
                'sort_unique_vals': boolean; True to sort unique values ascending
                'normalize_lcase': boolean; True to convert string-type featuers to lower-case PRIOR to collating all unique values
                'compute_non_alphanumeric_str': boolean; True to compute index of all string-stype features containing substrings that match the regex '[^a-zA-Z0-9]+' - i.e. any string that is not purely alphanumeric
                'sort_by_unique_count': boolean; True to sort the order output of 'feature' column of analysis df by unique count, descending
                'hide_cols': boolean; True to hide columns display of standard options (unique vals, null-count, probability of categorical, etc.)
        
        compute_top__kargs: if not None, analysis will compute the features constituting the top occurences according to the semantic specified in the 'strategy' key
            keys:
                'strategy': 'literal' or 'percentile'
                'value': int
                    if 'strategy' is 'literal', then analysis will locate the top <'value'> values - e.g. for 'value'==10, analysis locates the top 10 occurring values
                    if 'strategy' is 'percentile', then analysis will locate the values comprising the top <'value'> percent (count) of the population
                        e.g. for 'value'==90, analysis locates values comprising 90% of (count of) the population
                        location index (length) will not surpass this threshold UNLESS a SINGLE value comprises at least that portion of the population
                            e.g. if 'value'==90 and a SINGLE value comprises, for example, 98% of the population, then analysis will compute the index for all occurrences
                                of that index comprising 98% of the population
                'display_bar_plots': boolean; True to display bar plots of top values - note that this takes a while to execute, so proceed accordingly; disabled by default

        compute_duplicates__kargs:
            use this to conduct duplicates-analys - note that there are no additional columns added to the analysis data frame
                instead, this will conduct iterate all pair-wise combinations of features of the same data type and look for duplicate values
                it will minimally output the ration of duplicates found (on the above basis) in the original DF

            options for computing duplicate feature values on a combinatorial, pair-wise basis
                this option will produce a list of indices where values are identical for the feature-pair considered
                setting this argument (dict) to None will disable duplicates-analysis and the duplicates_entries part of the tuple returned by this function will be None
                    otherwise the tuple will duplicates_entries will be a list of (df_identical_combo_vals, feats_combo, dt) - i.e. 
                        the DF corresponding to index (in the original DF) where values are identical for the feature-pair considered
                        the feature-pair as a tuple
                        the data type of the feature-pair

            keys:
                'display_heads': boolean; True to display the resulting DF for each feature-pair


        return value:
            a tuple consisting of:
                the DF containing analysis results
                    columns that are always present:
                        'feature': name of the feature
                        'dt': (pandas) data type of the feature

                    columns present when standard_options_kargs is not None and standard_options_kargs['hide_cols'] is not True (or is undefined):
                        'n_unique': count of unique values
                        'n_unique_ratio': ratio of n_unique to total count of observations; used in computation of 'p_cat'
                        'unique_vals': the set of unique values
                        'p_cat': computed "probability" that this feature is categorical based entirely on 1 - n_unique_ratio; use this as a guide only
                        'n_null': count of null (nan or missing) values 
                        'n_null_ratio': ratio of n_null to total count of observations
                        'null_index': the index of all values that are nan or missing
                    
                    columns present when compute_top__kargs is not None:
                        when compute_top__kargs['strategy']=='literal':
                            'n_top_<literal_value>': the count of the top <literal_value> values
                                note that '<literal_value>' will be replaced by compute_top__kargs['value']
                            'val_count_top_<literal_value>': this will contain a dictionary keyed by the feature-values occurring in the top-values;
                                its dict-value will contain the tuple (count, ratio-of-population) 
                            'n_unique_top_<literal_value>': the count of unique <literal_value> values - this necessarily is equal to literal_value unless there a fewer unique values
                                note that '<literal_value>' will be replaced by compute_top__kargs['value']
                            'index_top_<literal_value>': the index of the top <literal_value> values occurring in the original DF
                                note that '<literal_value>' will be replaced by compute_top__kargs['value']
                        
                        when compute_top__kargs is not None and compute_top__kargs['strategy']=='percentile'
                            'n_top_<percentile_value>': the count of the top <percentile_value> values
                                note that '<percentile_value>' will be replaced by the string concatenation of compute_top__kargs['value'] and '_percent';
                            'val_count_top_<percentile_value>': this will contain a dictionary keyed by the feature-values occurring in the top-values;
                                its dict-value will contain the tuple (count, ratio-of-population) 
                            'n_unique_top_<percentile_value>': the count of unique values in the set comprising the top percentile values
                                note that '<percentile_value>' will be replaced by compute_top__kargs['value']
                            'index_top_<percentile_value>': the index of the values comprising the top <percentile_value> occurrences in the original DF
                                note that '<literal_value>' will be replaced by compute_top__kargs['value']
                    
                        always present:
                            'n_pop_rep': the ratio of the population represented by the top values

    """

    sort_unique_vals = standard_options_kargs is not None and 'sort_unique_vals' in standard_options_kargs and standard_options_kargs['sort_unique_vals']
    normalize_lcase = standard_options_kargs is not None and 'normalize_lcase' in standard_options_kargs and standard_options_kargs['normalize_lcase']
    compute_non_alphanumeric_str = standard_options_kargs is not None and 'compute_non_alphanumeric_str' in standard_options_kargs and standard_options_kargs['compute_non_alphanumeric_str']
    sort_by_unique_count = standard_options_kargs is not None and 'sort_by_unique_count' in standard_options_kargs and standard_options_kargs['sort_by_unique_count']
    hide_standard_cols = standard_options_kargs is None or 'hide_cols' in standard_options_kargs and standard_options_kargs['hide_cols']

    l = len(df)
    if not suppress_output:
        display(HTML(f"<h3>{df_name} Feature-Value Analysis (out of {l} total observations{', strings lcase normalized' if normalize_lcase else ''})</h3>"))
    
    feats_with_unique_count = {}
    for col in df.columns:
        if is_string_dtype(df[col]):
            unique_values = df[col].str.lower().unique() if normalize_lcase else df[col].astype(str).unique()
        else:
            unique_values = df[col].unique()
        if sort_unique_vals:
            unique_values = sorted(list(unique_values))
        n_unique = len(unique_values)
        feats_with_unique_count[col] = (n_unique, unique_values)

    cols = ['feature', 'dtype', 'n_unique', 'unique_vals']
    if not hide_standard_cols:
        cols.extend(['n_unique_ratio', 'p_cat', 'n_null', 'n_null_ratio', 'null_index'])

    compute_top = compute_top__kargs is not None and compute_top__kargs['strategy'] in ['percentile', 'literal'] and compute_top__kargs['value'] is not None
    compute_top__value = None
    if compute_top:
        compute_top_literal = compute_top__kargs['strategy']=='literal'
        compute_top__value = compute_top__kargs['value']

        s_label_suffix = f"{compute_top__value}" if compute_top_literal else f"{compute_top__value}_percent"
        s_n_top__label = f"n_top_{s_label_suffix}"
        cols.append(s_n_top__label)
        s_pop_representation_label = f"n_pop_rep"
        cols.append(s_pop_representation_label)
        s_val_count_top__label = f"val_count_top_{s_label_suffix}"
        cols.append(s_val_count_top__label)
        s_n_unique_top__label = f"n_unique_top_{s_label_suffix}"
        cols.append(s_n_unique_top__label)
        s_index_top__label = f"index_top_{s_label_suffix}"
        cols.append(s_index_top__label)

    if compute_non_alphanumeric_str:
        cols.append('n_non_alphanumeric_str')
        cols.append('non_alphanumeric_str_index')

    df_analysis = pd.DataFrame(columns=cols)
    feats = [t[0] for t in sorted(feats_with_unique_count.items(), key=lambda t: t[1][0])] if sort_by_unique_count else list(df.columns)

    for feat in feats:
        unique_values = feats_with_unique_count[feat][1]
        n_unique = feats_with_unique_count[feat][0]
        data = [{
            'feature': feat, 
            'dtype': df[feat].dtypes,
            'n_unique': n_unique,
            'unique_vals': unique_values
        }]

        if not hide_standard_cols:
            n_unique_ratio = n_unique/l
            p_cat = 1 - n_unique_ratio
            n_null = df[feat].isnull().sum()
            n_null_ratio = n_null/l
            p_null = 1 - n_null_ratio
            null_index = df[df[feat].isnull()==True].index if n_null > 0 else None
            data[0]['n_unique_ratio'] = n_unique_ratio
            data[0]['p_cat'] = round(p_cat,4)*100
            data[0]['n_null'] = n_null
            data[0]['n_null_ratio'] = n_null_ratio
            data[0]['null_index'] = null_index

        if compute_top:
            feat_val_counts = df[feat].value_counts()
            count_top = 0
            count_top_thresh = compute_top__value if compute_top_literal else int(l*(compute_top__value/100))
            val_count_top = {}
            for i, feat_val_count in enumerate(feat_val_counts):
                new_count_top = count_top + (1 if compute_top_literal else feat_val_count)
                if len(val_count_top)==0 or new_count_top <= count_top_thresh:
                    count_top = new_count_top
                    # feat_val = feat_val_counts.index[i]
                    feat_val = feat_val_counts.index.values[i]
                    val_count_top[feat_val] = (feat_val_count, feat_val_count/l)
                else:
                    break
            top_found_at = np.where(np.isin(df[feat].values, list(val_count_top.keys())))[0]
            data[0][s_n_top__label] = len(top_found_at)
            data[0][s_pop_representation_label] = len(top_found_at)/l
            data[0][s_val_count_top__label] = val_count_top
            data[0][s_n_unique_top__label] = len(df.iloc[top_found_at][feat].unique())
            data[0][s_index_top__label] = df[df[feat].isin(list(val_count_top.keys()))].index

        if compute_non_alphanumeric_str:
            str_cat_feat_non_alphanumeric_vals = df.loc[df[feat].str.contains('[^a-zA-Z0-9]+',regex=True)==True] if df[feat].dtype==object else None
            n_non_alphanumeric_str = len(str_cat_feat_non_alphanumeric_vals) if str_cat_feat_non_alphanumeric_vals is not None else None
            non_alphanumeric_str_index = str_cat_feat_non_alphanumeric_vals.index if n_non_alphanumeric_str is not None and n_non_alphanumeric_str>0 else None
            data[0]['n_non_alphanumeric_str'] = n_non_alphanumeric_str
            data[0]['non_alphanumeric_str_index'] = non_alphanumeric_str_index

        df_analysis = df_analysis.append(data, ignore_index=True, sort=False)

    if not suppress_output:
        display(HTML(df_analysis.to_html(notebook=True, justify='left')))

    # duplicates analysis
    compute_duplicates = compute_duplicates__kargs is not None
    display_duplicates_heads = compute_duplicates and 'display_heads' in compute_duplicates__kargs and compute_duplicates__kargs['display_heads']
    duplicates_entries = []
    if compute_duplicates and len(df.columns)>1:
        display(HTML(f"<p><br><h4>{df_name} Duplicates-Analysis:</h4>"))
        for dt in set(df.dtypes):
            df_of_type_dt = df.select_dtypes(dt)
            feats_of_same_type_combos = set(it.combinations(list(df_of_type_dt.columns), 2))
            for feats_combo in feats_of_same_type_combos:
                df_identical_combo_vals = df[df[feats_combo[0]]==df[feats_combo[1]]]
                duplicates_entries.append((df_identical_combo_vals, feats_combo, dt))
                n_dups = len(df_identical_combo_vals)
                display(HTML(f"combo ({dt}): <b>{feats_combo}</b>; duplicates-count: {n_dups}/{l} ({round((n_dups/l)*100, 4)}%)"))
                if display_duplicates_heads:
                    display(HTML(f"<h6>Head:</h6>"))
                    display(HTML(df_identical_combo_vals.head().to_html(notebook=True)))
                    display(HTML("<p><br><br>"))

    display_top_bar_plots = compute_top__value is not None and 'display_bar_plots' in compute_top__kargs and compute_top__kargs['display_bar_plots']
    if not suppress_output and display_top_bar_plots:
        display(HTML(f"<p><br><h4>{df_name} Plots (top {compute_top__value}{'%' if not compute_top_literal else ''}):</h4>"))
        plot_edge = 6
        n_feats = len(feats)
        r_w = 4*plot_edge if n_feats > plot_edge else (n_feats*4 if n_feats > 1 else plot_edge)
        r_h = plot_edge if n_feats > 4 else (plot_edge if n_feats > 1 else plot_edge)
        c_n = 4 if n_feats > 4 else n_feats
        r_n = n_feats/c_n
        r_n = int(r_n) + (1 if n_feats > 4 and r_n % int(r_n) != 0 else 0)        
        fig = plt.figure(figsize=(r_w, r_h*r_n))

        for idx, feat in enumerate(feats):
            ax = fig.add_subplot(r_n, c_n, idx+1)
            # df[feat].value_counts()[:count_top_thresh].plot(kind='bar')
            sns.distplot(df[feat].value_counts()[:count_top_thresh], bins=count_top_thresh)
            ax.set_title(feat)

        fig.tight_layout()
        plt.show()
            
    return df_analysis, duplicates_entries


def helper__display_unique_vals_from_analysis(df_analysis, max_unique_display=100):
    for i, row in df_analysis.iterrows():
        unique_vals = row['unique_vals']
        n_unique = len(unique_vals)
        display(HTML(f"<b>{row['feature']}</b> unique vals: {unique_vals if n_unique<=max_unique_display else 'count=='+str(n_unique)}"))


def helper__display_complement_from_duplicates_entries(duplicates_entries, df_complement_in, n_show=10):
    for duplicates_entry in duplicates_entries:
        duplicates_df = duplicates_entry[0]
        feat_combo = duplicates_entry[1]
        display(HTML(f"first {n_show} rows where <b>{feat_combo}</b> differ:"))
        display(HTML(df_complement_in.loc[~df_complement_in.index.isin(duplicates_df.index)][[feat_combo[0], feat_combo[1]]].head(n_show).to_html()))
        display(HTML(f"<p><br>"))

def get_outliers(df, feat, iqr_factor=1.5):
    q1, q3 = np.nanpercentile(df.sort_values(by=feat)[feat], [25, 75])
    delta = q3 - q1
    iqr_q_lb = q1 - (iqr_factor * delta) 
    iqr_q_ub = q3 + (iqr_factor * delta)
    return (df.query(f"{feat}<{iqr_q_lb} or {feat}>{iqr_q_ub}").index, iqr_q_lb, iqr_q_ub, q1, q3, iqr_factor)
    # print(f"get_outliers('{feat}'): iqr_q_lb=={iqr_q_lb}, iqr_q_ub=={iqr_q_ub}")
    # return (df.query(f"{feat}.notnull() and ({feat}<{iqr_q_lb} or {feat}>{iqr_q_ub})", engine='python').index, iqr_q_lb, iqr_q_ub, q1, q3, iqr_factor)


def analyze_value_detailed(df, df_name, feat, top_percentile=75, suppress_values=False, suppress_output=False):
    l = len(df)
    df_analysis, _ = analyze_values(
        df[[feat]], 
        f"{df_name} <i>{feat}</i> Detailed Value Analysis", 
        standard_options_kargs={'sort_unique_vals':True, 'hide_cols':True},
        compute_top__kargs={'strategy':'percentile', 'value':top_percentile},
        suppress_output=True
    )
    df_feat_value_analysis = df_analysis.loc[df_analysis['feature']==feat]
    unique_vals = df[feat].unique()
    n_unique = len(unique_vals)
    n_unique_top_percentile = df_analysis[f'n_unique_top_{top_percentile}_percent'].values[0]
    n_pop_rep = df_analysis['n_pop_rep'].values[0]
    n_top_percentile = df_analysis[f'n_top_{top_percentile}_percent'].values[0]
    index_top_percentile = df_analysis[f'index_top_{top_percentile}_percent'].values[0]
    val_count_top_percentile = df_analysis[f'val_count_top_{top_percentile}_percent'].values[0]

    reduction = abs(1 - 1/(n_unique_top_percentile/n_unique))

    if not suppress_output:
        display(HTML(f"<b>{n_unique_top_percentile} unique value(s)</b> (out of {n_unique} unique, categories-magnitude reduction: {round(reduction*100,2)}%) of <b><i>{feat}</i> constitute <u>{round(n_pop_rep*100,4)}%</u> of the total number of observations</b> ({n_top_percentile} out of {l})"))
    
    if not suppress_output and not suppress_values:
        display(HTML("values are:"))
    for val, val_count_data in val_count_top_percentile.items(): # yields: val, (count, ratio-of-population)
        if not suppress_output and not suppress_values:
            display(HTML(f"{helper__HTML_tabs(1)}{val}{': '+str(round(val_count_data[1]*100,2))+'%' if len(val_count_top_percentile)>1 else ''}"))

    return index_top_percentile, val_count_top_percentile, reduction

def analyze_value_detailed__top_n(df, df_name, feat, top_n, suppress_values=False, suppress_output=False):
    l = len(df)
    df_analysis, _ = analyze_values(
        df[[feat]], 
        f"{df_name} <i>{feat}</i> Detailed Value Analysis", 
        standard_options_kargs={'sort_unique_vals':True, 'hide_cols':True},
        compute_top__kargs={'strategy':'literal', 'value':top_n},
        suppress_output=True
    )
    df_feat_value_analysis = df_analysis.loc[df_analysis['feature']==feat]
    suffix = f"top_{top_n}"
    unique_vals = df[feat].unique()
    n_unique = len(unique_vals)
    n_unique_top_n= df_analysis[f'n_unique_{suffix}'].values[0]
    n_pop_rep = df_analysis['n_pop_rep'].values[0]
    n_top_percentile = df_analysis[f'n_{suffix}'].values[0]
    index_top_n = df_analysis[f'index_{suffix}'].values[0]
    val_count_top_n = df_analysis[f'val_count_{suffix}'].values[0]

    reduction = abs(1 - 1/(n_unique_top_n/n_unique))

    if not suppress_output:
        display(HTML(f"<b>{n_unique_top_n} unique value(s)</b> (out of {n_unique} unique, categories-magnitude reduction: {round(reduction*100,2)}%) of <b><i>{feat}</i> constitute <u>{round(n_pop_rep*100,4)}%</u> of the total number of observations</b> ({n_top_percentile} out of {l})"))
    
    if not suppress_output and not suppress_values:
        display(HTML("values are:"))
    for val, val_count_data in val_count_top_n.items(): # yields: val, (count, ratio-of-population)
        if not suppress_output and not suppress_values:
            display(HTML(f"{helper__HTML_tabs(1)}{val}{': '+str(round(val_count_data[1]*100,2))+'%' if len(val_count_top_n)>1 else ''}"))

    return index_top_n, val_count_top_n, reduction


def analyze_outliers(df, df_name, iqr_factor=1.5, feats=None, display_plots=True, plot_edge=2, suppress_output=False):
    l = len(df)
    if not suppress_output:
        display(HTML(f"<h3>{df_name} DataFrame Outlier (iqr_factor=={iqr_factor}) Analysis (out of {l} total observations)</h3>"))

    if feats is None:
        feats = list(df.columns)

    if not suppress_output and display_plots:
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
            if not suppress_output and display_plots:
                ax = fig.add_subplot(r_n, c_n, idx+1)
                df[[feat]].boxplot()
        else:
            if not suppress_output:
                display(HTML(f"<h4>*** WARNING: outlier analysis is not applicable for <i>{feat}</i> since it is not numeric ({df[feat].dtype}) ***</h4>"))

    if display_plots and not suppress_output:
        fig.tight_layout()
        plt.show()
            
    return df_outlier_analysis


def replace_outliers_index(df, feat, outliers_index, map_to_value):
    df_outliers = df.loc[outliers_index]
    unique_outlier_vals = df_outliers[feat].unique()

    replace_feat_outlier_rules = []
    for unique_outlier_val in unique_outlier_vals:
        replace_feat_outlier_rules.append({
            'missing_values': unique_outlier_val,
            'strategy': 'constant', 
            'fill_value': map_to_value
        })
    replace_outliers_rules = {feat: replace_feat_outlier_rules}
    svt_outliers = SimpleValueTransformer(replace_outliers_rules)
    df_outliers_replaced_by_index = svt_outliers.fit_transform(df_outliers) # apply only to outliers so we don't have to deal with nans
    df_outliers_replaced = df.copy()
    df_outliers_replaced.loc[df_outliers_replaced_by_index.index] = df_outliers_replaced_by_index

    return (df_outliers_replaced, replace_outliers_rules)

def replace_outliers(df, feat, map_to_value):
    df_outliers = analyze_outliers(df[[feat]], "", display_plots=False, suppress_output=True)
    df_feat_outlier_analysis = df_outliers.loc[df_outliers['feature']==feat]
    outliers_index = df_feat_outlier_analysis['outliers_index'].values[0]
    return replace_outliers_index(df, feat, outliers_index, map_to_value)[0]

def impute_TO_nan(df, feat, val_to_replace_with_nan):
    df_feat_val_as_nan = df.copy()
    df_feat_val_as_nan[feat] = df_feat_val_as_nan[feat].replace(val_to_replace_with_nan, np.NaN)
    return df_feat_val_as_nan

def find_and_impute(df, feat, to_replace=r"[^a-zA-Z0-9]", replace_with_val=""):
    df_copy = df.copy()
    df_copy[feat] = df_copy[feat].replace(to_replace=to_replace, value=replace_with_val, regex=True)
    return df_copy

def impute_TO_lcase(df, feat):
    #exclude nulls since this may not have been done yet
    df_nulls_excluded = df[df[feat].isnull()==False]

    df_feat_with_ucase_vals = df_nulls_excluded.loc[df_nulls_excluded[feat].str.contains('[A-Z]+',regex=True)==True]
    if len(df_feat_with_ucase_vals) > 0:
        impute_lcase_rules = {}
        impute_lcase_rules[feat] = lambda v: v.lower()
        lt_lcase_cat_predictors = LambdaTransformer(impute_lcase_rules)
        df_lcased = lt_lcase_cat_predictors.fit_transform(df_feat_with_ucase_vals)
        return pd.concat([df_lcased, df[~df.index.isin(df_lcased.index)]], axis=0, ignore_index=False)
    else:
        return df

def analyze_outliers_detailed(
    df, 
    df_name, 
    feat, 
    top_percentile=75, 
    outlier_ratio_reduction_threshold=.10,
    suppress_replacement_strat_analysis=False,
    suppress_values_display=True,
    suppress_candidate_head=True,
    suppress_output=False
):
    l = len(df)
    df_analysis, _ = analyze_values(
        df[[feat]], 
        f"{df_name} <i>{feat}</i> Detailed Outlier Analysis", 
        standard_options_kargs={'sort_unique_vals':True, 'hide_cols':True},
        compute_top__kargs={'strategy':'percentile', 'value':top_percentile},
        suppress_output=True
    )
    df_feat_value_analysis = df_analysis.loc[df_analysis['feature']==feat]
    unique_vals = df[feat].unique()
    n_unique = len(unique_vals)

    df_outliers = analyze_outliers(df[[feat]], f"{df_name} <i>{feat}</i>",  plot_edge=3, suppress_output=suppress_output)
    df_feat_outlier_analysis = df_outliers.loc[df_outliers['feature']==feat]
    
    q1 = df_feat_outlier_analysis['q1'].values[0]
    q3 = df_feat_outlier_analysis['q3'].values[0]
    n_outliers = df_feat_outlier_analysis['n_outliers'].values[0]
    n_outliers_ratio = df_feat_outlier_analysis['n_outliers_ratio'].values[0]
    outliers_index = df_feat_outlier_analysis['outliers_index'].values[0]
    df_outliers = df.loc[outliers_index]
    unique_outlier_vals = df_outliers[feat].unique()
    n_unique_outlier_vals = len(unique_outlier_vals)

    n_unique_top_percentile = df_feat_value_analysis[f'n_unique_top_{top_percentile}_percent'].values[0]
    n_pop_rep = df_feat_value_analysis['n_pop_rep'].values[0]
    n_top_percentile = df_feat_value_analysis[f'n_top_{top_percentile}_percent'].values[0]
    index_top_percentile = df_feat_value_analysis[f'index_top_{top_percentile}_percent'].values[0]
    val_count_top_percentile = df_feat_value_analysis[f'val_count_top_{top_percentile}_percent'].values[0]

    if not suppress_output:
        display(HTML(f"q1 of <b>{feat}</b> is: {q1}"))
        display(HTML(f"q3 of <b>{feat}</b> is: {q3}"))
        display(HTML(f"count of <b>{feat}</b> outliers is: {n_outliers} (out of {l})"))
        display(HTML(f"ratio of <b>{feat}</b> outliers is: {round(n_outliers_ratio*100,2)}%"))
        display(HTML(f"count of UNIQUE <b>{feat}</b> outlier values is: {n_unique_outlier_vals} (out of {n_unique} unique values)"))

        display(HTML("<br>"))
        display(HTML(f"{n_unique_top_percentile} value(s) (out of {n_unique} unique) of <b>{feat}</b> constitute {round(n_pop_rep*100,2)}% of the total number of observations ({n_top_percentile} out of {l})"))
        if not suppress_values_display:
            display(HTML("values are:"))

    top_percentile_unique_vals = []
    for val, val_count_data in val_count_top_percentile.items(): # yields: val, (count, ratio-of-population)
        top_percentile_unique_vals.append(val)
        if not suppress_output and not suppress_values_display:
            display(HTML(f"{helper__HTML_tabs(1)}{val}{': '+str(round(val_count_data[1]*100,2))+'%' if len(val_count_top_percentile)>1 else ''}"))
    top_percentile_observations = df.loc[index_top_percentile]
    if not suppress_output:
        display(HTML(f"mean of top {top_percentile}th percentile values: {top_percentile_observations[feat].mean()}"))
        display(HTML(f"median of top {top_percentile}th percentile values: {top_percentile_observations[feat].median()}"))
        display(HTML(f"mode of top {top_percentile}th percentile values: {top_percentile_observations[feat].mode()[0]}"))

    best_improvement_strategy = (None, n_outliers, 0)
    best_replace_outliers_rules = None
    all_replace_outliers_rules = {}

    # consider replacment with mean
    mean_feat = df[feat].mean()
    if not suppress_output:
        display(HTML(f"{'<br><br><br>' if n_unique_outlier_vals>0 else '<br>'}"))
        display(HTML(f"mean of all <b>{feat}</b> values is: {mean_feat}"))
    if not suppress_replacement_strat_analysis and n_unique_outlier_vals > 0:
        df_outlier_replacement_candidate, replace_outliers_rules = replace_outliers_index(df, feat, outliers_index, mean_feat)
        all_replace_outliers_rules['mean'] = replace_outliers_rules

        if not suppress_output and not suppress_candidate_head:
            display(HTML(f"{df.loc[outliers_index][[feat]].head().to_html()}"))
            display(HTML(f"{df_outlier_replacement_candidate.loc[outliers_index][[feat]].head().to_html()}"))

        df_outliers_analysis = analyze_outliers(
            df_outlier_replacement_candidate[[feat]], 
            f'{df_name} Outlier-replacement (with mean: {mean_feat}) Candidate', 
            display_plots=True, 
            plot_edge=3,
            suppress_output=suppress_output
        )
        if not suppress_output:
            display(HTML(df_outliers_analysis.to_html(notebook=True)))
        n_new_outliers = df_outliers_analysis['n_outliers'].values[0]
        n_new_outliers_ratio = df_outliers_analysis['n_outliers_ratio'].values[0]
        n_new_unique_vals = len(df_outlier_replacement_candidate[feat].unique())
        s_fail_reason = ""
        b_fewer_outliers = n_new_outliers < best_improvement_strategy[1]
        s_fail_reason += "will not result in fewer outliers" if not b_fewer_outliers else ""
        b_lt_reduc_ratio = n_new_outliers_ratio < outlier_ratio_reduction_threshold
        s_fail_reason += (", " if len(s_fail_reason)>0 else "") + f"will not result in outlier ratio less than max threshold ({outlier_ratio_reduction_threshold})" if not b_lt_reduc_ratio else ""
        b_gt_one_val = n_new_unique_vals > 1
        s_fail_reason += (", " if len(s_fail_reason)>0 else "") + "will reduce to only one unique value" if not b_gt_one_val else ""
        if b_fewer_outliers and b_lt_reduc_ratio and b_gt_one_val:
            best_improvement_strategy = ('mean', n_new_outliers, n_new_outliers_ratio)
            best_replace_outliers_rules = replace_outliers_rules
        if not suppress_output:
            display(HTML(f"<h4>replacing outliers with <u>mean</u> {'will reduce outlier count to '+str(n_new_outliers)+' (from '+str(n_outliers)+')' if best_improvement_strategy[0]=='mean' else s_fail_reason}</h4>"))

    # consider replacment with median
    median_feat = df[feat].median()
    if not suppress_output:
        display(HTML(f"{'<br><br><br>' if n_unique_outlier_vals>0 else ''}"))
        display(HTML(f"median of all <b>{feat}</b> values is: {median_feat}"))
    if not suppress_replacement_strat_analysis and n_unique_outlier_vals > 0:
        if median_feat != mean_feat:
            df_outlier_replacement_candidate, replace_outliers_rules = replace_outliers_index(df, feat, outliers_index, median_feat)
            all_replace_outliers_rules['median'] = replace_outliers_rules

            if not suppress_output and not suppress_candidate_head:
                display(HTML(f"{df.loc[outliers_index][[feat]].head().to_html()}"))
                display(HTML(f"{df_outlier_replacement_candidate.loc[outliers_index][[feat]].head().to_html()}"))

            df_outliers_analysis = analyze_outliers(
                df_outlier_replacement_candidate[[feat]], 
                f'{df_name} Outlier-replacement (with median: {median_feat}) Candidate', 
                display_plots=True, 
                plot_edge=3, 
                suppress_output=suppress_output
            )
            if not suppress_output:
                display(HTML(df_outliers_analysis.to_html(notebook=True)))
            n_new_outliers = df_outliers_analysis['n_outliers'].values[0]
            n_new_outliers_ratio = df_outliers_analysis['n_outliers_ratio'].values[0]
            n_new_unique_vals = len(df_outlier_replacement_candidate[feat].unique())
            s_fail_reason = ""
            b_fewer_outliers = n_new_outliers < best_improvement_strategy[1]
            s_fail_reason += "will not result in fewer outliers" if not b_fewer_outliers else ""
            b_lt_reduc_ratio = n_new_outliers_ratio < outlier_ratio_reduction_threshold
            s_fail_reason += (", " if len(s_fail_reason)>0 else "") + f"will not result in outlier ratio less than max threshold ({outlier_ratio_reduction_threshold})" if not b_lt_reduc_ratio else ""
            b_gt_one_val = n_new_unique_vals > 1
            s_fail_reason += (", " if len(s_fail_reason)>0 else "") + "will reduce to only one unique value" if not b_gt_one_val else ""
            if b_fewer_outliers and b_lt_reduc_ratio and b_gt_one_val:
                best_improvement_strategy = ('median', n_new_outliers, n_new_outliers_ratio)
                best_replace_outliers_rules = replace_outliers_rules
            if not suppress_output:
                display(HTML(f"<h4>replacing outliers with <u>median</u> {'will reduce outlier count to '+str(n_new_outliers)+' (from '+str(n_outliers)+')' if best_improvement_strategy[0]=='median' else s_fail_reason}</h4>"))
        else:
            if not suppress_output:
                display(HTML("(the particular-value replacement scheme was already considered above)"))

    # consider replacment with mode
    mode_feat = df.mode(numeric_only=True)[feat].values[0]
    if not suppress_output:
        display(HTML(f"{'<br><br><br>' if n_unique_outlier_vals>0 else ''}"))
        display(HTML(f"mode of all <b>{feat}</b> values is: {mode_feat}"))
    if not suppress_replacement_strat_analysis and n_unique_outlier_vals > 0:
        if mode_feat != median_feat:
            df_outlier_replacement_candidate, replace_outliers_rules = replace_outliers_index(df, feat, outliers_index, mode_feat)
            all_replace_outliers_rules['mode'] = replace_outliers_rules

            if not suppress_output and not suppress_candidate_head:
                display(HTML(f"{df.loc[outliers_index][[feat]].head().to_html()}"))
                display(HTML(f"{df_outlier_replacement_candidate.loc[outliers_index][[feat]].head().to_html()}"))

            df_outliers_analysis = analyze_outliers(
                df_outlier_replacement_candidate[[feat]], 
                f'{df_name} Outlier-replacement (with mode: {mode_feat}) Candidate', 
                display_plots=True, 
                plot_edge=3,
                suppress_output=suppress_output
            )
            if not suppress_output:
                display(HTML(df_outliers_analysis.to_html(notebook=True)))
            n_new_outliers = df_outliers_analysis['n_outliers'].values[0]
            n_new_outliers_ratio = df_outliers_analysis['n_outliers_ratio'].values[0]
            n_new_unique_vals = len(df_outlier_replacement_candidate[feat].unique())
            s_fail_reason = ""
            b_fewer_outliers = n_new_outliers < best_improvement_strategy[1]
            s_fail_reason += "will not result in fewer outliers" if not b_fewer_outliers else ""
            b_lt_reduc_ratio = n_new_outliers_ratio < outlier_ratio_reduction_threshold
            s_fail_reason += (", " if len(s_fail_reason)>0 else "") + f"will not result in outlier ratio less than max threshold ({outlier_ratio_reduction_threshold})" if not b_lt_reduc_ratio else ""
            b_gt_one_val = n_new_unique_vals > 1
            s_fail_reason += (", " if len(s_fail_reason)>0 else "") + "will reduce to only one unique value" if not b_gt_one_val else ""
            if b_fewer_outliers and b_lt_reduc_ratio and b_gt_one_val:
                best_improvement_strategy = ('mode', n_new_outliers, n_new_outliers_ratio)
                best_replace_outliers_rules = replace_outliers_rules
            if not suppress_output:
                display(HTML(f"<h4>replacing outliers with <u>mode</u> {'will reduce outlier count to '+str(n_new_outliers)+' (from '+str(n_outliers)+')' if best_improvement_strategy[0]=='mode' else s_fail_reason}</h4>"))
        else:
            if not suppress_output:
                display(HTML("(the particular-value replacement scheme was already considered above)"))

    if not suppress_output and n_unique_outlier_vals > 0:
        display(HTML(f"<br><h3>outlier-analysis recommendation: <b>{'replace <i>'+feat+'</i> outlier values with <u>'+best_improvement_strategy[0]+'</u>' if best_improvement_strategy[0] is not None else 'drop this feature'}</b></h3>"))

    return outliers_index, best_replace_outliers_rules, all_replace_outliers_rules

def analyze_outliers_grouped_by(df, feat, group_by, suppress_header=False):
    # the "header"
    if not suppress_header:
        display(HTML(df.groupby(group_by)[feat].describe().to_html(notebook=True, justify='left')))
        display(HTML("<p/><br/>"))
        plot_group_by(df, feat, group_by)
        display(HTML("<p/><br/>"))

    display(HTML("<p/><br/>"))
    classes = list(df[group_by].unique())
    for _class in classes:
        display(HTML(f"<h2>Outlier-replacement Strategy analysis of <i>{feat}</i> for <i>{group_by}</i> class '<font color='blue'>{_class}</font>'</h2>"))
        analyze_outliers_detailed(
            df.query(f"{group_by}=='{_class}'"), 
            f"df.query(\"{group_by}=='{_class}'\")", 
            feat, 
            top_percentile=100, 
            outlier_ratio_reduction_threshold=.10
        );
        display(HTML("<p/><br/><br/><br/>"))


def analyze_non_alphanumeric_strings(df, df_name, truncate_output_threshold=50, suppress_output=False):
    l = len(df)
    
    if not suppress_output:
        display(HTML(f"<h3>{df_name} DataFrame Non-Alphanumeric-String Value Analysis (out of {l} total observations)</h3>"))

    df_analysis, _ = analyze_values(df, df_name, standard_options_kargs={'compute_non_alphanumeric_str':True}, suppress_output=True)
    df_analysis = df_analysis.loc[df_analysis['dtype']==object][['feature', 'n_unique', 'n_non_alphanumeric_str', 'non_alphanumeric_str_index']]
    
    if not suppress_output:
        display(HTML(df_analysis.to_html(notebook=True)))
        display(HTML("<p><br>"))

    str_feat_unique_nonalphanumeric_string_vals = {}
    for str_feat in list(df_analysis.feature.unique()):
        feat_nonalphanumeric_index = df_analysis.loc[df_analysis['feature']==str_feat]['non_alphanumeric_str_index'].values[0] # n_non_alphanumeric_str
        if feat_nonalphanumeric_index is not None:
            unique_nonalphanumeric_strings = df.loc[feat_nonalphanumeric_index][str_feat].unique()
            str_feat_unique_nonalphanumeric_string_vals[str_feat] = df.loc[feat_nonalphanumeric_index][str_feat].unique()
            n_unique_non_alphanumeric = len(unique_nonalphanumeric_strings)
            
            if not suppress_output:
                display(HTML(f"in {df_name}, string-type feature <b>{str_feat}</b> has, occurring in {len(feat_nonalphanumeric_index)} observations, {n_unique_non_alphanumeric} unique non-alphanumeric values:"))
            
                for i, nonalphanumeric_string in enumerate(sorted(unique_nonalphanumeric_strings)):
                    if i < truncate_output_threshold:
                        display(HTML(f"{helper__HTML_tabs(1)}'{nonalphanumeric_string}'"))
                    else:
                        display(HTML(f"*** TRUNCATED since <b>{str_feat}</b> has {n_unique_non_alphanumeric} unique non-alphanumeric values***"))
                        break
                display(HTML("<p><br>"))

    return (str_feat_unique_nonalphanumeric_string_vals, df_analysis)


def index_of_values(df, df_name, feat, values):
    values_found_at = np.where(np.isin(df[feat].values, values))[0]
    return (values_found_at, df.iloc[values_found_at][feat].unique())


def analyze_overlap(feat_group_name, feat_group, df_name, df, max_unique_display=100, suppress_value_analysis_output=False):
    s_list = '[' + ''.join([f"{', ' if i > 0 else ''}'{feat}'" for i, feat in enumerate(feat_group)]) + ']'
    display(HTML(f"<h4>Overlap Analysis: <i>{feat_group_name}</i> Types: {s_list}</h4>"))

    df_analysis, duplicates_entries = analyze_values(
        df[feat_group], 
        f"{df_name} <i>{feat_group_name}</i> Type", 
        standard_options_kargs={'sort_unique_vals':True, 'hide_cols':True},
        compute_duplicates__kargs={'display_heads':False},
        suppress_output=suppress_value_analysis_output
    )

    display(HTML("<p><br>"))
    helper__display_unique_vals_from_analysis(df_analysis, max_unique_display)

    display(HTML("<p><br>"))
    helper__display_complement_from_duplicates_entries(duplicates_entries, df)


def analyze_distributions(
    df, 
    df_name, 
    feat, 
    percentiles=None, 
    suppress_100th_percentile_display=False,
    truncate_sig_after_n=100,
    map_to_candidates=['none','other','unknown'],
    suppress_map_to_candidates_display=False,
    suppress_output=False,
    fs=(20,10)
):
    if not suppress_output:
        display(HTML(f"***** Distributions Analysis for <b>{feat}</b> in {df_name}: BEGIN *****"))

    result_by_percentile = {}

    final_percentiles = [100]

    if percentiles is not None:
        final_percentiles.extend(percentiles)

    for i_p, percentile in enumerate(final_percentiles):
        if (percentile!=100 or not suppress_100th_percentile_display) and not suppress_output:
            display(HTML(f"<h5>PERCENTILE: {percentile}</h5>"))

        (
            many_categories_top_percent_index, 
            many_categories_val_count_top_percent,
            reduction
        ) = analyze_value_detailed(
            df, 
            df_name, 
            feat, 
            top_percentile=percentile, 
            suppress_values=True, 
            suppress_output=(percentile==100 or not suppress_100th_percentile_display) and not suppress_output
        )

        all_unique = df[feat].unique()
        n_all_unique = len(all_unique)

        if (percentile != 100 or not suppress_100th_percentile_display) and not suppress_output:
            width = 1
            fig = plt.figure(figsize=fs)
            plt.title(f"{n_all_unique} TOTAL unique " + r"$\bf{" + feat + "}$" + f" categories in {len(df)} observations", y=1.08)
            ax = plt.gca()
            rects = ax.patches

        bar_label_voffset = .0375
        sig_alpha = 0.4

        sig_cats = list(filter(lambda e: e[1][1], many_categories_val_count_top_percent.items()))
        if len(many_categories_val_count_top_percent) > truncate_sig_after_n:
            all_cats = list(many_categories_val_count_top_percent.items())
            all_densities = [c[1][1] for c in all_cats]
            all_counts = [c[1][0] for c in all_cats]
            sig_cats = all_cats[:truncate_sig_after_n]
            sig_densities = [sc[1][1] for sc in sig_cats]
            sig_counts = [sc[1][0] for sc in sig_cats]
            sig_cats.extend([(f'** TRUNCATED SIGNIFICANT ({len(all_cats)-truncate_sig_after_n} categories) **', (sum(all_counts)-sum(sig_counts), sum(all_densities)-sum(sig_densities)))])

        densities = [sig_cat[1][1] for sig_cat in sig_cats]
        counts = [sig_cat[1][0] for sig_cat in sig_cats]
        labels = [str(sig_cat[0]) for sig_cat in sig_cats]
        pop_rep = sum(densities)
        pop_rep_count = sum(counts)

        d_pop_rep = [pop_rep for d in densities]
        d_pop_rep_labels = [l for l in labels]
        colors = ['lime' for d in densities]
        if len(many_categories_val_count_top_percent) < n_all_unique:
            d_pop_rep.extend([1-pop_rep])
            d_pop_rep_labels.extend([f'** INSIGNIFICANT ({n_all_unique-len(many_categories_val_count_top_percent)} categories) **'])
            colors.extend(['red'])

        if (percentile != 100 or not suppress_100th_percentile_display) and not suppress_output:
            plt.bar(d_pop_rep_labels, d_pop_rep, width=width, color=colors)

            # top label for grouping of significant (based on percentile)
            ax.text(
                (rects[-2].get_x()-rects[0].get_x())/2, 
                pop_rep+bar_label_voffset,
                f"{round(pop_rep*100,4)}% ({len(many_categories_val_count_top_percent)} {'SIGNIFICANT ' if percentile != 100 else ''}categories in {pop_rep_count} observations)",
                ha='center', 
                weight='bold'
            )
            # annotation for insignificant (based on percentile)
            if len(many_categories_val_count_top_percent) < n_all_unique:
                ax.text(
                    rects[-1].get_x()+rects[-1].get_width()/2, 

                    # d_pop_rep[-1]+bar_label_voffset,
                    rects[-1].get_y()+rects[-1].get_height()+bar_label_voffset,

                    f"{round(d_pop_rep[-1]*100,4)}% ({len(df)-pop_rep_count} observations)",
                    ha='center', 
                    rotation=90,
                    weight='bold'
                )

            plt.bar(labels, densities, width=width, color='blue')
            # annotation for significant (based on percentile)
            for i_r_d, (rect, d) in enumerate(zip(rects, densities)):
                ax.text(
                    rect.get_x()+rect.get_width()/2, 

                    d+bar_label_voffset,

                    f"{round(d*100,4)}% ({counts[i_r_d]} observations)",
                    ha='center', 
                    rotation=90,
                    weight='bold'
                )

            ax.set_ybound(upper=1, lower=0)
            # ax.set_ybound(lower=0, upper=pop_rep)

            plt.xticks(rotation=90)
            plt.show()

        if percentile==100 and not suppress_100th_percentile_display and not suppress_output:
            display(HTML("<br><br>"))
            plt.figure(figsize=fs)
            # plt.hist(df[feat], bins=n_all_unique)
            if df[feat].dtype==object:
                matplotlib.rcParams.update(matplotlib.rcParamsDefault)
                sns.countplot(df[feat])
            else:
                sns.distplot(df[feat], bins=n_all_unique)
            # plt.gca().set_xlim(left=df[feat].min(), right=df[feat].max())
            plt.xticks(rotation=90)
            if n_all_unique > 100:
                plt.gca().get_xaxis().set_visible(False)
            plt.show()
            sns.set_style('whitegrid')

        percentile_entry = {}

        sig_contains_candidate_map_to_category = {}
        insig_contains_candidate_map_to_category = {}
        insig_percentile_entry_categories = list(set(all_unique) - set(many_categories_val_count_top_percent.keys()))
        insig_percentile_entry_index = df[~df.index.isin(many_categories_top_percent_index)].index

        for candidate_map_to_category in map_to_candidates:
            sig_contains_candidate_map_to_category[candidate_map_to_category] = candidate_map_to_category in many_categories_val_count_top_percent.keys()
            if not suppress_map_to_candidates_display and percentile < 100 and not suppress_output:
                display(HTML(f"{helper__HTML_tabs(1)}categories {'of top '+str(percentile)+'%' if percentile!=100 else ''} of '{feat}' contain '{candidate_map_to_category}'? {sig_contains_candidate_map_to_category[candidate_map_to_category]}"))
            insig_contains_candidate_map_to_category[candidate_map_to_category] = candidate_map_to_category in insig_percentile_entry_categories
            if not suppress_map_to_candidates_display and percentile < 100 and not suppress_output:
                display(HTML(f"{helper__HTML_tabs(1)}insignificant categories of '{feat}' contain '{candidate_map_to_category}'? {insig_contains_candidate_map_to_category[candidate_map_to_category]}"))

        percentile_entry['sig'] = (list(many_categories_val_count_top_percent.keys()), many_categories_top_percent_index, sig_contains_candidate_map_to_category)
        percentile_entry['insig'] = (insig_percentile_entry_categories, insig_percentile_entry_index, insig_contains_candidate_map_to_category)

        # if i_p < len(final_percentiles)-1:
        if (percentile!=100 or not suppress_100th_percentile_display) and not suppress_output:
            display(HTML("<br><br><br><br>"))

        result_by_percentile[percentile] = percentile_entry
    if not suppress_output:
        display(HTML(f"***** Distributions Analysis for <b>{feat}</b> in {df_name}: END *****"))

    return result_by_percentile

def analyze_distributions__top_n(
    df, 
    df_name, 
    feat, 
    top_n,
    truncate_sig_after_n=100,
    map_to_candidates=['none','other','unknown'],
    suppress_map_to_candidates_display=False,
    suppress_output=False
):
    if not suppress_output:
        display(HTML(f"***** Distributions Analysis for <b>{feat}</b> in {df_name}: BEGIN *****"))

    result_by_top_n = {}

    (
        many_categories_top_n_index, 
        many_categories_val_count_top_n,
        reduction
    ) = analyze_value_detailed__top_n(
        df, 
        df_name, 
        feat, 
        top_n, 
        suppress_values=True, 
        suppress_output=suppress_output
    )
    
    truncate_sig_after_n = min(top_n, truncate_sig_after_n)

    all_unique = df[feat].unique()
    n_all_unique = len(all_unique)

    if not suppress_output:
        display(HTML("<p><br>"))

        fs = (20,10)
        width = 1
        fig = plt.figure(figsize=fs)
        plt.title(f"{n_all_unique} TOTAL unique " + r"$\bf{" + feat + "}$" + f" categories in {len(df)} observations", y=1.08)
        ax = plt.gca()
        rects = ax.patches

        bar_label_voffset = .0375
        sig_alpha = 0.4

        sig_cats = list(filter(lambda e: e[1][1], many_categories_val_count_top_n.items()))
        if len(many_categories_top_n_index) > truncate_sig_after_n:
            all_cats = list(many_categories_val_count_top_n.items())
            all_densities = [c[1][1] for c in all_cats]
            all_counts = [c[1][0] for c in all_cats]
            sig_cats = all_cats[:truncate_sig_after_n]
            sig_densities = [sc[1][1] for sc in sig_cats]
            sig_counts = [sc[1][0] for sc in sig_cats]
            if len(all_cats)-truncate_sig_after_n > 0:
                sig_cats.extend([(f'** TRUNCATED SIGNIFICANT ({len(all_cats)-truncate_sig_after_n} categories) **', (sum(all_counts)-sum(sig_counts), sum(all_densities)-sum(sig_densities)))])

        densities = [sig_cat[1][1] for sig_cat in sig_cats]
        counts = [sig_cat[1][0] for sig_cat in sig_cats]
        labels = [str(sig_cat[0]) for sig_cat in sig_cats]
        pop_rep = sum(densities)
        pop_rep_count = sum(counts)

        d_pop_rep = [pop_rep for d in densities]
        d_pop_rep_labels = [l for l in labels]
        colors = ['lime' for d in densities]
        if len(many_categories_val_count_top_n) < n_all_unique:
            d_pop_rep.extend([1-pop_rep])
            d_pop_rep_labels.extend([f'** INSIGNIFICANT ({n_all_unique-len(many_categories_val_count_top_n)} categories) **'])
            colors.extend(['red'])

        plt.bar(d_pop_rep_labels, d_pop_rep, width=width, color=colors)
        # top label for grouping of significant
        ax.text(
            (rects[-2].get_x()-rects[0].get_x())/2, 
            pop_rep+bar_label_voffset,
            f"{round(pop_rep*100,4)}% ({len(many_categories_val_count_top_n)} categories in {pop_rep_count} observations)",
            ha='center', 
            weight='bold'
        )
        # annotation for insignificant
        if len(many_categories_val_count_top_n) < n_all_unique:
            ax.text(
                rects[-1].get_x()+rects[-1].get_width()/2, 

                # d_pop_rep[-1]+bar_label_voffset,
                rects[-1].get_y()+rects[-1].get_height()+bar_label_voffset,

                f"{round(d_pop_rep[-1]*100,4)}% ({len(df)-pop_rep_count} observations)",
                ha='center', 
                rotation=90,
                weight='bold'
            )

        plt.bar(labels, densities, width=width, color='blue')

        # annotation for significant (based on percentile)
        for i_r_d, (rect, d) in enumerate(zip(rects, densities)):
            ax.text(
                rect.get_x()+rect.get_width()/2, 

                d+bar_label_voffset,
                
                f"{round(d*100,4)}% ({counts[i_r_d]} observations)",
                ha='center', 
                rotation=90,
                weight='bold'
            )

        # ax.set_ybound(upper=pop_rep, lower=0)
        ax.set_ybound(upper=1, lower=0)

        plt.xticks(rotation=90)

        plt.show()

        display(HTML("<br><br>"))
        plt.figure()
        plt.hist(df[feat])
        plt.gca().get_xaxis().set_visible(False)
        plt.show()

    top_n_entry = {}

    sig_contains_candidate_map_to_category = {}
    insig_contains_candidate_map_to_category = {}
    insig_percentile_entry_categories = list(set(all_unique) - set(many_categories_val_count_top_n.keys()))
    insig_percentile_entry_index = df[~df.index.isin(many_categories_top_n_index)].index

    for candidate_map_to_category in map_to_candidates:
        sig_contains_candidate_map_to_category[candidate_map_to_category] = candidate_map_to_category in many_categories_val_count_top_n.keys()
        if not suppress_output:
            display(HTML(f"{helper__HTML_tabs(1)}categories of top {top_n} values of '{feat}' contain '{candidate_map_to_category}'? {sig_contains_candidate_map_to_category[candidate_map_to_category]}"))
        insig_contains_candidate_map_to_category[candidate_map_to_category] = candidate_map_to_category in insig_percentile_entry_categories
        if not suppress_output:
            display(HTML(f"{helper__HTML_tabs(1)}insignificant categories of '{feat}' contain '{candidate_map_to_category}'? {insig_contains_candidate_map_to_category[candidate_map_to_category]}"))

    top_n_entry['sig'] = (list(many_categories_val_count_top_n.keys()), many_categories_top_n_index, sig_contains_candidate_map_to_category)
    top_n_entry['insig'] = (insig_percentile_entry_categories, insig_percentile_entry_index, insig_contains_candidate_map_to_category)

    result_by_top_n[top_n] = top_n_entry

    if not suppress_output:
        display(HTML(f"***** Distributions Analysis for <b>{feat}</b> in {df_name}: END *****"))

    return result_by_top_n

def analyze_distributions_grouped_by(df, feat, group_by, suppress_header=False, fs=(20,10)):
    # the "header"
    if not suppress_header:
        display(HTML(df.groupby(group_by)[feat].describe().to_html(notebook=True, justify='left')))
        display(HTML("<p/><br/>"))
        plot_group_by(df, feat, group_by)
        display(HTML("<p/><br/>"))

    classes = list(df[group_by].unique())
    for _class in classes:
        display(HTML(f"<h2>Distributions analysis of <i>{feat}</i> for <i>{group_by}</i> class '<font color='blue'>{_class}</font>'</h2>"))
        analyze_distributions(
            df.query(f"{group_by}=='{_class}'"), 
            f"df.query(\"{group_by}=='{_class}'\")", 
            feat,
            fs=fs
        );
        display(HTML("<p/><br/><br/><br/>"))


def helper__summarize_categorical_reduction(result, feat, sig, map_to):
    n_unique_before = len(result[100]['sig'][0])
    n_sig_unique_after = len(result[sig]['sig'][0])
    n_sig_after = len(result[sig]['sig'][1])
    n_insig_unique_after = len(result[sig]['insig'][0])
    n_insig_after = len(result[sig]['insig'][1])
    insig_map_to_categories_occurence_after = result[sig]['insig'][2]

    display(HTML(f"There are {n_insig_unique_after} unique categories in {n_insig_after} INSIGNIFICANT (bottom {100-sig}%) <b>{feat}</b> observations."))
    display(HTML(f"Occurrence of candidate-map-to-categories in these INSIGNIFICANT categories is as follows: {insig_map_to_categories_occurence_after}."))
    display(HTML(f"<br>By selecting categories from the top {sig}% (by count) of feature <b>{feat}</b>, we reduce the magnitude of the set of unique categories from {n_unique_before} to {n_sig_unique_after} (a {round(abs(1 - 1/(n_sig_unique_after/n_unique_before))*100,2)}% reduction in size!).  This will yield a significant reduction of One-Hot Encoded column explosion and, consequently, a significant performance gain when building classification models."))
    display(HTML(f"<br><br>We will map these \"insignificant\" {n_insig_unique_after} categories ({n_insig_after} observations) to \"{map_to}\"."))


def reduce_unique__flat_n(df, df_name, feat, target_n, suppress_values=False):
    l = len(df)
    starting_unique_vals = df[feat].unique()
    n_unique = len(starting_unique_vals)
    n_unique_top_percentile = n_unique
    actual_percentile = 0
    top_percentile = 100

    display(HTML(f"Attempting to reduce unique {feat} values in {df_name} to a flat count of {target_n} members..."))

    # stopping condition: n_unique < 
    while n_unique > 1 and n_unique_top_percentile > target_n and top_percentile > 0:
        df_analysis, _ = analyze_values(
            df[[feat]], 
            f"{df_name} <i>{feat}</i> Detailed Value Analysis", 
            standard_options_kargs={'sort_unique_vals':True, 'hide_cols':True},
            compute_top__kargs={'strategy':'percentile', 'value':top_percentile},
            suppress_output=True
        )
        df_feat_value_analysis = df_analysis.loc[df_analysis['feature']==feat]
        n_unique_top_percentile = df_analysis[f'n_unique_top_{top_percentile}_percent'].values[0]
        actual_percentile = df_analysis['n_pop_rep'].values[0]*100
        n_top_percentile = df_analysis[f'n_top_{top_percentile}_percent'].values[0]
        val_count_top_percentile = df_analysis[f'val_count_top_{top_percentile}_percent'].values[0]

        top_percentile -= 1

    display(HTML(f"{helper__HTML_tabs(1)}FOUND: <b>{n_unique_top_percentile} unique value(s)</b> (out of {n_unique} unique) <b>of <i>{feat}</i> constitute <u>{round(actual_percentile,2)}%</u> of the total number of observations</b> ({n_top_percentile} out of {l})"))

def reduce_unique__first_n_chars(df, df_name, feat, target_first_n_chars=10, suppress_no_reduc=True, suppress_output=False):
    df_abbrev = df.copy()

    cols = [f'{feat}_abbrev', f'{feat}s', f'n_{feat}s']

    n_unique = len(df_abbrev[feat].unique())

    df_abbrev[cols[0]] = df_abbrev[feat].str.slice(0,target_first_n_chars)

    unique_abbrev = df_abbrev[cols[0]].unique()

    df_abbrev_analysis = pd.DataFrame(columns=cols)
    for ufa in list(unique_abbrev):
        u_vals_for_ufa = df_abbrev.query(f"{cols[0]}=='{ufa}'")[feat].unique()
        data = [{
            f'{cols[0]}': ufa, 
            f'{cols[1]}': list(u_vals_for_ufa),
            f'{cols[2]}': len(u_vals_for_ufa),
        }]
        df_abbrev_analysis = df_abbrev_analysis.append(data, ignore_index=True, sort=False)
    df_abbrev_analysis = df_abbrev_analysis.sort_values(by=cols[0]).reset_index().drop('index', axis=1)

    df_abbrev_unique_reduc = df_abbrev_analysis.query(f"{cols[2]}>1").set_index(cols[0])
    if not suppress_output:
        display(HTML(f"<h3>Reduction of <i>{feat}</i> unique-values space by replacing {feat} with first {target_first_n_chars} chars</h3>"))
        display(HTML(f"n-unique abbrev: {len(list(unique_abbrev))} (down from {n_unique} total)"))
        display(HTML(df_abbrev_unique_reduc.to_html()))
    n_reduction = df_abbrev_unique_reduc[cols[2]].sum() - len(df_abbrev_unique_reduc)
    if not suppress_output:
        display(HTML(f"<p><br><h4>TOTAL UNIQUE-COUNT REDUCTION by replacing <i>{feat}</i> with <i>{cols[0]}</i> (first {target_first_n_chars} chars): {n_reduction}</h4>"))

    df_abbrev_NO_unique_reduc = df_abbrev_analysis.query(f"{cols[2]}<=1")
    n_no_reduction = len(unique_abbrev) - n_reduction
    df_abbrev_NO_unique_reduc = df_abbrev_NO_unique_reduc.set_index(cols[0])
    if not suppress_no_reduc and not suppress_output:
        display(HTML(df_abbrev_NO_unique_reduc.head().to_html()))
    if not suppress_output:
        display(HTML(f"<h4>no. categories of <i>{feat}</i> where replacement does not reduce unique-count - i.e. where this is only one <i>{feat}</i> category mapped to an abbreviation: {n_no_reduction}</h4>"))

    # now do the replacement
    df_abbrev[feat] = df_abbrev[cols[0]]
    df_abbrev = df_abbrev.drop(cols[0], axis=1)

    # now check for non-identifying differences
    df_abbrev_analysis = df_abbrev_analysis.set_index(cols[0])
    non_indentifying_abbrevs = []
    for abbrev, row in df_abbrev_analysis.iterrows():
        vals_to_replace = row[cols[1]]

        _abbrev_plus_one_char = [val_to_replace[0:target_first_n_chars+1] for val_to_replace in vals_to_replace]
        _abbrev_plus_one_char_combinations = list(combinations(_abbrev_plus_one_char, 2))
        for _abbrev_plus_one_char_combo in _abbrev_plus_one_char_combinations:
            if _abbrev_plus_one_char_combo[0] != _abbrev_plus_one_char_combo[1]:
                non_indentifying_abbrevs.append((abbrev, _abbrev_plus_one_char_combo))
                break

    if not suppress_output:
        display(HTML("<p><br>"))

    return df_abbrev, n_reduction, df_abbrev_unique_reduc, non_indentifying_abbrevs

def minimize_loss__first_n_chars_reduction(df, df_name, feat, target_first_n_chars__start, target_first_n_chars__stop, verbose=False):
    min_loss_of_info = None

    for target_n in range(target_first_n_chars__start, target_first_n_chars__stop-1, -1):
        _, n_reduction, df_abbrev_analysis, non_indentifying_abbrevs = reduce_unique__first_n_chars(
            df, 
            df_name, 
            feat, 
            target_first_n_chars=target_n,
            suppress_output=not verbose
        )

        if len(non_indentifying_abbrevs) > 0:
            _abbrevs_potential_loss_of_info = list(map(lambda nic: nic[0], non_indentifying_abbrevs))
            n_count = len(_abbrevs_potential_loss_of_info)
            loss_ratio = n_count/n_reduction
            display(HTML(f"***WARNING!!!***  Some abbreviations ({n_count} out of {n_reduction}, ratio: {round(loss_ratio,4)}) from this scheme (abbrev. len: {target_n}) may result in lost information."))
            if min_loss_of_info is None or loss_ratio < min_loss_of_info[3]:
                min_loss_of_info = (target_n, n_count, n_reduction, loss_ratio)

    display(HTML(f"<p><br>scheme with minimal loss of info: {min_loss_of_info}"))

    df_abbrev_replacement_candidate, _, _, _ = reduce_unique__first_n_chars(
        df, 
        f"suggested <i>{feat}</i> abbreviation scheme for {df_name}", 
        feat, 
        target_first_n_chars=min_loss_of_info[0],
        suppress_output=not verbose
    )

    return df_abbrev_replacement_candidate, min_loss_of_info


def json_to_md5_hash_digest(json_object):
    json_object_as_string = json.dumps(json_object, separators=(',', ':'))
    hashed_json_object_as_string = hashlib.md5(json_object_as_string.encode()) # UTF-8 is default
    return hashed_json_object_as_string.hexdigest()

def get_data_fname(eda_cfg, preprocessing_options_cfg, data_kwargs):
    is_data_cached = 'is_cached' in data_kwargs and data_kwargs['is_cached']
    digest_str = eda_cfg['digest'] if is_data_cached else json_to_md5_hash_digest(preprocessing_options_cfg)
    #print(f"eda_cfg as md5 hash digest: {digest_str}")

    if data_kwargs['is_labels']: # then we want the fname for labels
        fname = f"{eda_cfg['labels'][data_kwargs['type']]['fname_prefix']}-{digest_str}.{eda_cfg['labels']['fname_ext']}"
    else: # then we want the fname for predictors
        fname = f"{eda_cfg['wrangled_data'][data_kwargs['type']]['fname_prefix']}-{digest_str}.{eda_cfg['wrangled_data']['fname_ext']}"

    return fname

def get_model_result_fname(eda_cfg, preprocessing_options_cfg, data_kwargs):
    is_data_cached = 'is_cached' in data_kwargs and data_kwargs['is_cached']
    digest_str = eda_cfg['digest'] if is_data_cached else json_to_md5_hash_digest(preprocessing_options_cfg)
    #print(f"eda_cfg as md5 hash digest: {digest_str}")

    return f"models-results-{digest_str}.json"

def get_preprocessing_options_fname(eda_cfg, preprocessing_options_cfg, data_kwargs=None):
    is_data_cached = data_kwargs is not None and 'is_cached' in data_kwargs and data_kwargs['is_cached']
    digest_str = eda_cfg['digest'] if is_data_cached else json_to_md5_hash_digest(preprocessing_options_cfg)
    #print(f"eda_cfg as md5 hash digest: {digest_str}")

    return f"preprocessing-options-{digest_str}.json", "preprocessing-spec-last.json"


# def find_weird_vals(df, df_name, regx_weird_val=r"\b[^a-zA-Z]+\b", suppress_output=False):
def find_weird_vals(df, df_name, regx_weird_val=r"( )*[^a-zA-Z]+( )*", suppress_output=False):
    df_values_analysis, _ = analyze_values(df, df_name, standard_options_kargs={'sort_unique_vals':True,'hide_cols':True}, suppress_output=True)
    df_analysis_object_feats = df_values_analysis.loc[df_values_analysis['dtype']==object]
    weird_vals_map = {}
    for idx, row in df_analysis_object_feats.iterrows():
        feat = row['feature']
        unique_nonalphabetic_string_vals = row['unique_vals']
        weird_vals = []
        for val in unique_nonalphabetic_string_vals:
            weird_val = re.match(regx_weird_val, val)
            if weird_val:
                weird_vals.append(val)
        if len(weird_vals) > 0:
            weird_vals_map[feat] = weird_vals

    if not suppress_output:
        if len(weird_vals_map) > 0:
            for feat, weird_vals in weird_vals_map.items():
                print(f"{feat}:")
                for weird_val in weird_vals:
                    print(f"\t'{weird_val}' is weird (according to weird-value regex: '{regx_weird_val}')")
            print()
        else:
            print(f"there are no weird vals (according to weird-value regex: '{regx_weird_val}')")

    return weird_vals_map


def display_sourcecode(f):
    display(HTML(f"<code class='python'><pre>{inspect.getsource(f)}</pre></code>"))


def convert_col_type(df, feat, to_type):
    df_copy = df.copy()
    df_copy[feat] = df_copy[feat].astype(to_type)
    return df_copy

# because a date can be represented in many different formats (as a string), there is a specialized function for conversion to datetime type (from stringß)
def convert_col_to_date_type(df, feat, format="%Y-%m-%d"):
    df_copy = df.copy()
    df_copy[feat] = pd.to_datetime(df_copy[feat], format=format)
    return df_copy


def plot_group_by(df, feat, groupby):
    plt.figure(figsize=(10,5))
    # plt.scatter(df[feat], df[groupby])
    sns.boxplot(df[groupby], df[feat])
    plt.show()

def update_df_from_other(df_dest, df_src, feat):
    df_dest[feat] = df_src[feat]
    return df_dest    



def pipeline__fit(pipeline_to_copy, X_to_fit, y_to_fit):
    pipeline = copy(pipeline_to_copy)
    return pipeline.fit(X_to_fit, y_to_fit) # returns pipeline

def pipeline__transform(pipeline, X_to_transform):
    return pipeline.transform(X_to_transform)   # returns transformed version of X_to_transform

def pipeline__fit_transform(pipeline_to_copy, X_to_fit, y_to_fit, X_to_transform=None):
    pipeline = pipeline__fit(pipeline_to_copy, X_to_fit, y_to_fit)
    return pipeline__transform(pipeline, X_to_transform if X_to_transform is not None else X_to_fit)


def display_feature_grouping_header(feat_groupings, group_id, df, df_name):
    display(HTML(f"<h3>Feature grouping: <i><a id='{group_id}'><font color='red'>{group_id}</font></a></i></h3>"))
    display(HTML(f"<h4>Type: <i>{feat_groupings[group_id]['description']['type']}</i></h4>"))
    display(HTML(f"<h4>Description:</h4>"))
    desc = feat_groupings[group_id]['description']['description']
    desc = desc if desc is not None and len(desc.strip())>0 else "<i>None provided.</i>"
    display(HTML(desc))
    display(HTML(f"<h4>Features in this group:</h4>"))
    display(HTML(f"{feat_groupings[group_id]['features']}<p><br>"))
    analyze_values(df[feat_groupings[group_id]['features']], df_name, standard_options_kargs={'sort_unique_vals':True})

def display_pretty_feature_groupings(feat_groupings, include_preprocessing=False):
    feat_groupings_copy = feat_groupings.copy()
    if not include_preprocessing:
        for group_id, feat_group in feat_groupings_copy.items():
            if 'preprocessing_strategy' in feat_group:
                del feat_group['preprocessing_strategy']
    # print(json.dumps(feat_groupings_copy, indent=4))
    display(HTML(f"<pre>{json.dumps(feat_groupings_copy, indent=4)}</pre>"))