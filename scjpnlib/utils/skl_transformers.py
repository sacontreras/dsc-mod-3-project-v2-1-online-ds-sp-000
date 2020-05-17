import pandas as pd
import numpy as np
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

class DropColumnsTransformer(object) :
    def __init__(self, columns_to_drop, error_on_non_existent_col=False):
        self.columns_to_drop = columns_to_drop
        self.error_on_non_existent_col = error_on_non_existent_col
        self.prior_columns = None
        self.dropped_columns = None

    def __filter_columns(self, df):
        cols = []
        existing_cols = list(df.columns)
        for col in self.columns_to_drop:
            if col in existing_cols:
                cols.append(col)
        return cols

    def fit(self, X, y=None): # nothing will be done to y
        self.prior_columns = list(X.columns)
        if not self.error_on_non_existent_col: # get list of existing columns from those passed in
            self.columns_to_drop = self.__filter_columns(X)
        else:
            X.drop(self.columns_to_drop, axis=1) # not in place; done for the sake of error_on_non_existent_col
        return self

    def transform(self, X):
        X_after_drop = X.drop(self.columns_to_drop, axis=1)
        self.dropped_columns = self.columns_to_drop
        return X_after_drop

    def fit_transform(self, X, y=None): # nothing will be done to y
        self = self.fit(X, y)
        return self.transform(X)


class StringCaseTransformer(object):
    def __init__(self, replacement_rules_dict, error_on_bad_col=False):
        self.replacement_rules_dict = replacement_rules_dict
        self.error_on_bad_col = error_on_bad_col
        self.columns_replacement_index_dict = None

    def __filter_columns(self, df):
        cols = []
        existing_cols = list(df.columns)
        replacement_rules_dict_copy = self.replacement_rules_dict.copy()
        for key in self.replacement_rules_dict.keys():
            if key not in existing_cols or not is_string_dtype(df[key]):
                del replacement_rules_dict_copy[key]
        return replacement_rules_dict_copy

    def fit(self, X, y=None): # nothing will be done to y
        if not self.error_on_bad_col: # remove non-existent and non-string columns from keys
            self.replacement_rules_dict = self.__filter_columns(X)
        else:
            X.drop(self.replacement_rules_dict.keys(), axis=1) # not in place; done for the sake of error_on_non_existent_col
        return self

    def transform(self, X):
        X_case_replaced = X.copy()
        self.columns_replacement_index_dict = {}
        for feat, f_replacement in self.replacement_rules_dict.items():
            if f_replacement is str.upper:
                self.columns_replacement_index_dict[feat] = X[X[feat].str.isupper()==False].index
                X_case_replaced[feat] = X_case_replaced[feat].str.upper()
            else:
                self.columns_replacement_index_dict[feat] = X[X[feat].str.islower()==False].index
                X_case_replaced[feat] = X_case_replaced[feat].str.lower()
        return X_case_replaced

    def fit_transform(self, X, y=None): # nothing will be done to y
        self = self.fit(X, y)
        return self.transform(X)


class SimpleValueTransformer(object):
    def __init__(self, replacement_rules_dict, error_on_non_existent_col=False, verbose=False):
        self.replacement_rules_dict = replacement_rules_dict
        self.error_on_non_existent_col = error_on_non_existent_col
        self.verbose = verbose
        self.simple_imputers = None
        self.columns_replacement_index_dict = None

    def __filter_columns(self, df):
        cols = []
        existing_cols = list(df.columns)
        replacement_rules_dict_copy = self.replacement_rules_dict.copy()
        for key in self.replacement_rules_dict.keys():
            if key not in existing_cols:
                del replacement_rules_dict_copy[key]
        return replacement_rules_dict_copy

    def fit(self, X, y=None): # nothing will be done to y
        if not self.error_on_non_existent_col: # remove non-existent columns from keys
            self.replacement_rules_dict = self.__filter_columns(X)
        else:
            X.drop(self.replacement_rules_dict.keys(), axis=1) # not in place; done for the sake of error_on_non_existent_col
        return self

    def transform(self, X):
        X_vals_replaced = X.copy()
        self.columns_replacement_index_dict = {}
        self.simple_imputers = {}
        for feat_with_vals_to_replace, replacement_rule in self.replacement_rules_dict.items():
            simple_imputer_default = SimpleImputer()
            simple_imputer = SimpleImputer(
                missing_values = replacement_rule['missing_values'] if 'missing_values' in replacement_rule else simple_imputer_default.missing_values
                , strategy = replacement_rule['strategy'] if 'strategy' in replacement_rule else simple_imputer_default.strategy
                , fill_value = replacement_rule['fill_value'] if 'fill_value' in replacement_rule else simple_imputer_default.fill_value
            )
            if self.verbose:
                print(simple_imputer)
            self.simple_imputers[feat_with_vals_to_replace] = simple_imputer
            
            if simple_imputer.missing_values is np.nan:
                self.columns_replacement_index_dict[feat_with_vals_to_replace] = X[X[feat_with_vals_to_replace].isnull()==True].index
            else:
                self.columns_replacement_index_dict[feat_with_vals_to_replace] = X.loc[X[feat_with_vals_to_replace]==simple_imputer.missing_values].index
            if self.verbose:
                print(self.columns_replacement_index_dict[feat_with_vals_to_replace])

            if len(self.columns_replacement_index_dict[feat_with_vals_to_replace]) > 0:
                replaced = simple_imputer.fit_transform(X_vals_replaced[[feat_with_vals_to_replace]])
                if self.verbose:
                    print(result)
                X_vals_replaced[feat_with_vals_to_replace] = replaced
                X_vals_replaced[feat_with_vals_to_replace] = X_vals_replaced[feat_with_vals_to_replace].astype(type(simple_imputer.fill_value))
            else:
                if self.verbose:
                    print("missing value not found - nothing replaced")
            
        return X_vals_replaced

    def fit_transform(self, X, y=None): # nothing will be done to y
        self = self.fit(X, y)
        return self.transform(X)


class OneHotEncodingTransformer(object):
    def __init__(self, cat_feats_to_encode, categories_by_feat_idx, error_on_non_existent_col=False):
        self.cat_feats_to_encode = cat_feats_to_encode
        self.categories_by_feat_idx = categories_by_feat_idx
        self.ohe = None
        self.error_on_non_existent_col = error_on_non_existent_col
        self.encoded_columns = None

    def __filter_columns(self, df):
        cols = []
        existing_cols = list(df.columns)
        for col in self.cat_feats_to_encode:
            if col in existing_cols:
                cols.append(col)
        return cols

    def fit(self, X, y=None): # nothing will be done to y
        if not self.error_on_non_existent_col: # get list of existing columns from those passed in
            self.cat_feats_to_encode = self.__filter_columns(X)
        else:
            X.drop(self.cat_feats_to_encode, axis=1) # not in place; done for the sake of error_on_non_existent_col
        self.ohe = OneHotEncoder(categories=self.categories_by_feat_idx, drop='first', sparse=False)
        self.ohe.fit(X[self.cat_feats_to_encode])
        return self

    def transform(self, X):
        X_after_encoding = pd.DataFrame(self.ohe.transform(X[self.cat_feats_to_encode]), columns=self.ohe.get_feature_names(self.cat_feats_to_encode), index=X.index)
        self.encoded_columns = self.cat_feats_to_encode
        return X_after_encoding

    def fit_transform(self, X, y=None): # nothing will be done to y
        self = self.fit(X, y)
        return self.transform(X)