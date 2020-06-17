import pandas as pd
import numpy as np
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.base import BaseEstimator, TransformerMixin

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


class LambdaTransformer(object):
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
            X_case_replaced[feat] = X_case_replaced[feat].apply(f_replacement)
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
        for feat_with_vals_to_replace, replacement_rules in self.replacement_rules_dict.items(): # each item maps to a list of rules for the column
            self.simple_imputers[feat_with_vals_to_replace] = []
            self.columns_replacement_index_dict[feat_with_vals_to_replace] = []
            for replacement_rule in replacement_rules:
                simple_imputer_default = SimpleImputer()
                simple_imputer = SimpleImputer(
                    missing_values = replacement_rule['missing_values'] if 'missing_values' in replacement_rule else simple_imputer_default.missing_values
                    , strategy = replacement_rule['strategy'] if 'strategy' in replacement_rule else simple_imputer_default.strategy
                    , fill_value = replacement_rule['fill_value'] if 'fill_value' in replacement_rule else simple_imputer_default.fill_value
                )
                if self.verbose:
                    print(simple_imputer)
                self.simple_imputers[feat_with_vals_to_replace].append(simple_imputer)
                
                the_index = None
                if simple_imputer.missing_values is np.nan:
                    the_index = X[X[feat_with_vals_to_replace].isnull()==True].index
                else:
                    the_index = X.loc[X[feat_with_vals_to_replace]==simple_imputer.missing_values].index
                self.columns_replacement_index_dict[feat_with_vals_to_replace].append(the_index)
                if self.verbose:
                    print(the_index)

                if len(the_index) > 0:
                    replaced = simple_imputer.fit_transform(X_vals_replaced[[feat_with_vals_to_replace]])
                    # if self.verbose:
                    #     print(replaced)
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
        self.feats_not_encoded = list(X.columns)
        for cat_feat_to_encode in self.cat_feats_to_encode:
            self.feats_not_encoded.remove(cat_feat_to_encode)
        return self

    def transform(self, X):
        # X_after_encoding = pd.DataFrame(self.ohe.transform(X[self.cat_feats_to_encode]), columns=self.ohe.get_feature_names(self.cat_feats_to_encode), index=X.index)
        cols = self.feats_not_encoded.copy()
        cols.extend(self.ohe.get_feature_names(self.cat_feats_to_encode))
        ohe_result = self.ohe.transform(X[self.cat_feats_to_encode])
        X_after_encoding = pd.concat(
            [
                X[self.feats_not_encoded],
                pd.DataFrame(ohe_result, columns=self.ohe.get_feature_names(self.cat_feats_to_encode), index=X.index)
            ], 
            axis=1,
            join='inner'
        )
        X_after_encoding.columns = cols
        self.encoded_columns = self.cat_feats_to_encode
        return X_after_encoding

    def fit_transform(self, X, y=None): # nothing will be done to y
        self = self.fit(X, y)
        return self.transform(X)


class LabelEncodingTransformer(object):
    def __init__(self, cat_feats_to_encode, error_on_non_existent_col=False):
        self.cat_feats_to_encode = cat_feats_to_encode
        self.labelencoder = None
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
        self.labelencoder = LabelEncoder()
        self.labelencoder.fit(X[self.cat_feats_to_encode].values.ravel())
        self.feats_not_encoded = list(X.columns)
        for cat_feat_to_encode in self.cat_feats_to_encode:
            self.feats_not_encoded.remove(cat_feat_to_encode)
        self.df_classes = pd.DataFrame({'class':list(range(len(self.labelencoder.classes_))), 'label':list(self.labelencoder.classes_)})
        self.df_classes = self.df_classes.set_index('class')
        return self

    def transform(self, X):
        cols = self.feats_not_encoded.copy()
        cols.extend(self.cat_feats_to_encode)
        label_encoding_result = self.labelencoder.transform(X[self.cat_feats_to_encode].values.ravel())
        X_after_encoding = pd.concat(
            [
                X[self.feats_not_encoded],
                pd.DataFrame(label_encoding_result, columns=self.cat_feats_to_encode, index=X.index)
            ], 
            axis=1,
            join='inner'
        )
        X_after_encoding.columns = cols
        self.encoded_columns = self.cat_feats_to_encode
        return X_after_encoding

    def fit_transform(self, X, y=None): # nothing will be done to y
        self = self.fit(X, y)
        return self.transform(X)


# referenced from https://github.com/brendanhasz/target-encoding/blob/master/Target_Encoding.ipynb
class TargetEncoderTransformer(BaseEstimator, TransformerMixin):
    """Target encoder.
    
    Replaces categorical column(s) with the mean target value for
    each category.

    """
    
    def __init__(self, cols=None):
        """Target encoder
        
        Parameters
        ----------
        cols : list of str
            Columns to target encode.  Default is to target encode all 
            categorical columns in the DataFrame.
        """
        if isinstance(cols, str):
            self.cols = [cols]
        else:
            self.cols = cols
        
        
    def fit(self, X, y):
        """Fit target encoder to X and y
        
        Parameters
        ----------
        X : pandas DataFrame, shape [n_samples, n_columns]
            DataFrame containing columns to encode
        y : pandas Series, shape = [n_samples]
            Target values.
            
        Returns
        -------
        self : encoder
            Returns self.
        """
        
        # Encode all categorical cols by default
        if self.cols is None:
            self.cols = [col for col in X 
                         if str(X[col].dtype)=='object']

        # Check columns are in X
        for col in self.cols:
            if col not in X:
                raise ValueError('Column \''+col+'\' not in X')

        # Encode each element of each column
        self.maps = dict() #dict to store map for each column
        for col in self.cols:
            tmap = dict()
            uniques = X[col].unique()
            for unique in uniques:
                tmap[unique] = y[X[col]==unique].mean()
            self.maps[col] = tmap
            
        return self

        
    def transform(self, X, y=None):
        """Perform the target encoding transformation.
        
        Parameters
        ----------
        X : pandas DataFrame, shape [n_samples, n_columns]
            DataFrame containing columns to encode
            
        Returns
        -------
        pandas DataFrame
            Input DataFrame with transformed columns
        """
        Xo = X.copy()
        for col, tmap in self.maps.items():
            vals = np.full(X.shape[0], np.nan)
            for val, mean_target in tmap.items():
                vals[X[col]==val] = mean_target
            Xo[col] = vals
        return Xo
            
            
    def fit_transform(self, X, y=None):
        """Fit and transform the data via target encoding.
        
        Parameters
        ----------
        X : pandas DataFrame, shape [n_samples, n_columns]
            DataFrame containing columns to encode
        y : pandas Series, shape = [n_samples]
            Target values (required!).

        Returns
        -------
        pandas DataFrame
            Input DataFrame with transformed columns
        """
        return self.fit(X, y).transform(X, y)


# referenced from https://github.com/brendanhasz/target-encoding/blob/master/Target_Encoding.ipynb
class TargetEncoderLOOTransformer(TargetEncoderTransformer):
    """Leave-one-out target encoder.
    """
    
    def __init__(self, n_splits=3, shuffle=True, cols=None):
        """Leave-one-out target encoding for categorical features.
        
        Parameters
        ----------
        cols : list of str
            Columns to target encode.
        """
        self.cols = cols
        

    def fit(self, X, y):
        """Fit leave-one-out target encoder to X and y
        
        Parameters
        ----------
        X : pandas DataFrame, shape [n_samples, n_columns]
            DataFrame containing columns to target encode
        y : pandas Series, shape = [n_samples]
            Target values.
            
        Returns
        -------
        self : encoder
            Returns self.
        """
        
        # Encode all categorical cols by default
        if self.cols is None:
            self.cols = [col for col in X if str(X[col].dtype)=='object']

        # Check columns are in X
        for col in self.cols:
            if col not in X:
                raise ValueError('Column \''+col+'\' not in X')

        # Encode each element of each column
        self.sum_count = dict() #dict for sum + counts for each column
        for col in self.cols:
            self.sum_count[col] = dict()
            uniques = X[col].unique()
            for unique in uniques:
                ix = X[col]==unique
                self.sum_count[col][unique] = (y[ix].sum(), ix.sum())
            
        # Return the fit object
        return self

    def transform(self, X, y=None):
        """Perform the target encoding transformation.

        Uses leave-one-out target encoding for the training fold, and
        uses normal target encoding for the test fold.

        Parameters
        ----------
        X : pandas DataFrame, shape [n_samples, n_columns]
            DataFrame containing columns to encode

        Returns
        -------
        pandas DataFrame
            Input DataFrame with transformed columns
        """
        
        # Create output dataframe
        Xo = X.copy()

        # Use normal target encoding if this is test data
        if y is None:
            for col in self.sum_count:
                vals = np.full(X.shape[0], np.nan)
                for cat, sum_count in self.sum_count[col].items():
                    vals[X[col]==cat] = sum_count[0]/sum_count[1]
                Xo[col] = vals

        # LOO target encode each column
        else:
            for col in self.sum_count:
                vals = np.full(X.shape[0], np.nan)
                for cat, sum_count in self.sum_count[col].items():
                    ix = X[col]==cat
                    vals[ix] = (sum_count[0]-y[ix])/(sum_count[1]-1)
                Xo[col] = vals
            
        # Return encoded DataFrame
        return Xo
      
            
    def fit_transform(self, X, y=None):
        """Fit and transform the data via target encoding.
        
        Parameters
        ----------
        X : pandas DataFrame, shape [n_samples, n_columns]
            DataFrame containing columns to encode
        y : pandas Series, shape = [n_samples]
            Target values (required!).

        Returns
        -------
        pandas DataFrame
            Input DataFrame with transformed columns
        """
        return self.fit(X, y).transform(X, y)