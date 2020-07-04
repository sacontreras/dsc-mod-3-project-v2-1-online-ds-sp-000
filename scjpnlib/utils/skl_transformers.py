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
    def __init__(self, columns_to_drop, error_on_non_existent_col=True):
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
            for col in self.columns_to_drop:
                if col not in X:
                    raise ValueError('Column \''+col+'\' not in X')
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
            for col in self.replacement_rules_dict.keys():
                if col not in X:
                    raise ValueError('Column \''+col+'\' not in X')
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
    def __init__(self, replacement_rules_dict, error_on_non_existent_col=True, verbose=False):
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
            for col in self.replacement_rules_dict.keys():
                if col not in X:
                    raise ValueError('Column \''+col+'\' not in X')
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
    def __init__(self, cat_feats_to_encode, categories_by_feat_idx, error_on_non_existent_col=True, verbose=False):
        self.cat_feats_to_encode = cat_feats_to_encode
        self.categories_by_feat_idx = categories_by_feat_idx
        self.ohe = None
        self.error_on_non_existent_col = error_on_non_existent_col
        self.encoded_columns = None
        self.verbose = verbose

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
            for col in self.cat_feats_to_encode:
                if col not in X:
                    raise ValueError('Column \''+col+'\' not in X')
        self.ohe = OneHotEncoder(categories=self.categories_by_feat_idx, drop='first', sparse=False)
        self.ohe.fit(X[self.cat_feats_to_encode])
        self.encoded_columns = list(self.ohe.get_feature_names(self.cat_feats_to_encode))
        self.feats_not_encoded = list(X.columns)
        for cat_feat_to_encode in self.cat_feats_to_encode:
            self.feats_not_encoded.remove(cat_feat_to_encode)
        # if self.verbose:
        #     print(f"** OneHotEncodingTransformer FIT DEBUG **: cat_feats_to_encode: {self.cat_feats_to_encode if len(self.cat_feats_to_encode) < 100 else 'count = '+str(len(self.cat_feats_to_encode))}")
        #     print(f"** OneHotEncodingTransformer FIT DEBUG **: encoded_columns: {self.encoded_columns if len(self.encoded_columns) < 100 else 'count = '+str(len(self.encoded_columns))}")
        #     print(f"** OneHotEncodingTransformer FIT DEBUG **: feats_not_encoded: {self.feats_not_encoded if len(self.feats_not_encoded) < 100 else 'count = '+str(len(self.feats_not_encoded))}")
        return self

    def transform(self, X):
        cols = self.feats_not_encoded.copy()
        cols.extend(self.encoded_columns)
        ohe_result = self.ohe.transform(X[self.cat_feats_to_encode])
        X_after_encoding = pd.concat(
            [
                X[self.feats_not_encoded],
                pd.DataFrame(ohe_result, columns=self.encoded_columns, index=X.index)
            ], 
            axis=1,
            join='inner'
        )
        # Note that obviously the original feat (that was one-hot encoded) is NOT in the transformed data set
        X_after_encoding.columns = cols
        return X_after_encoding

    def fit_transform(self, X, y=None): # nothing will be done to y
        self = self.fit(X, y)
        return self.transform(X)


class LabelEncodingTransformer(object):
    def __init__(self, cat_feats_to_encode, error_on_non_existent_col=True):
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
            for col in self.cat_feats_to_encode:
                if col not in X:
                    raise ValueError('Column \''+col+'\' not in X')
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


# adapted from https://github.com/brendanhasz/target-encoding/blob/master/Target_Encoding.ipynb
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
    
    def __init__(self, n_splits=3, shuffle=True, cols=None, post_encode_null_to_global_mean=True, verbose=False):
        """Leave-one-out target encoding for categorical features.
        
        Parameters
        ----------
        cols : list of str
            Columns to target encode.
        """
        self.cols = cols
        self.verbose = verbose
        self.post_encode_null_to_global_mean = post_encode_null_to_global_mean
        

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
        self.idx = {}
        for col in self.cols:
            self.sum_count[col] = dict()
            
            # S.C.
            self.idx[col] = {}
            
            unique_cats = X[col].unique()
            for cat in unique_cats:
                col_cat_mask = X[col]==cat
                
                # S.C.
                self.idx[col][cat] = X[col_cat_mask].index
                
                self.sum_count[col][cat] = (y[col_cat_mask].sum(), col_cat_mask.sum())
            
        # S.C.: add global mean
        self.target_global_mean = y.mean()

        if self.verbose:
            print(f"** TargetEncoderLOOTransformer FIT INFO **: transformer has been fit to X")
            
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
        
        self.transform_null_index = {}
        self.transform_unique_cats = {}
        self.transform_unfit_cats = {}

        # Use normal target encoding if this is test data
        if y is None:
            if self.verbose:
                print(f"** TargetEncoderLOOTransformer TRANSFORM INFO **: NOT using Leave-One-Out")
            
            for col in self.sum_count:
                # S.C.: added to track null index after target encoding for this col
                self.transform_null_index[col] = pd.Index([])
                
                vals = np.full(X.shape[0], np.nan)
                
                unique_cats = sorted(list(X[col].unique()))
                self.transform_unique_cats[col] = unique_cats
                unfit_cats = sorted(list(set(unique_cats) - set(self.sum_count[col].keys())))
                self.transform_unfit_cats[col] = unfit_cats
                if self.verbose:
                    if  len(unfit_cats) > 0:
                        print(f"** TargetEncoderLOOTransformer TRANSFORM WARNING!! **: {len(unfit_cats)} categories of '{col}' occur in X (out of {len(unique_cats)} unique) that do not exist in the set of fit categories - modeled accuracy on X will drop as a result")
                    else:
                        print(f"** TargetEncoderLOOTransformer TRANSFORM INFO **: unique categories of '{col}' in X match those that were previously fit")
                
                for cat, sum_count in self.sum_count[col].items():
                    col_cat_mask = X[col]==cat
                    
                    if self.verbose and sum_count[1] == 0:
                        print(f"** TargetEncoderLOOTransformer TRANSFORM DEBUG **: sum_count[1]==0 for category '{cat}'")
                    
                    vals[col_cat_mask] = sum_count[0]/sum_count[1]
                    
                Xo[col] = vals
                
                # S.C.: replace null with target global mean
                null_mask = Xo[col].isna()==True
                n_null = len(Xo[null_mask])
                if n_null > 0:
                    self.transform_null_index[col] = Xo[null_mask].index
                    if self.verbose:
                        s_warning = f"** TargetEncoderLOOTransformer TRANSFORM WARNING!! **: feat '{col}' has {n_null} nan values after target encoding"
                        s_warning += f"; replacing these with last fit target global mean: {self.target_global_mean}" if self.post_encode_null_to_global_mean else " but post_encode_null_to_global_mean is False"
                        print(s_warning)
                    if self.post_encode_null_to_global_mean:
                        Xo[col].fillna(self.target_global_mean, inplace=True)

        # LOO target encode each column
        else:
            if self.verbose:
                print(f"** TargetEncoderLOOTransformer TRANSFORM INFO **: using Leave-One-Out")
            
            _classes = y.unique()
            
            for col in self.sum_count:
                # S.C.: added to track null index after target encoding for this col
                self.transform_null_index[col] = pd.Index([])
                
                vals = np.full(X.shape[0], np.nan)
                
                unique_cats = sorted(list(X[col].unique()))
                self.transform_unique_cats[col] = unique_cats
                unfit_cats = sorted(list(set(unique_cats) - set(self.sum_count[col].keys())))
                self.transform_unfit_cats[col] = unfit_cats
                if self.verbose:
                    if len(unfit_cats) > 0:
                        print(f"** TargetEncoderLOOTransformer TRANSFORM WARNING!! **: {len(unfit_cats)} categories of '{col}' occur in X (out of {len(unique_cats)} unique) that do not exist in the set of fit categories - modeled accuracy on X will drop as a result")
                    else:
                        print(f"** TargetEncoderLOOTransformer TRANSFORM INFO **: unique categories of '{col}' in X match those that were previously fit")
                
                for cat, sum_count in self.sum_count[col].items():
                    col_cat_mask = X[col]==cat
                    
                    # for debug only
                    if self.verbose and sum_count[1] == 1:
                        print(f"** TargetEncoderLOOTransformer TRANSFORM DEBUG **: for col '{col}', category '{cat}', sum_count[1] == 1")
                        for _class in _classes:
                            col_cat_class_mask = y[(col_cat_mask)]==_class
                            print(f"\tclass {_class} count: {len(col_cat_class_mask)}")
                        
                    vals[col_cat_mask] = (sum_count[0]-y[col_cat_mask])/(sum_count[1]-1)
                Xo[col] = vals
                
                # S.C.: replace null with target global mean
                n_null = len(Xo[Xo[col].isna()==True])
                if n_null > 0:
                    self.transform_null_index[col] = Xo[Xo[col].isna()==True].index
                    s_warning = f"** TargetEncoderLOOTransformer TRANSFORM WARNING!! **: feat '{col}' has {n_null} nan values after target encoding"
                    s_warning += f"; replacing these with last fit target global mean: {self.target_global_mean}" if self.post_encode_null_to_global_mean else " but post_encode_null_to_global_mean is False"
                    print(s_warning)
                    if self.post_encode_null_to_global_mean:
                        Xo[col].fillna(self.target_global_mean, inplace=True)
            
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

# helper functions
def fit_target_encoder(feat, X, y_target_label_encoded, post_encode_null_to_global_mean=True, verbose=False):
    target_encoder = TargetEncoderLOOTransformer(cols=[feat], post_encode_null_to_global_mean=post_encode_null_to_global_mean, verbose=verbose)
    return target_encoder.fit(X, y_target_label_encoded)

def target_encoder_transform(target_encoder, feat, X, y_target_label_encoded=None):
    X_feat_encoded = target_encoder.transform(X, y_target_label_encoded)
    feat_target_encoded = f"{feat}_target_encoded"
    X_feat_encoded[feat_target_encoded] = X_feat_encoded[feat]
    X_feat_encoded[feat] = X[feat]
    if target_encoder.verbose:
        print(f"added new feature: {feat_target_encoded}")
    return X_feat_encoded


class TargetEncoderKFoldTransformer(TargetEncoderTransformer):
    """K-Fold target encoder.
    """
    
    def __init__(self, n_splits=3, shuffle=True, cols=None):
        """K-Fold target encoding for categorical features.
        
        Parameters
        ----------
        cols : list of str
            Columns to target encode.
        """
        self.cols = cols
        

    def fit(self, X, y):
        """Fit K-Fold target encoder to X and y
        
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