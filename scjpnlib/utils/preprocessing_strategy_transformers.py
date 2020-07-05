import importlib
from abc import ABC, abstractmethod
from . import impute_TO_nan, impute_TO_lcase, find_and_impute, analyze_outliers_detailed, convert_col_to_date_type, convert_col_type, analyze_distributions__top_n
from sklearn.preprocessing import FunctionTransformer
from .skl_transformers import fit_target_encoder, target_encoder_transform, DropColumnsTransformer, SimpleValueTransformer, OneHotEncodingTransformer
from .submodels import tfidf_fit, tfidf_transform, tfidf_kmeans_classify_feature__fit, tfidf_kmeans_classify_feature__transform
from IPython.core.display import display, HTML, Markdown
import numpy as np
import inspect
import json
import io




# note that since these are instantiated through the instantiate_strategy_transfomer() method
#   by specifying the descendant as a string (in the EDA config file), all descendant CBaseStrategyTransformer
#   class constructors MUST have the signature (self, feat, pipeline_data_preprocessor, verbose)
class CBaseStrategyTransformer():
    def __init__(self, feat, pipeline_data_preprocessor, description, verbose=False):
        self.feat = feat
        self.transformed_feat_name = feat
        self.transformer = None
        self.pipeline_data_preprocessor = pipeline_data_preprocessor
        self.pipeline_step = None
        self.description = description
        self.verbose = verbose

    @abstractmethod
    def get_transformer(self, X, y=None): # fitting should occur in the override
        pass

    def _get_transformed_feat_name(self):
        return self.transformed_feat_name

    def _set_transformed_feat_name(self, transformed_feat_name):
        self.transformed_feat_name = transformed_feat_name

    def _append_pipeline(self):
        if self.pipeline_data_preprocessor is not None:
            self.pipeline_data_preprocessor.steps.append([self.description, self.transformer])
            self.pipeline_step = self.pipeline_data_preprocessor.steps[-1]
            if self.verbose:
                print(f"strategy appended step {self.pipeline_step} to pipeline")

        return self

    def fit(self, X, y=None):
        self.transformer = self.get_transformer(X, y)
        return self._append_pipeline()

    def transform(self, X):
        X_transformed = self.pipeline_step[1].transform(X) if self.pipeline_step is not None else self.transformer.fit_transform(X)
        if self.verbose:
            print(f"strategy \"{self.description}\" transformation is COMPLETE!")
        return X_transformed

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)




class C__leave_it_as_is__StrategyTransformer(CBaseStrategyTransformer):
    def __init__(self, feat, pipeline_data_preprocessor, verbose=False):
        super(C__leave_it_as_is__StrategyTransformer, self).__init__(
            feat, 
            pipeline_data_preprocessor, 
            description=f"leave feature as is (do nothing): {feat}",
            verbose=verbose
        )

    def get_transformer(self, X, y=None):
        return FunctionTransformer(lambda X: X, validate=False)




class C__drop_it__StrategyTransformer(CBaseStrategyTransformer):
    def __init__(self, feat, pipeline_data_preprocessor, verbose=False):
        super(C__drop_it__StrategyTransformer, self).__init__(
            feat, 
            pipeline_data_preprocessor, 
            description=f"drop feature: {feat}",
            verbose=verbose
        )

    def get_transformer(self, X, y=None):
        self._set_transformed_feat_name(None)
        return DropColumnsTransformer([self.feat])




class C__value_replacement__StrategyTransformer(CBaseStrategyTransformer):
    def __init__(self, feat, replacement_rules, pipeline_data_preprocessor, verbose=False):
        super(C__value_replacement__StrategyTransformer, self).__init__(
            feat, 
            pipeline_data_preprocessor, 
            description=f"replace values for feature: {feat}",
            verbose=verbose
        )
        self.replacement_rules = {feat: replacement_rules}
        
    def get_transformer(self, X, y=None):
        if self.verbose:
            print(f"strategy \"{self.description}\" replacement_rules:\n{json.dumps(self.replacement_rules, indent=4)}")
        return SimpleValueTransformer(self.replacement_rules)




class C__strip_nonalphanumeric__StrategyTransformer(CBaseStrategyTransformer):
    def __init__(self, feat, pipeline_data_preprocessor, verbose=False):
        super(C__strip_nonalphanumeric__StrategyTransformer, self).__init__(
            feat, 
            pipeline_data_preprocessor, 
            description=f"strip non-alphanumeric: {feat}",
            verbose=verbose
        )

    def get_transformer(self, X, y=None):
        return FunctionTransformer(lambda X: find_and_impute(X, self.feat, to_replace=r"[^a-zA-Z0-9]", replace_with_val=""), validate=False)




# note that we would like to inherit from C__value_replacement__StrategyTransformer here
#   but apparently there is a bug when using sklearn's SimpleImputer (wrapped by SimpleValueTransformer) 
#   to impute nan - a weird rounding error occurs and the resulting replacement 
#   turns out to be some weird floating point number and, in fact, NOT null
#   therefore, we must use the impute_TO_nan function written to handle this special case
class C__replace_value_with_nan__StrategyTransformer(CBaseStrategyTransformer):
    def __init__(self, feat, value_to_replace_with_nan, pipeline_data_preprocessor, verbose=False):
        super(C__replace_value_with_nan__StrategyTransformer, self).__init__(
            feat, 
            pipeline_data_preprocessor, 
            description=f"replace \"{feat}\" values ({value_to_replace_with_nan}) with nan",
            verbose=verbose
        )
        self.value_to_replace_with_nan = value_to_replace_with_nan

    def get_transformer(self, X, y=None):
        return FunctionTransformer(lambda X: impute_TO_nan(X, self.feat, self.value_to_replace_with_nan), validate=False)

class C__replace_0_with_nan__StrategyTransformer(C__replace_value_with_nan__StrategyTransformer):
    def __init__(self, feat, pipeline_data_preprocessor, verbose=False):
        super(C__replace_0_with_nan__StrategyTransformer, self).__init__(
            feat, 
            0,
            pipeline_data_preprocessor, 
            verbose=verbose
        )

    def get_transformer(self, X, y=None):
        return FunctionTransformer(lambda X: impute_TO_nan(X, self.feat, 0), validate=False)




class C__replace_outliers__StrategyTransformer(CBaseStrategyTransformer):
    def __init__(self, feat, replacement_strategy, pipeline_data_preprocessor, verbose=False):
        super(C__replace_outliers__StrategyTransformer, self).__init__(
            feat, 
            pipeline_data_preprocessor, 
            description=f"replace \"{feat}\" outliers with {replacement_strategy}",
            verbose=verbose
        )
        self.replacement_strategy = replacement_strategy
        
    def get_transformer(self, X, y=None):
        _, _, all_replace_outliers_rules = analyze_outliers_detailed(X, '', self.feat, suppress_output=True)
        if self.replacement_strategy in all_replace_outliers_rules:
            replacement_rules = all_replace_outliers_rules[self.replacement_strategy]

            # these can be quite long so it's been explicitly suppressed
            # if self.verbose:
            #     print(f"strategy \"{self.description}\" replacement_rules:\n{json.dumps(replacement_rules, indent=4)}")

            return SimpleValueTransformer(replacement_rules)
        else:
            return FunctionTransformer(lambda X: X, validate=False) # leave it as is

# specializations of C__replace_outliers__StrategyTransformer: differences are based on replacement_strategy
class C__replace_outliers_with_mean__StrategyTransformer(C__replace_outliers__StrategyTransformer):
    def __init__(self, feat, pipeline_data_preprocessor, verbose=False):
        super(C__replace_outliers_with_mean__StrategyTransformer, self).__init__(
            feat, 
            'mean',
            pipeline_data_preprocessor, 
            verbose=verbose
        )
        
class C__replace_outliers_with_median__StrategyTransformer(C__replace_outliers__StrategyTransformer):
    def __init__(self, feat, pipeline_data_preprocessor, verbose=False):
        super(C__replace_outliers_with_median__StrategyTransformer, self).__init__(
            feat, 
            'median',
            pipeline_data_preprocessor, 
            verbose=verbose
        )




class C__target_encode__StrategyTransformer(CBaseStrategyTransformer):
    def __init__(self, feat, leave_one_out, post_encode_null_to_global_mean, pipeline_data_preprocessor, verbose=False):
        super(C__target_encode__StrategyTransformer, self).__init__(
            feat, 
            pipeline_data_preprocessor, 
            description=f"(prefit) target-encoder (LOO=={leave_one_out}, post_encode_null_to_global_mean=={post_encode_null_to_global_mean}) transform: {feat}",
            verbose=verbose
        )
        self.leave_one_out = leave_one_out
        self.post_encode_null_to_global_mean = post_encode_null_to_global_mean

    def get_transformer(self, X, y_encoded):
        self.y_encoded = y_encoded
        self.target_encoder = fit_target_encoder(
            self.feat, 
            X, 
            self.y_encoded, 
            post_encode_null_to_global_mean=self.post_encode_null_to_global_mean,
            verbose=self.verbose
        )
        self._set_transformed_feat_name(f"{self.feat}_target_encoded")

        return FunctionTransformer(
            lambda X: target_encoder_transform(
                self.target_encoder,
                self.feat,
                X
            ), 
            validate=False
        )

    def transform(self, X):
        if self.leave_one_out:
            X_transformed = target_encoder_transform(self.target_encoder, self.feat, X, self.y_encoded)
        else:
            X_transformed = self.pipeline_step[1].transform(X) if self.pipeline_step is not None else self.transformer.fit_transform(X)

        # now add the step to drop the original feature since we have the new target encoded feature (named f"{feat}_target_encoded"
        dct_after_target_encode = DropColumnsTransformer([self.feat])
        pipeline_step = None
        if self.pipeline_data_preprocessor is not None:
            self.pipeline_data_preprocessor.steps.append([f"drop after target encoding: {self.feat}", dct_after_target_encode])
            pipeline_step = self.pipeline_data_preprocessor.steps[-1]
            if self.verbose:
                print(f"strategy appended step {pipeline_step} to pipeline")

        X_transformed = pipeline_step[1].transform(X_transformed) if pipeline_step is not None else dct_after_target_encode.fit_transform(X_transformed)
        if self.verbose:
            print(f"strategy \"{self.description}\" dropped feature '{self.feat}' after target encoding")
            print(f"strategy transformation of feature '{self.feat}' to '{self._get_transformed_feat_name()}' is COMPLETE!")

        return X_transformed

# specializations of C__target_encode__StrategyTransformer: differences are based on leave-one-out, post_encode_null_to_global_mean
class C__target_encode__not_LOO__post_encode_null_to_global_mean__StrategyTransformer(C__target_encode__StrategyTransformer):
    def __init__(self, feat, pipeline_data_preprocessor, verbose=False):
        super(C__target_encode__not_LOO__post_encode_null_to_global_mean__StrategyTransformer, self).__init__(
            feat, 
            leave_one_out=False,
            post_encode_null_to_global_mean=True,
            pipeline_data_preprocessor=pipeline_data_preprocessor, 
            verbose=verbose
        )

class C__target_encode__not_LOO__not_post_encode_null_to_global_mean__StrategyTransformer(C__target_encode__StrategyTransformer):
    def __init__(self, feat, pipeline_data_preprocessor, verbose=False):
        super(C__target_encode__not_LOO__not_post_encode_null_to_global_mean__StrategyTransformer, self).__init__(
            feat, 
            leave_one_out=False,
            post_encode_null_to_global_mean=False,
            pipeline_data_preprocessor=pipeline_data_preprocessor, 
            verbose=verbose
        )

class C__target_encode__LOO__post_encode_null_to_global_mean__StrategyTransformer(C__target_encode__StrategyTransformer):
    def __init__(self, feat, pipeline_data_preprocessor, verbose=False):
        super(C__target_encode__LOO__post_encode_null_to_global_mean__StrategyTransformer, self).__init__(
            feat, 
            leave_one_out=True,
            post_encode_null_to_global_mean=True,
            pipeline_data_preprocessor=pipeline_data_preprocessor, 
            verbose=verbose
        )

class C__target_encode__LOO__not_post_encode_null_to_global_mean__StrategyTransformer(C__target_encode__StrategyTransformer):
    def __init__(self, feat, pipeline_data_preprocessor, verbose=False):
        super(C__target_encode__LOO__not_post_encode_null_to_global_mean__StrategyTransformer, self).__init__(
            feat, 
            leave_one_out=True,
            post_encode_null_to_global_mean=False,
            pipeline_data_preprocessor=pipeline_data_preprocessor, 
            verbose=verbose
        )




class C__impute_lcase__StrategyTransformer(CBaseStrategyTransformer):
    def __init__(self, feat, pipeline_data_preprocessor, verbose=False):
        super(C__impute_lcase__StrategyTransformer, self).__init__(
            feat, 
            pipeline_data_preprocessor, 
            description=f"impute lower-case transform: {feat}",
            verbose=verbose
        )
        
    def get_transformer(self, X, y=None):
        return FunctionTransformer(lambda X: impute_TO_lcase(X, self.feat), validate=False)




class C__convert_string_date_to_datetime__StrategyTransformer(CBaseStrategyTransformer):
    def __init__(self, feat, from_format, pipeline_data_preprocessor, verbose=False):
        super(C__convert_string_date_to_datetime__StrategyTransformer, self).__init__(
            feat, 
            pipeline_data_preprocessor, 
            description=f"convert (from string date format '{from_format}') to datetime type: {feat}",
            verbose=verbose
        )
        self.from_format = from_format
        
    def get_transformer(self, X, y=None):
        return FunctionTransformer(lambda X: convert_col_to_date_type(X, self.feat, self.from_format), validate=False)




class C__convert_to_arbitrary_type__StrategyTransformer(CBaseStrategyTransformer):
    def __init__(self, feat, to_type, pipeline_data_preprocessor, verbose=False):
        super(C__convert_to_arbitrary_type__StrategyTransformer, self).__init__(
            feat, 
            pipeline_data_preprocessor, 
            description=f"convert to {to_type} type: {feat}",
            verbose=verbose
        )
        self.to_type = to_type
        
    def get_transformer(self, X, y=None):
        return FunctionTransformer(lambda X: convert_col_type(X, self.feat, self.to_type), validate=False)




class C__tfidf_normalize__StrategyTransformer(CBaseStrategyTransformer):
    def __init__(self, feat, pipeline_data_preprocessor, verbose=False):
        super(C__tfidf_normalize__StrategyTransformer, self).__init__(
            feat, 
            pipeline_data_preprocessor, 
            description=f"tfidf normalize string-categorical: {feat}",
            verbose=verbose
        )
        self.corpus = None
        self.tfidf = None
        self.tfidf_vectorizer = None
        self.idx_term_map = None
        
    # do the fit here since base fit() wraps it
    def get_transformer(self, X, y=None):
        self.corpus, self.tfidf, self.tfidf_vectorizer, self.idx_term_map = tfidf_fit(
            X, 
            '', 
            self.feat
        )
        return FunctionTransformer(lambda X: tfidf_transform(
                X, 
                '', 
                self.feat, 
                self.tfidf_vectorizer,
                self.idx_term_map
            ), 
            validate=False
        )
    
    
    
class C__tfidf_kmeans_classify__StrategyTransformer(CBaseStrategyTransformer):
    def __init__(self, feat, pipeline_data_preprocessor, verbose=False):
        super(C__tfidf_kmeans_classify__StrategyTransformer, self).__init__(
            feat, 
            pipeline_data_preprocessor, 
            description=f"tfidf kmeans classify high-cardinality string-categorical: {feat}",
            verbose=verbose
        )
        self.corpus = None
        self.tfidf = None
        self.tfidf_vectorizer = None
        self.idx_term_map = None
        self.kmeans = None
        self.df_kmeans_clusters = None
    
    # do the fit here since base fit() wraps it
    def get_transformer(self, X, y=None):
        _, self.corpus, self.tfidf, self.tfidf_vectorizer, self.idx_term_map, self.kmeans, self.df_kmeans_clusters = tfidf_kmeans_classify_feature__fit(
            X, 
            '', 
            self.feat, 
            verbosity=1 if self.verbose else 0
        )
        return FunctionTransformer(lambda X: tfidf_kmeans_classify_feature__transform(
                X, 
                '', 
                self.feat, 
                self.tfidf_vectorizer,
                self.idx_term_map,
                self.kmeans,
                self.df_kmeans_clusters,
                verbosity=1 if self.verbose else 0
            )[0], 
            validate=False
        )




class C__top_n_significance__StrategyTransformer(CBaseStrategyTransformer):
    def __init__(self, feat, top_n, insig_map_to, pipeline_data_preprocessor, verbose=False):
        super(C__top_n_significance__StrategyTransformer, self).__init__(
            feat, 
            pipeline_data_preprocessor, 
            description=f"keep top {top_n} significant (by frequency) categories and replace insignificant with '{insig_map_to}': {feat}",
            verbose=verbose
        )
        self.top_n = top_n
        self.insig_map_to = insig_map_to
        
    def replace_insig_categories(self, X):
        _result = analyze_distributions__top_n(
            X, 
            '', 
            self.feat, 
            top_n=self.top_n, 
            suppress_output=True
        )
        replace_insig_categories_rules = []
        for insig_cat_val in _result[self.top_n]['insig'][0]:
            replace_insig_categories_rules.append({
                'missing_values': insig_cat_val,
                'strategy': 'constant',
                'fill_value': self.insig_map_to
            })
        replace_insig_categories_rules = {self.feat: replace_insig_categories_rules}
        svt_insig_categories = SimpleValueTransformer(replace_insig_categories_rules)
        return svt_insig_categories.fit_transform(X)
    
    def get_transformer(self, X, y=None):
        return FunctionTransformer(lambda X: self.replace_insig_categories(X), validate=False)




class C__OneHotEncode__StrategyTransformer(CBaseStrategyTransformer):
    def __init__(self, feat, override_categories=None, pipeline_data_preprocessor=None, verbose=False):
        super(C__OneHotEncode__StrategyTransformer, self).__init__(
            feat, 
            pipeline_data_preprocessor, 
            description=f"OneHot Encode: {feat}",
            verbose=verbose
        )
        self.override_categories = override_categories

    # do anything involving fit here since base fit() wraps it
    def get_transformer(self, X, y=None):
        if self.override_categories is None:
            self.override_categories = sorted(list(X[self.feat].unique()))
        transformer = OneHotEncodingTransformer(cat_feats_to_encode=[self.feat], categories_by_feat_idx=[self.override_categories], verbose=self.verbose)
        transformer.fit(X)
        self._set_transformed_feat_name(transformer.encoded_columns)
        return transformer    
    
    


class CCompositeStrategyTransformer():
    def __init__(self, description, feat_transformer_sequence, pipeline_data_preprocessor=None, verbose=False):
        self.description = description
        self.feat_transformer_sequence = []
        for feat_transformer in feat_transformer_sequence:
            feat = feat_transformer[0]
            TransformerClass = feat_transformer[1]
            transformerInstance = TransformerClass(feat, pipeline_data_preprocessor, verbose)
            self.feat_transformer_sequence.append(transformerInstance)
        self.pipeline_data_preprocessor = pipeline_data_preprocessor

    # def fit(self, X, y=None):
    #     fit_feat_transformer_sequence = []
    #     for feat_transformer in self.feat_transformer_sequence:
    #         fit_feat_transformer_sequence.append(feat_transformer.fit(X, y))
    #     self.feat_transformer_sequence = fit_feat_transformer_sequence
    #     return self

    def transform(self, X):
        X_transformed = X.copy()
        for feat_transformer in self.feat_transformer_sequence:
            X_transformed = feat_transformer.transform(X_transformed)
        return X_transformed

    def fit_transform(self, X, y=None):
        X_transformed = X.copy()
        for feat_transformer in self.feat_transformer_sequence:
            X_transformed = feat_transformer.fit_transform(X_transformed, y)
        return X_transformed

def coalesce_transformed_feat_names(transformer, lst, debug=False):
    if not isinstance(transformer, CCompositeStrategyTransformer):
        transformed_feat_name = transformer._get_transformed_feat_name()
        if debug:
            print(f"type(transformer._get_transformed_feat_name()): {type(transformed_feat_name)}")
            print(f"{transformer.__class__.__name__}: original feat: {transformer.feat}; transformed feat: {transformed_feat_name}")
        if transformed_feat_name is not None:
            if type(transformed_feat_name) is list or type(transformed_feat_name) is tuple:
                for tfn in transformed_feat_name:
                    lst.append(tfn)
            else:
                lst.append(transformed_feat_name)
    else:
        for contained_transformer in transformer.feat_transformer_sequence:
            coalesce_transformed_feat_names(contained_transformer, lst, debug)



# ******************* API for string (reflection) based instantiation - used in conjunction with invokinng strategies specified in config file: BEGIN *******************

# used for "reflection" instantation - i.e. instantiation via class name (string)
#   this is integral to being able to dynamically switch to a different strategy via the config file
def strategy_transformer_name_to_class(strategy_transformer_class_name):
    return getattr(importlib.import_module("scjpnlib.utils.preprocessing_strategy_transformers"), strategy_transformer_class_name)

class BadCtorSignature(Exception):
    def __init__(self, class_name):
        self.message = f"class {class_name} ctor does not match required signature: self, feat, pipeline_data_preprocessor, verbose"

def instantiate_strategy_transformer(strategy_composition, description, pipeline, verbose=False):
    feat_transformer_sequence = []
    for strategy_component in strategy_composition:
        StratTransformerClass = strategy_transformer_name_to_class(strategy_component[1])

        # check ctor signature - it must match required sig which is 4 args: self, feat, pipeline_data_preprocessor, verbose
        ctor_argspec = inspect.getargspec(StratTransformerClass.__init__)
        if len(ctor_argspec.args) != 4:
            raise(BadCtorSignature(f"{strategy_component[1]}: {ctor_argspec} does not match required arg spec: (self, feat, pipeline_data_preprocessor, verbose)"))

        feat_transformer_sequence.append((strategy_component[0], StratTransformerClass))

    return CCompositeStrategyTransformer(description, feat_transformer_sequence, pipeline, verbose=verbose)

def _html_prettify_strategy_transformer_description(strategy_transformer):
    if isinstance(strategy_transformer, CCompositeStrategyTransformer):
        s_html = f"<b>(composite) strategy name/description: <i><font color='red'>{strategy_transformer.description}</font></i></b>"
        s_html += "<ol>"
        for feat_transformer in strategy_transformer.feat_transformer_sequence:
            s_html += f"<li>{_html_prettify_strategy_transformer_description(feat_transformer)}</li>"
        s_html += "</ol>"
        return s_html
    else:
        return f"<b>strategy description</b>: <i><font color='blue' style='font-size: x-large;'>{strategy_transformer.description}</font></i>"

def html_prettify_strategy_transformer_description(strategy_transformer):
    display(HTML(_html_prettify_strategy_transformer_description(strategy_transformer)))

# ******************* API for string (reflection) based instantiation - used in conjunction with invokinng strategies specified in config file: END *******************

# ******************* Other (useful) APIs: BEGIN *******************
def get_features_affected_by_transformation(X, composite_transformer, baseline_cols, cols_prior, inclusion_desc, suppress_output=False, debug=False):
    # get list of cols transformed by this option
    transformed_cols = []
    coalesce_transformed_feat_names(composite_transformer, transformed_cols, debug)
    transformed_cols = set(transformed_cols)
    
    # filter out any columns that were dropped
    filtered_baseline_cols = sorted(list(filter(lambda feat_name: feat_name in list(X.columns), baseline_cols)))
    if not suppress_output:
        display(HTML(f"<h4>baseline features in X after this transformation:</h4>"))
        display(HTML(f"<pre>{filtered_baseline_cols}</pre>"))
    
    # only include cols in filtered_transformed_feat_names not already in filtered_baseline_cols
    cols_on_filtered_transformed_feat_names_not_in_filtered_baseline_cols = transformed_cols - set(filtered_baseline_cols)
    prior_cols_transformed_not_in_filtered_baseline_cols = set(cols_prior) - set(filtered_baseline_cols)
    # now we need to filter out the ones not in X
    filtered_prior_cols = sorted(list(filter(lambda feat_name: feat_name in list(X.columns), prior_cols_transformed_not_in_filtered_baseline_cols)))
    filtered_transformed_feat_names = sorted(list(filter(lambda feat_name: feat_name in cols_on_filtered_transformed_feat_names_not_in_filtered_baseline_cols, transformed_cols)))
    filtered_transformed_feat_names = sorted(list(filter(lambda feat_name: feat_name in list(X.columns), filtered_transformed_feat_names)))
    filtered_transformed_feat_names = filtered_prior_cols + filtered_transformed_feat_names
    if not suppress_output:
        display(HTML(f"<h4>features (not in baseline) in X after transformation (including best priors):</h4>"))
        display(HTML(f"<pre>{filtered_transformed_feat_names}</pre>"))
    
    baseline_plus_transformed_cols = filtered_baseline_cols
    if len(filtered_transformed_feat_names) > 0:
        baseline_plus_transformed_cols.extend(filtered_transformed_feat_names)
        
    if not suppress_output:
        display(HTML(f"<h4>features {inclusion_desc} model:</h4>"))
        buffer = io.StringIO()
        X[baseline_plus_transformed_cols].info(buf=buffer)
        s_info = buffer.getvalue()
        display(HTML(f"<pre>{s_info}</pre>"))

    return baseline_plus_transformed_cols
# ******************* Other (useful) APIs: END *******************












# Below are strategy transformers that are specific to features
# ************* StrategyTransformers specific to pump_age (and construction_year, date_recorded): BEGIN *************
class C__convert_string_date_to_datetime__date_recorded__StrategyTransformer(C__convert_string_date_to_datetime__StrategyTransformer):
    def __init__(self, not_used_but_req_for_reflection_instantiation=None, pipeline_data_preprocessor=None, verbose=False):
        super(C__convert_string_date_to_datetime__date_recorded__StrategyTransformer, self).__init__(
            'date_recorded',
            from_format="%Y-%m-%d",
            pipeline_data_preprocessor=pipeline_data_preprocessor, 
            verbose=verbose
        )

class C__required_proprocessing__date_recorded__StrategyTransformer(CCompositeStrategyTransformer):
    def __init__(self, not_used_but_req_for_reflection_instantiation=None, pipeline_data_preprocessor=None, verbose=False):
        super(C__required_proprocessing__date_recorded__StrategyTransformer, self).__init__(
            description="required preprocessing for date_recorded", 
            feat_transformer_sequence=[
                ['date_recorded', C__convert_string_date_to_datetime__date_recorded__StrategyTransformer]
            ],
            pipeline_data_preprocessor=pipeline_data_preprocessor, 
            verbose=verbose
        )

# requires C__required_proprocessing__date_recorded__StrategyTransformer to be done first
class C__convert_to_int__date_recorded__StrategyTransformer(C__convert_to_arbitrary_type__StrategyTransformer):
    def __init__(self, not_used_but_req_for_reflection_instantiation=None, pipeline_data_preprocessor=None, verbose=False):
        super(C__convert_to_int__date_recorded__StrategyTransformer, self).__init__(
            'date_recorded',
            to_type='int',
            pipeline_data_preprocessor=pipeline_data_preprocessor, 
            verbose=verbose
        )


class C__replace_0_construction_year_with_date_recorded__StrategyTransformer(CBaseStrategyTransformer):
    def __init__(self, not_used_but_req_for_reflection_instantiation=None, pipeline_data_preprocessor=None, verbose=False):
        super(C__replace_0_construction_year_with_date_recorded__StrategyTransformer, self).__init__(
            'construction_year', 
            pipeline_data_preprocessor, 
            description=f"replace 0 with date_recorded value: construction_year",
            verbose=verbose
        )

    def replace_0_construction_year_with_date_recorded(self, X):
        X_copy = X.copy()
        X_copy_0_construction_year = X_copy[X_copy.construction_year==0]
        X_copy['dt_recorded_yr'] = X_copy.date_recorded.dt.year
        X_copy['construction_year'] = np.where(X_copy['construction_year']==0, X_copy['dt_recorded_yr'], X_copy['construction_year'])
        X_copy = X_copy.drop('dt_recorded_yr', axis=1)
        return X_copy

    def get_transformer(self, X, y=None):
        return FunctionTransformer(lambda X: self.replace_0_construction_year_with_date_recorded(X), validate=False)

class C__convert_string_date_to_datetime__construction_year__StrategyTransformer(C__convert_string_date_to_datetime__StrategyTransformer):
    def __init__(self, not_used_but_req_for_reflection_instantiation=None, pipeline_data_preprocessor=None, verbose=False):
        super(C__convert_string_date_to_datetime__construction_year__StrategyTransformer, self).__init__(
            'construction_year',
            from_format="%Y",
            pipeline_data_preprocessor=pipeline_data_preprocessor, 
            verbose=verbose
        )

class C__required_proprocessing__construction_year__StrategyTransformer(CCompositeStrategyTransformer):
    def __init__(self, not_used_but_req_for_reflection_instantiation=None, pipeline_data_preprocessor=None, verbose=False):
        super(C__required_proprocessing__construction_year__StrategyTransformer, self).__init__(
            description="required preprocessing for construction_year", 
            feat_transformer_sequence=[
                ['construction_year', C__replace_0_construction_year_with_date_recorded__StrategyTransformer],
                ['construction_year', C__convert_string_date_to_datetime__construction_year__StrategyTransformer]
            ],
            pipeline_data_preprocessor=pipeline_data_preprocessor, 
            verbose=verbose
        )

# requires C__required_proprocessing__construction_year__StrategyTransformer to be done first
class C__convert_to_int__construction_year__StrategyTransformer(C__convert_to_arbitrary_type__StrategyTransformer):
    def __init__(self, not_used_but_req_for_reflection_instantiation=None, pipeline_data_preprocessor=None, verbose=False):
        super(C__convert_to_int__construction_year__StrategyTransformer, self).__init__(
            'construction_year',
            to_type='int',
            pipeline_data_preprocessor=pipeline_data_preprocessor, 
            verbose=verbose
        )

class C__create_pump_age_feature_from_date_recorded_and_construction_year__StrategyTransformer(CBaseStrategyTransformer):
    def __init__(self, not_used_but_req_for_reflection_instantiation=None, pipeline_data_preprocessor=None, verbose=False):
        super(C__create_pump_age_feature_from_date_recorded_and_construction_year__StrategyTransformer, self).__init__(
            'pump_age', 
            pipeline_data_preprocessor, 
            description=f"CREATE NEW FEATURE from date_recorded and construction_year: pump_age",
            verbose=verbose
        )

    def add_pump_age_feature(self, X):
        X_copy = X.copy()

        X__pump_age__debug = X_copy[['date_recorded', 'construction_year']].copy()

        # now simply compute date diff (in years)
        X__pump_age__debug['pump_age'] = X__pump_age__debug['date_recorded'].dt.year - X__pump_age__debug['construction_year'].dt.year
        X_copy['pump_age'] = X__pump_age__debug['pump_age']

        return X_copy

    def get_transformer(self, X, y=None):
        return FunctionTransformer(lambda X: self.add_pump_age_feature(X), validate=False)

class C__required_proprocessing__pump_age__StrategyTransformer(CCompositeStrategyTransformer):
    def __init__(self, not_used_but_req_for_reflection_instantiation=None, pipeline_data_preprocessor=None, verbose=False):
        super(C__required_proprocessing__pump_age__StrategyTransformer, self).__init__(
            description="required preprocessing for pump_age", 
            feat_transformer_sequence=[
                ['date_recorded', C__required_proprocessing__date_recorded__StrategyTransformer],
                ['construction_year', C__required_proprocessing__construction_year__StrategyTransformer],
                ['pump_age', C__create_pump_age_feature_from_date_recorded_and_construction_year__StrategyTransformer],
                ['date_recorded', C__drop_it__StrategyTransformer],
                ['construction_year', C__drop_it__StrategyTransformer]
            ],
            pipeline_data_preprocessor=pipeline_data_preprocessor, 
            verbose=verbose
        )
# ************* StrategyTransformers specific to pump_age (and construction_year, date_recorded): END *************


# ************* StrategyTransformers specific to funder: BEGIN *************
class C__impute_lcase__funder__StrategyTransformer(C__impute_lcase__StrategyTransformer):
    def __init__(self, not_used_but_req_for_reflection_instantiation=None, pipeline_data_preprocessor=None, verbose=False):
        super(C__impute_lcase__funder__StrategyTransformer, self).__init__(
            'funder', 
            pipeline_data_preprocessor, 
            verbose=verbose
        )

class C__missing_value_imputer__funder__StrategyTransformer(C__value_replacement__StrategyTransformer):
    def __init__(self, not_used_but_req_for_reflection_instantiation=None, pipeline_data_preprocessor=None, verbose=False):
        super(C__missing_value_imputer__funder__StrategyTransformer, self).__init__(
            'funder', 
            [
                {'missing_values': np.nan, 'strategy': 'constant', 'fill_value': "none"},
                {'missing_values': '0', 'strategy': 'constant', 'fill_value': "none"}
            ],
            pipeline_data_preprocessor, 
            verbose=verbose
        )

class C__not_known_literal_value_replacement__funder__StrategyTransformer(C__value_replacement__StrategyTransformer):
    def __init__(self, not_used_but_req_for_reflection_instantiation=None, pipeline_data_preprocessor=None, verbose=False):
        super(C__not_known_literal_value_replacement__funder__StrategyTransformer, self).__init__(
            'funder', 
            [{'missing_values': 'not known', 'strategy': 'constant', 'fill_value': "unknown"}],
            pipeline_data_preprocessor, 
            verbose=verbose
        )

class C__required_proprocessing__funder__StrategyTransformer(CCompositeStrategyTransformer):
    def __init__(self, not_used_but_req_for_reflection_instantiation=None, pipeline_data_preprocessor=None, verbose=False):
        super(C__required_proprocessing__funder__StrategyTransformer, self).__init__(
            description="required preprocessing for funder", 
            feat_transformer_sequence=[
                ['funder', C__impute_lcase__funder__StrategyTransformer],
                ['funder', C__missing_value_imputer__funder__StrategyTransformer],
                ['funder', C__not_known_literal_value_replacement__funder__StrategyTransformer]
            ],
            pipeline_data_preprocessor=pipeline_data_preprocessor, 
            verbose=verbose
        )
        
class C__tfidf_normalize__funder__StrategyTransformer(C__tfidf_normalize__StrategyTransformer):
    def __init__(self, not_used_but_req_for_reflection_instantiation=None, pipeline_data_preprocessor=None, verbose=False):
        super(C__tfidf_normalize__funder__StrategyTransformer, self).__init__(
            'funder', 
            pipeline_data_preprocessor, 
            verbose=verbose
        )

class C__tfidf_kmeans_classify__funder__StrategyTransformer(C__tfidf_kmeans_classify__StrategyTransformer):
    def __init__(self, not_used_but_req_for_reflection_instantiation=None, pipeline_data_preprocessor=None, verbose=False):
        super(C__tfidf_kmeans_classify__funder__StrategyTransformer, self).__init__(
            'funder', 
            pipeline_data_preprocessor, 
            verbose=verbose
        )

class C__strip_nonalphanumeric__funder__StrategyTransformer(C__strip_nonalphanumeric__StrategyTransformer):
    def __init__(self, not_used_but_req_for_reflection_instantiation=None, pipeline_data_preprocessor=None, verbose=False):
        super(C__strip_nonalphanumeric__funder__StrategyTransformer, self).__init__(
            'funder', 
            pipeline_data_preprocessor, 
            verbose=verbose
        )
        
class C__top_n_significance__funder__StrategyTransformer(C__top_n_significance__StrategyTransformer):
    def __init__(self, not_used_but_req_for_reflection_instantiation=None, pipeline_data_preprocessor=None, verbose=False):
        super(C__top_n_significance__funder__StrategyTransformer, self).__init__(
            'funder',
            top_n=10, # note that this class is highly tailored to this feature and this value may therefore need to be adjusted
            insig_map_to='none',
            pipeline_data_preprocessor=pipeline_data_preprocessor, 
            verbose=verbose
        )
# ************* StrategyTransformers specific to funder: END *************


# ************* StrategyTransformers specific to installer: BEGIN *************
class C__impute_lcase__installer__StrategyTransformer(C__impute_lcase__StrategyTransformer):
    def __init__(self, not_used_but_req_for_reflection_instantiation=None, pipeline_data_preprocessor=None, verbose=False):
        super(C__impute_lcase__installer__StrategyTransformer, self).__init__(
            'installer', 
            pipeline_data_preprocessor, 
            verbose=verbose
        )

class C__missing_value_imputer__installer__StrategyTransformer(C__value_replacement__StrategyTransformer):
    def __init__(self, not_used_but_req_for_reflection_instantiation=None, pipeline_data_preprocessor=None, verbose=False):
        super(C__missing_value_imputer__installer__StrategyTransformer, self).__init__(
            'installer', 
            [
                {'missing_values': np.nan, 'strategy': 'constant', 'fill_value': "none"},
                {'missing_values': '0', 'strategy': 'constant', 'fill_value': "none"},
                {'missing_values': '-', 'strategy': 'constant', 'fill_value': "none"}
            ],
            pipeline_data_preprocessor, 
            verbose=verbose
        )

class C__not_known_literal_value_replacement__installer__StrategyTransformer(C__value_replacement__StrategyTransformer):
    def __init__(self, not_used_but_req_for_reflection_instantiation=None, pipeline_data_preprocessor=None, verbose=False):
        super(C__not_known_literal_value_replacement__installer__StrategyTransformer, self).__init__(
            'installer', 
            [{'missing_values': 'not known', 'strategy': 'constant', 'fill_value': "unknown"}],
            pipeline_data_preprocessor, 
            verbose=verbose
        )

class C__required_proprocessing__installer__StrategyTransformer(CCompositeStrategyTransformer):
    def __init__(self, not_used_but_req_for_reflection_instantiation=None, pipeline_data_preprocessor=None, verbose=False):
        super(C__required_proprocessing__installer__StrategyTransformer, self).__init__(
            description="required preprocessing for installer", 
            feat_transformer_sequence=[
                ['installer', C__impute_lcase__installer__StrategyTransformer],
                ['installer', C__missing_value_imputer__installer__StrategyTransformer],
                ['installer', C__not_known_literal_value_replacement__installer__StrategyTransformer]
            ],
            pipeline_data_preprocessor=pipeline_data_preprocessor, 
            verbose=verbose
        )

class C__tfidf_normalize__installer__StrategyTransformer(C__tfidf_normalize__StrategyTransformer):
    def __init__(self, not_used_but_req_for_reflection_instantiation=None, pipeline_data_preprocessor=None, verbose=False):
        super(C__tfidf_normalize__installer__StrategyTransformer, self).__init__(
            'installer', 
            pipeline_data_preprocessor, 
            verbose=verbose
        )
        
class C__tfidf_kmeans_classify__installer__StrategyTransformer(C__tfidf_kmeans_classify__StrategyTransformer):
    def __init__(self, not_used_but_req_for_reflection_instantiation=None, pipeline_data_preprocessor=None, verbose=False):
        super(C__tfidf_kmeans_classify__installer__StrategyTransformer, self).__init__(
            'installer', 
            pipeline_data_preprocessor, 
            verbose=verbose
        )

class C__strip_nonalphanumeric__installer__StrategyTransformer(C__strip_nonalphanumeric__StrategyTransformer):
    def __init__(self, not_used_but_req_for_reflection_instantiation=None, pipeline_data_preprocessor=None, verbose=False):
        super(C__strip_nonalphanumeric__installer__StrategyTransformer, self).__init__(
            'installer', 
            pipeline_data_preprocessor, 
            verbose=verbose
        )
        
class C__top_n_significance__installer__StrategyTransformer(C__top_n_significance__StrategyTransformer):
    def __init__(self, not_used_but_req_for_reflection_instantiation=None, pipeline_data_preprocessor=None, verbose=False):
        super(C__top_n_significance__installer__StrategyTransformer, self).__init__(
            'installer',
            top_n=10, # note that this class is highly tailored to this feature and this value may therefore need to be adjusted
            insig_map_to='other',
            pipeline_data_preprocessor=pipeline_data_preprocessor, 
            verbose=verbose
        )
    
# ************* StrategyTransformers specific to installer: END *************


# ************* StrategyTransformers specific to gps_coordinates: BEGIN *************
class C__weird_literal_value_replacement__latitude__StrategyTransformer(C__value_replacement__StrategyTransformer):
    def __init__(self, not_used_but_req_for_reflection_instantiation=None, pipeline_data_preprocessor=None, verbose=False):
        super(C__weird_literal_value_replacement__latitude__StrategyTransformer, self).__init__(
            'latitude', 
            [{'missing_values': -2.e-08, 'strategy': 'constant', 'fill_value': 0.0}],
            pipeline_data_preprocessor, 
            verbose=verbose
        )

class C__required_proprocessing__latitude__StrategyTransformer(CCompositeStrategyTransformer):
    def __init__(self, not_used_but_req_for_reflection_instantiation=None, pipeline_data_preprocessor=None, verbose=False):
        super(C__required_proprocessing__latitude__StrategyTransformer, self).__init__(
            description="required preprocessing for latitude", 
            feat_transformer_sequence=[
                ['latitude', C__weird_literal_value_replacement__latitude__StrategyTransformer]
            ],
            pipeline_data_preprocessor=pipeline_data_preprocessor, 
            verbose=verbose
        )
# ************* StrategyTransformers specific to gps_coordinates: END *************


# ************* StrategyTransformers specific to basin: BEGIN *************
class C__impute_lcase__basin__StrategyTransformer(C__impute_lcase__StrategyTransformer):
    def __init__(self, not_used_but_req_for_reflection_instantiation=None, pipeline_data_preprocessor=None, verbose=False):
        super(C__impute_lcase__basin__StrategyTransformer, self).__init__(
            'basin', 
            pipeline_data_preprocessor, 
            verbose=verbose
        )

class C__required_proprocessing__basin__StrategyTransformer(CCompositeStrategyTransformer):
    def __init__(self, not_used_but_req_for_reflection_instantiation=None, pipeline_data_preprocessor=None, verbose=False):
        super(C__required_proprocessing__basin__StrategyTransformer, self).__init__(
            description="required preprocessing for basin", 
            feat_transformer_sequence=[
                ['basin', C__impute_lcase__basin__StrategyTransformer]
            ],
            pipeline_data_preprocessor=pipeline_data_preprocessor, 
            verbose=verbose
        )

class C__strip_nonalphanumeric__basin__StrategyTransformer(C__strip_nonalphanumeric__StrategyTransformer):
    def __init__(self, not_used_but_req_for_reflection_instantiation=None, pipeline_data_preprocessor=None, verbose=False):
        super(C__strip_nonalphanumeric__basin__StrategyTransformer, self).__init__(
            'basin', 
            pipeline_data_preprocessor, 
            verbose=verbose
        )

class _C__OneHotEncode__basin__StrategyTransformer(C__OneHotEncode__StrategyTransformer):
    def __init__(self, not_used_but_req_for_reflection_instantiation=None, pipeline_data_preprocessor=None, verbose=False):
        super(_C__OneHotEncode__basin__StrategyTransformer, self).__init__(
            'basin', 
            override_categories=None,
            pipeline_data_preprocessor=pipeline_data_preprocessor, 
            verbose=verbose
        )

class C__OneHotEncode__basin__StrategyTransformer(CCompositeStrategyTransformer):
    def __init__(self, not_used_but_req_for_reflection_instantiation=None, pipeline_data_preprocessor=None, verbose=False):
        super(C__OneHotEncode__basin__StrategyTransformer, self).__init__(
            description="required preprocessing to OneHot-Encode basin", 
            feat_transformer_sequence=[
                ['basin', C__strip_nonalphanumeric__basin__StrategyTransformer],
                ['basin', _C__OneHotEncode__basin__StrategyTransformer] 
                # note that since _C__OneHotEncode__basin__StrategyTransformer wraps OneHotEncodingTransformer, 
                #   the original feat that was encoded will obviously not be in the transformed data set;
                #   therefore, there is no need to drop it after this
            ],
            pipeline_data_preprocessor=pipeline_data_preprocessor, 
            verbose=verbose
        )
# ************* StrategyTransformers specific to basin: END *************


# ************* StrategyTransformers specific to geographic_location__group: BEGIN *************
# ************* StrategyTransformers specific to region_code: BEGIN *************
class _C__OneHotEncode__region_code__StrategyTransformer(C__OneHotEncode__StrategyTransformer):
    def __init__(self, not_used_but_req_for_reflection_instantiation=None, pipeline_data_preprocessor=None, verbose=False):
        super(_C__OneHotEncode__region_code__StrategyTransformer, self).__init__(
            'region_code', 
            override_categories=None,
            pipeline_data_preprocessor=pipeline_data_preprocessor, 
            verbose=verbose
        )

class C__OneHotEncode__region_code__StrategyTransformer(CCompositeStrategyTransformer):
    def __init__(self, not_used_but_req_for_reflection_instantiation=None, pipeline_data_preprocessor=None, verbose=False):
        super(C__OneHotEncode__region_code__StrategyTransformer, self).__init__(
            description="required preprocessing to OneHot-Encode region_code", 
            feat_transformer_sequence=[
                ['region_code', _C__OneHotEncode__region_code__StrategyTransformer] 
                # note that since _C__OneHotEncode__region_code__StrategyTransformer wraps OneHotEncodingTransformer, 
                #   the original feat that was encoded will obviously not be in the transformed data set;
                #   therefore, there is no need to drop it after this
            ],
            pipeline_data_preprocessor=pipeline_data_preprocessor, 
            verbose=verbose
        )
# ************* StrategyTransformers specific to region_code: BEGIN *************

# ************* StrategyTransformers specific to district_code: BEGIN *************
class _C__OneHotEncode__district_code__StrategyTransformer(C__OneHotEncode__StrategyTransformer):
    def __init__(self, not_used_but_req_for_reflection_instantiation=None, pipeline_data_preprocessor=None, verbose=False):
        super(_C__OneHotEncode__district_code__StrategyTransformer, self).__init__(
            'district_code', 
            override_categories=None,
            pipeline_data_preprocessor=pipeline_data_preprocessor, 
            verbose=verbose
        )

class C__OneHotEncode__district_code__StrategyTransformer(CCompositeStrategyTransformer):
    def __init__(self, not_used_but_req_for_reflection_instantiation=None, pipeline_data_preprocessor=None, verbose=False):
        super(C__OneHotEncode__district_code__StrategyTransformer, self).__init__(
            description="required preprocessing to OneHot-Encode district_code", 
            feat_transformer_sequence=[
                ['district_code', _C__OneHotEncode__district_code__StrategyTransformer] 
                # note that since _C__OneHotEncode__district_code__StrategyTransformer wraps OneHotEncodingTransformer, 
                #   the original feat that was encoded will obviously not be in the transformed data set;
                #   therefore, there is no need to drop it after this
            ],
            pipeline_data_preprocessor=pipeline_data_preprocessor, 
            verbose=verbose
        )
# ************* StrategyTransformers specific to district_code: BEGIN *************

# ************* StrategyTransformers specific to region: BEGIN *************
class C__strip_nonalphanumeric__region__StrategyTransformer(C__strip_nonalphanumeric__StrategyTransformer):
    def __init__(self, not_used_but_req_for_reflection_instantiation=None, pipeline_data_preprocessor=None, verbose=False):
        super(C__strip_nonalphanumeric__region__StrategyTransformer, self).__init__(
            'region', 
            pipeline_data_preprocessor, 
            verbose=verbose
        )

class _C__OneHotEncode__region__StrategyTransformer(C__OneHotEncode__StrategyTransformer):
    def __init__(self, not_used_but_req_for_reflection_instantiation=None, pipeline_data_preprocessor=None, verbose=False):
        super(_C__OneHotEncode__region__StrategyTransformer, self).__init__(
            'region', 
            override_categories=None,
            pipeline_data_preprocessor=pipeline_data_preprocessor, 
            verbose=verbose
        )

class C__OneHotEncode__region__StrategyTransformer(CCompositeStrategyTransformer):
    def __init__(self, not_used_but_req_for_reflection_instantiation=None, pipeline_data_preprocessor=None, verbose=False):
        super(C__OneHotEncode__region__StrategyTransformer, self).__init__(
            description="required preprocessing to OneHot-Encode basin", 
            feat_transformer_sequence=[
                ['region', C__strip_nonalphanumeric__region__StrategyTransformer],
                ['region', _C__OneHotEncode__region__StrategyTransformer] 
                # note that since _C__OneHotEncode__region__StrategyTransformer wraps OneHotEncodingTransformer, 
                #   the original feat that was encoded will obviously not be in the transformed data set;
                #   therefore, there is no need to drop it after this
            ],
            pipeline_data_preprocessor=pipeline_data_preprocessor, 
            verbose=verbose
        )
# ************* StrategyTransformers specific to region: END *************

# ************* StrategyTransformers specific to lga: lga *************
class C__strip_nonalphanumeric__lga__StrategyTransformer(C__strip_nonalphanumeric__StrategyTransformer):
    def __init__(self, not_used_but_req_for_reflection_instantiation=None, pipeline_data_preprocessor=None, verbose=False):
        super(C__strip_nonalphanumeric__lga__StrategyTransformer, self).__init__(
            'lga', 
            pipeline_data_preprocessor, 
            verbose=verbose
        )

class _C__OneHotEncode__lga__StrategyTransformer(C__OneHotEncode__StrategyTransformer):
    def __init__(self, not_used_but_req_for_reflection_instantiation=None, pipeline_data_preprocessor=None, verbose=False):
        super(_C__OneHotEncode__lga__StrategyTransformer, self).__init__(
            'lga', 
            override_categories=None,
            pipeline_data_preprocessor=pipeline_data_preprocessor, 
            verbose=verbose
        )

class C__OneHotEncode__lga__StrategyTransformer(CCompositeStrategyTransformer):
    def __init__(self, not_used_but_req_for_reflection_instantiation=None, pipeline_data_preprocessor=None, verbose=False):
        super(C__OneHotEncode__lga__StrategyTransformer, self).__init__(
            description="required preprocessing to OneHot-Encode lga", 
            feat_transformer_sequence=[
                ['lga', C__strip_nonalphanumeric__lga__StrategyTransformer],
                ['lga', _C__OneHotEncode__lga__StrategyTransformer] 
                # note that since _C__OneHotEncode__region__StrategyTransformer wraps OneHotEncodingTransformer, 
                #   the original feat that was encoded will obviously not be in the transformed data set;
                #   therefore, there is no need to drop it after this
            ],
            pipeline_data_preprocessor=pipeline_data_preprocessor, 
            verbose=verbose
        )
# ************* StrategyTransformers specific to lga: END *************

# ************* StrategyTransformers specific to ward: BEGIN *************
class C__tfidf_normalize__ward__StrategyTransformer(C__tfidf_normalize__StrategyTransformer):
    def __init__(self, not_used_but_req_for_reflection_instantiation=None, pipeline_data_preprocessor=None, verbose=False):
        super(C__tfidf_normalize__ward__StrategyTransformer, self).__init__(
            'ward', 
            pipeline_data_preprocessor, 
            verbose=verbose
        )
        
class C__top_n_significance__ward__StrategyTransformer(C__top_n_significance__StrategyTransformer):
    def __init__(self, not_used_but_req_for_reflection_instantiation=None, pipeline_data_preprocessor=None, verbose=False):
        super(C__top_n_significance__ward__StrategyTransformer, self).__init__(
            'ward',
            top_n=10, # note that this class is highly tailored to this feature and this value may therefore need to be adjusted
            insig_map_to='other',
            pipeline_data_preprocessor=pipeline_data_preprocessor, 
            verbose=verbose
        )
# ************* StrategyTransformers specific to ward: END *************

# ************* StrategyTransformers specific to subvillage: BEGIN *************
class C__impute_lcase__subvillage__StrategyTransformer(C__impute_lcase__StrategyTransformer):
    def __init__(self, not_used_but_req_for_reflection_instantiation=None, pipeline_data_preprocessor=None, verbose=False):
        super(C__impute_lcase__subvillage__StrategyTransformer, self).__init__(
            'subvillage', 
            pipeline_data_preprocessor, 
            verbose=verbose
        )
        
class C__missing_value_imputer__subvillage__StrategyTransformer(C__value_replacement__StrategyTransformer):
    def __init__(self, not_used_but_req_for_reflection_instantiation=None, pipeline_data_preprocessor=None, verbose=False):
        super(C__missing_value_imputer__subvillage__StrategyTransformer, self).__init__(
            'subvillage', 
            [{'missing_values': np.nan, 'strategy': 'constant', 'fill_value': 'unknown'}],
            pipeline_data_preprocessor, 
            verbose=verbose
        )
        
class C__required_proprocessing__subvillage__StrategyTransformer(CCompositeStrategyTransformer):
    def __init__(self, not_used_but_req_for_reflection_instantiation=None, pipeline_data_preprocessor=None, verbose=False):
        super(C__required_proprocessing__subvillage__StrategyTransformer, self).__init__(
            description="required preprocessing for subvillage", 
            feat_transformer_sequence=[
                ['subvillage', C__missing_value_imputer__subvillage__StrategyTransformer],
                ['subvillage', C__impute_lcase__subvillage__StrategyTransformer]
            ],
            pipeline_data_preprocessor=pipeline_data_preprocessor, 
            verbose=verbose
        )
        
class C__top_n_significance__subvillage__StrategyTransformer(C__top_n_significance__StrategyTransformer):
    def __init__(self, not_used_but_req_for_reflection_instantiation=None, pipeline_data_preprocessor=None, verbose=False):
        super(C__top_n_significance__subvillage__StrategyTransformer, self).__init__(
            'subvillage',
            top_n=10, # note that this class is highly tailored to this feature and this value may therefore need to be adjusted
            insig_map_to='other',
            pipeline_data_preprocessor=pipeline_data_preprocessor, 
            verbose=verbose
        )
# ************* StrategyTransformers specific to subvillage: BEGIN *************
# ************* StrategyTransformers specific to geographic_location__group: BEGIN *************


# ************* StrategyTransformers specific to public_meeting: BEGIN *************
class C__missing_value_imputer__public_meeting__StrategyTransformer(C__value_replacement__StrategyTransformer):
    def __init__(self, not_used_but_req_for_reflection_instantiation=None, pipeline_data_preprocessor=None, verbose=False):
        super(C__missing_value_imputer__public_meeting__StrategyTransformer, self).__init__(
            'public_meeting', 
            [{'missing_values': np.nan, 'strategy': 'constant', 'fill_value': False}],
            pipeline_data_preprocessor, 
            verbose=verbose
        )

class C__required_proprocessing__public_meeting__StrategyTransformer(CCompositeStrategyTransformer):
    def __init__(self, not_used_but_req_for_reflection_instantiation=None, pipeline_data_preprocessor=None, verbose=False):
        super(C__required_proprocessing__public_meeting__StrategyTransformer, self).__init__(
            description="required preprocessing for public_meeting", 
            feat_transformer_sequence=[
                ['public_meeting', C__missing_value_imputer__public_meeting__StrategyTransformer]
            ],
            pipeline_data_preprocessor=pipeline_data_preprocessor, 
            verbose=verbose
        )
# ************* StrategyTransformers specific to public_meeting: END *************


# ************* StrategyTransformers specific to wpt_operator: BEGIN *************
# ************* StrategyTransformers specific to scheme_management: BEGIN *************
class C__impute_lcase__scheme_management__StrategyTransformer(C__impute_lcase__StrategyTransformer):
    def __init__(self, not_used_but_req_for_reflection_instantiation=None, pipeline_data_preprocessor=None, verbose=False):
        super(C__impute_lcase__scheme_management__StrategyTransformer, self).__init__(
            'scheme_management', 
            pipeline_data_preprocessor, 
            verbose=verbose
        )

class C__missing_value_imputer__scheme_management__StrategyTransformer(C__value_replacement__StrategyTransformer):
    def __init__(self, not_used_but_req_for_reflection_instantiation=None, pipeline_data_preprocessor=None, verbose=False):
        super(C__missing_value_imputer__scheme_management__StrategyTransformer, self).__init__(
            'scheme_management', 
            [{'missing_values': np.nan, 'strategy': 'constant', 'fill_value': 'none'}],
            pipeline_data_preprocessor, 
            verbose=verbose
        )

class C__required_proprocessing__scheme_management__StrategyTransformer(CCompositeStrategyTransformer):
    def __init__(self, not_used_but_req_for_reflection_instantiation=None, pipeline_data_preprocessor=None, verbose=False):
        super(C__required_proprocessing__scheme_management__StrategyTransformer, self).__init__(
            description="required preprocessing for scheme_management", 
            feat_transformer_sequence=[
                ['scheme_management', C__impute_lcase__scheme_management__StrategyTransformer],
                ['scheme_management', C__missing_value_imputer__scheme_management__StrategyTransformer]
            ],
            pipeline_data_preprocessor=pipeline_data_preprocessor, 
            verbose=verbose
        )

class C__strip_nonalphanumeric__scheme_management__StrategyTransformer(C__strip_nonalphanumeric__StrategyTransformer):
    def __init__(self, not_used_but_req_for_reflection_instantiation=None, pipeline_data_preprocessor=None, verbose=False):
        super(C__strip_nonalphanumeric__scheme_management__StrategyTransformer, self).__init__(
            'scheme_management', 
            pipeline_data_preprocessor, 
            verbose=verbose
        )

class _C__OneHotEncode__scheme_management__StrategyTransformer(C__OneHotEncode__StrategyTransformer):
    def __init__(self, not_used_but_req_for_reflection_instantiation=None, pipeline_data_preprocessor=None, verbose=False):
        super(_C__OneHotEncode__scheme_management__StrategyTransformer, self).__init__(
            'scheme_management', 
            override_categories=None,
            pipeline_data_preprocessor=pipeline_data_preprocessor, 
            verbose=verbose
        )

class C__OneHotEncode__scheme_management__StrategyTransformer(CCompositeStrategyTransformer):
    def __init__(self, not_used_but_req_for_reflection_instantiation=None, pipeline_data_preprocessor=None, verbose=False):
        super(C__OneHotEncode__scheme_management__StrategyTransformer, self).__init__(
            description="required preprocessing to OneHot-Encode scheme_management", 
            feat_transformer_sequence=[
                ['scheme_management', C__strip_nonalphanumeric__scheme_management__StrategyTransformer],
                ['scheme_management', _C__OneHotEncode__scheme_management__StrategyTransformer] 
                # note that since _C__OneHotEncode__scheme_management__StrategyTransformer wraps OneHotEncodingTransformer, 
                #   the original feat that was encoded will obviously not be in the transformed data set;
                #   therefore, there is no need to drop it after this
            ],
            pipeline_data_preprocessor=pipeline_data_preprocessor, 
            verbose=verbose
        )
# ************* StrategyTransformers specific to scheme_management: END *************

# ************* StrategyTransformers specific to scheme_name: BEGIN *************
class C__impute_lcase__scheme_name__StrategyTransformer(C__impute_lcase__StrategyTransformer):
    def __init__(self, not_used_but_req_for_reflection_instantiation=None, pipeline_data_preprocessor=None, verbose=False):
        super(C__impute_lcase__scheme_name__StrategyTransformer, self).__init__(
            'scheme_name', 
            pipeline_data_preprocessor, 
            verbose=verbose
        )

class C__missing_value_imputer__scheme_name__StrategyTransformer(C__value_replacement__StrategyTransformer):
    def __init__(self, not_used_but_req_for_reflection_instantiation=None, pipeline_data_preprocessor=None, verbose=False):
        super(C__missing_value_imputer__scheme_name__StrategyTransformer, self).__init__(
            'scheme_name', 
            [{'missing_values': np.nan, 'strategy': 'constant', 'fill_value': 'none'}],
            pipeline_data_preprocessor, 
            verbose=verbose
        )

class C__not_known_literal_value_replacement__scheme_name__StrategyTransformer(C__value_replacement__StrategyTransformer):
    def __init__(self, not_used_but_req_for_reflection_instantiation=None, pipeline_data_preprocessor=None, verbose=False):
        super(C__not_known_literal_value_replacement__scheme_name__StrategyTransformer, self).__init__(
            'scheme_name', 
            [{'missing_values': 'not known', 'strategy': 'constant', 'fill_value': "unknown"}],
            pipeline_data_preprocessor, 
            verbose=verbose
        )

class C__required_proprocessing__scheme_name__StrategyTransformer(CCompositeStrategyTransformer):
    def __init__(self, not_used_but_req_for_reflection_instantiation=None, pipeline_data_preprocessor=None, verbose=False):
        super(C__required_proprocessing__scheme_name__StrategyTransformer, self).__init__(
            description="required preprocessing for scheme_name", 
            feat_transformer_sequence=[
                ['scheme_name', C__impute_lcase__scheme_name__StrategyTransformer],
                ['scheme_name', C__missing_value_imputer__scheme_name__StrategyTransformer],
                ['scheme_name', C__not_known_literal_value_replacement__scheme_name__StrategyTransformer]
            ],
            pipeline_data_preprocessor=pipeline_data_preprocessor, 
            verbose=verbose
        )

class C__strip_nonalphanumeric__scheme_name__StrategyTransformer(C__strip_nonalphanumeric__StrategyTransformer):
    def __init__(self, not_used_but_req_for_reflection_instantiation=None, pipeline_data_preprocessor=None, verbose=False):
        super(C__strip_nonalphanumeric__scheme_name__StrategyTransformer, self).__init__(
            'scheme_name', 
            pipeline_data_preprocessor, 
            verbose=verbose
        )
        
class C__tfidf_normalize__scheme_name__StrategyTransformer(C__tfidf_normalize__StrategyTransformer):
    def __init__(self, not_used_but_req_for_reflection_instantiation=None, pipeline_data_preprocessor=None, verbose=False):
        super(C__tfidf_normalize__scheme_name__StrategyTransformer, self).__init__(
            'scheme_name', 
            pipeline_data_preprocessor, 
            verbose=verbose
        )

class C__tfidf_kmeans_classify__scheme_name__StrategyTransformer(C__tfidf_kmeans_classify__StrategyTransformer):
    def __init__(self, not_used_but_req_for_reflection_instantiation=None, pipeline_data_preprocessor=None, verbose=False):
        super(C__tfidf_kmeans_classify__scheme_name__StrategyTransformer, self).__init__(
            'scheme_name', 
            pipeline_data_preprocessor, 
            verbose=verbose
        )
        
class C__top_n_significance__scheme_name__StrategyTransformer(C__top_n_significance__StrategyTransformer):
    def __init__(self, not_used_but_req_for_reflection_instantiation=None, pipeline_data_preprocessor=None, verbose=False):
        super(C__top_n_significance__scheme_name__StrategyTransformer, self).__init__(
            'scheme_name',
            top_n=10, # note that this class is highly tailored to this feature and this value may therefore need to be adjusted
            insig_map_to='other',
            pipeline_data_preprocessor=pipeline_data_preprocessor, 
            verbose=verbose
        )
# ************* StrategyTransformers specific to scheme_name: END *************
# ************* StrategyTransformers specific to wpt_operator: BEGIN *************


# ************* StrategyTransformers specific to permit: BEGIN *************
class C__missing_value_imputer__permit__StrategyTransformer(C__value_replacement__StrategyTransformer):
    def __init__(self, not_used_but_req_for_reflection_instantiation=None, pipeline_data_preprocessor=None, verbose=False):
        super(C__missing_value_imputer__permit__StrategyTransformer, self).__init__(
            'permit', 
            [{'missing_values': np.nan, 'strategy': 'constant', 'fill_value': False}],
            pipeline_data_preprocessor, 
            verbose=verbose
        )

class C__required_proprocessing__permit__StrategyTransformer(CCompositeStrategyTransformer):
    def __init__(self, not_used_but_req_for_reflection_instantiation=None, pipeline_data_preprocessor=None, verbose=False):
        super(C__required_proprocessing__permit__StrategyTransformer, self).__init__(
            description="required preprocessing for permit", 
            feat_transformer_sequence=[
                ['permit', C__missing_value_imputer__permit__StrategyTransformer]
            ],
            pipeline_data_preprocessor=pipeline_data_preprocessor, 
            verbose=verbose
        )
# ************* StrategyTransformers specific to public_meeting: END *************


# ************* StrategyTransformers specific to wpt_extraction_type_class__group: BEGIN *************
# ************* StrategyTransformers specific to extraction_type: BEGIN *************
class C__strip_nonalphanumeric__extraction_type__StrategyTransformer(C__strip_nonalphanumeric__StrategyTransformer):
    def __init__(self, not_used_but_req_for_reflection_instantiation=None, pipeline_data_preprocessor=None, verbose=False):
        super(C__strip_nonalphanumeric__extraction_type__StrategyTransformer, self).__init__(
            'extraction_type', 
            pipeline_data_preprocessor, 
            verbose=verbose
        )

class _C__OneHotEncode__extraction_type__StrategyTransformer(C__OneHotEncode__StrategyTransformer):
    def __init__(self, not_used_but_req_for_reflection_instantiation=None, pipeline_data_preprocessor=None, verbose=False):
        super(_C__OneHotEncode__extraction_type__StrategyTransformer, self).__init__(
            'extraction_type', 
            override_categories=None,
            pipeline_data_preprocessor=pipeline_data_preprocessor, 
            verbose=verbose
        )

class C__OneHotEncode__extraction_type__StrategyTransformer(CCompositeStrategyTransformer):
    def __init__(self, not_used_but_req_for_reflection_instantiation=None, pipeline_data_preprocessor=None, verbose=False):
        super(C__OneHotEncode__extraction_type__StrategyTransformer, self).__init__(
            description="required preprocessing to OneHot-Encode extraction_type", 
            feat_transformer_sequence=[
                ['extraction_type', C__strip_nonalphanumeric__extraction_type__StrategyTransformer],
                ['extraction_type', _C__OneHotEncode__extraction_type__StrategyTransformer] 
                # note that since _C__OneHotEncode__extraction_type__StrategyTransformer wraps OneHotEncodingTransformer, 
                #   the original feat that was encoded will obviously not be in the transformed data set;
                #   therefore, there is no need to drop it after this
            ],
            pipeline_data_preprocessor=pipeline_data_preprocessor, 
            verbose=verbose
        )
# ************* StrategyTransformers specific to extraction_type: END *************
# ************* StrategyTransformers specific to wpt_extraction_type_class__group: END *************

# ************* StrategyTransformers specific to wpt_management__group: BEGIN *************
# ************* StrategyTransformers specific to management: BEGIN *************
class C__strip_nonalphanumeric__management__StrategyTransformer(C__strip_nonalphanumeric__StrategyTransformer):
    def __init__(self, not_used_but_req_for_reflection_instantiation=None, pipeline_data_preprocessor=None, verbose=False):
        super(C__strip_nonalphanumeric__management__StrategyTransformer, self).__init__(
            'management', 
            pipeline_data_preprocessor, 
            verbose=verbose
        )

class _C__OneHotEncode__management__StrategyTransformer(C__OneHotEncode__StrategyTransformer):
    def __init__(self, not_used_but_req_for_reflection_instantiation=None, pipeline_data_preprocessor=None, verbose=False):
        super(_C__OneHotEncode__management__StrategyTransformer, self).__init__(
            'management', 
            override_categories=None,
            pipeline_data_preprocessor=pipeline_data_preprocessor, 
            verbose=verbose
        )

class C__OneHotEncode__management__StrategyTransformer(CCompositeStrategyTransformer):
    def __init__(self, not_used_but_req_for_reflection_instantiation=None, pipeline_data_preprocessor=None, verbose=False):
        super(C__OneHotEncode__management__StrategyTransformer, self).__init__(
            description="required preprocessing to OneHot-Encode management", 
            feat_transformer_sequence=[
                ['management', C__strip_nonalphanumeric__management__StrategyTransformer],
                ['management', _C__OneHotEncode__management__StrategyTransformer] 
                # note that since _C__OneHotEncode__management__StrategyTransformer wraps OneHotEncodingTransformer, 
                #   the original feat that was encoded will obviously not be in the transformed data set;
                #   therefore, there is no need to drop it after this
            ],
            pipeline_data_preprocessor=pipeline_data_preprocessor, 
            verbose=verbose
        )
# ************* StrategyTransformers specific to management: END *************
# ************* StrategyTransformers specific to wpt_management__group: END *************


# ************* StrategyTransformers specific to payment_frequency_class__group: BEGIN *************
# ************* StrategyTransformers specific to payment_type: BEGIN *************
class C__strip_nonalphanumeric__payment_type__StrategyTransformer(C__strip_nonalphanumeric__StrategyTransformer):
    def __init__(self, not_used_but_req_for_reflection_instantiation=None, pipeline_data_preprocessor=None, verbose=False):
        super(C__strip_nonalphanumeric__payment_type__StrategyTransformer, self).__init__(
            'payment_type', 
            pipeline_data_preprocessor, 
            verbose=verbose
        )

class _C__OneHotEncode__payment_type__StrategyTransformer(C__OneHotEncode__StrategyTransformer):
    def __init__(self, not_used_but_req_for_reflection_instantiation=None, pipeline_data_preprocessor=None, verbose=False):
        super(_C__OneHotEncode__payment_type__StrategyTransformer, self).__init__(
            'payment_type', 
            override_categories=None,
            pipeline_data_preprocessor=pipeline_data_preprocessor, 
            verbose=verbose
        )

class C__OneHotEncode__payment_type__StrategyTransformer(CCompositeStrategyTransformer):
    def __init__(self, not_used_but_req_for_reflection_instantiation=None, pipeline_data_preprocessor=None, verbose=False):
        super(C__OneHotEncode__payment_type__StrategyTransformer, self).__init__(
            description="required preprocessing to OneHot-Encode payment_type", 
            feat_transformer_sequence=[
                ['payment_type', C__strip_nonalphanumeric__payment_type__StrategyTransformer],
                ['payment_type', _C__OneHotEncode__payment_type__StrategyTransformer] 
                # note that since _C__OneHotEncode__payment_type__StrategyTransformer wraps OneHotEncodingTransformer, 
                #   the original feat that was encoded will obviously not be in the transformed data set;
                #   therefore, there is no need to drop it after this
            ],
            pipeline_data_preprocessor=pipeline_data_preprocessor, 
            verbose=verbose
        )
# ************* StrategyTransformers specific to payment_type: END *************
# ************* StrategyTransformers specific to payment_frequency_class__group: END *************


# ************* StrategyTransformers specific to water_quality_class__group: BEGIN *************
# ************* StrategyTransformers specific to water_quality: BEGIN *************
class C__strip_nonalphanumeric__water_quality__StrategyTransformer(C__strip_nonalphanumeric__StrategyTransformer):
    def __init__(self, not_used_but_req_for_reflection_instantiation=None, pipeline_data_preprocessor=None, verbose=False):
        super(C__strip_nonalphanumeric__water_quality__StrategyTransformer, self).__init__(
            'water_quality', 
            pipeline_data_preprocessor, 
            verbose=verbose
        )

class _C__OneHotEncode__water_quality__StrategyTransformer(C__OneHotEncode__StrategyTransformer):
    def __init__(self, not_used_but_req_for_reflection_instantiation=None, pipeline_data_preprocessor=None, verbose=False):
        super(_C__OneHotEncode__water_quality__StrategyTransformer, self).__init__(
            'water_quality', 
            override_categories=None,
            pipeline_data_preprocessor=pipeline_data_preprocessor, 
            verbose=verbose
        )

class C__OneHotEncode__water_quality__StrategyTransformer(CCompositeStrategyTransformer):
    def __init__(self, not_used_but_req_for_reflection_instantiation=None, pipeline_data_preprocessor=None, verbose=False):
        super(C__OneHotEncode__water_quality__StrategyTransformer, self).__init__(
            description="required preprocessing to OneHot-Encode water_quality", 
            feat_transformer_sequence=[
                ['water_quality', C__strip_nonalphanumeric__water_quality__StrategyTransformer],
                ['water_quality', _C__OneHotEncode__water_quality__StrategyTransformer] 
                # note that since _C__OneHotEncode__water_quality__StrategyTransformer wraps OneHotEncodingTransformer, 
                #   the original feat that was encoded will obviously not be in the transformed data set;
                #   therefore, there is no need to drop it after this
            ],
            pipeline_data_preprocessor=pipeline_data_preprocessor, 
            verbose=verbose
        )
# ************* StrategyTransformers specific to water_quality: END *************
# ************* StrategyTransformers specific to water_quality_class__group: END *************


# ************* StrategyTransformers specific to water_quantity_class__group: BEGIN *************
# ************* StrategyTransformers specific to quantity: BEGIN *************
class C__strip_nonalphanumeric__quantity__StrategyTransformer(C__strip_nonalphanumeric__StrategyTransformer):
    def __init__(self, not_used_but_req_for_reflection_instantiation=None, pipeline_data_preprocessor=None, verbose=False):
        super(C__strip_nonalphanumeric__quantity__StrategyTransformer, self).__init__(
            'quantity', 
            pipeline_data_preprocessor, 
            verbose=verbose
        )

class _C__OneHotEncode__quantity__StrategyTransformer(C__OneHotEncode__StrategyTransformer):
    def __init__(self, not_used_but_req_for_reflection_instantiation=None, pipeline_data_preprocessor=None, verbose=False):
        super(_C__OneHotEncode__quantity__StrategyTransformer, self).__init__(
            'quantity', 
            override_categories=None,
            pipeline_data_preprocessor=pipeline_data_preprocessor, 
            verbose=verbose
        )

class C__OneHotEncode__quantity__StrategyTransformer(CCompositeStrategyTransformer):
    def __init__(self, not_used_but_req_for_reflection_instantiation=None, pipeline_data_preprocessor=None, verbose=False):
        super(C__OneHotEncode__quantity__StrategyTransformer, self).__init__(
            description="required preprocessing to OneHot-Encode quantity", 
            feat_transformer_sequence=[
                ['quantity', C__strip_nonalphanumeric__quantity__StrategyTransformer],
                ['quantity', _C__OneHotEncode__quantity__StrategyTransformer] 
                # note that since _C__OneHotEncode__quantity__StrategyTransformer wraps OneHotEncodingTransformer, 
                #   the original feat that was encoded will obviously not be in the transformed data set;
                #   therefore, there is no need to drop it after this
            ],
            pipeline_data_preprocessor=pipeline_data_preprocessor, 
            verbose=verbose
        )
# ************* StrategyTransformers specific to quantity: END *************
# ************* StrategyTransformers specific to water_quantity_class__group: END *************


# ************* StrategyTransformers specific to water_source_type_class__group: BEGIN *************
# ************* StrategyTransformers specific to source: BEGIN *************
class C__strip_nonalphanumeric__source__StrategyTransformer(C__strip_nonalphanumeric__StrategyTransformer):
    def __init__(self, not_used_but_req_for_reflection_instantiation=None, pipeline_data_preprocessor=None, verbose=False):
        super(C__strip_nonalphanumeric__source__StrategyTransformer, self).__init__(
            'source', 
            pipeline_data_preprocessor, 
            verbose=verbose
        )

class _C__OneHotEncode__source__StrategyTransformer(C__OneHotEncode__StrategyTransformer):
    def __init__(self, not_used_but_req_for_reflection_instantiation=None, pipeline_data_preprocessor=None, verbose=False):
        super(_C__OneHotEncode__source__StrategyTransformer, self).__init__(
            'source', 
            override_categories=None,
            pipeline_data_preprocessor=pipeline_data_preprocessor, 
            verbose=verbose
        )

class C__OneHotEncode__source__StrategyTransformer(CCompositeStrategyTransformer):
    def __init__(self, not_used_but_req_for_reflection_instantiation=None, pipeline_data_preprocessor=None, verbose=False):
        super(C__OneHotEncode__source__StrategyTransformer, self).__init__(
            description="required preprocessing to OneHot-Encode source", 
            feat_transformer_sequence=[
                ['source', C__strip_nonalphanumeric__source__StrategyTransformer],
                ['source', _C__OneHotEncode__source__StrategyTransformer] 
                # note that since _C__OneHotEncode__source__StrategyTransformer wraps OneHotEncodingTransformer, 
                #   the original feat that was encoded will obviously not be in the transformed data set;
                #   therefore, there is no need to drop it after this
            ],
            pipeline_data_preprocessor=pipeline_data_preprocessor, 
            verbose=verbose
        )
# ************* StrategyTransformers specific to source: END *************
# ************* StrategyTransformers specific to water_source_type_class__group: END *************


# ************* StrategyTransformers specific to wpt_type_class__group: BEGIN *************
# ************* StrategyTransformers specific to waterpoint_type: BEGIN *************
class C__strip_nonalphanumeric__waterpoint_type__StrategyTransformer(C__strip_nonalphanumeric__StrategyTransformer):
    def __init__(self, not_used_but_req_for_reflection_instantiation=None, pipeline_data_preprocessor=None, verbose=False):
        super(C__strip_nonalphanumeric__waterpoint_type__StrategyTransformer, self).__init__(
            'waterpoint_type', 
            pipeline_data_preprocessor, 
            verbose=verbose
        )

class _C__OneHotEncode__waterpoint_type__StrategyTransformer(C__OneHotEncode__StrategyTransformer):
    def __init__(self, not_used_but_req_for_reflection_instantiation=None, pipeline_data_preprocessor=None, verbose=False):
        super(_C__OneHotEncode__waterpoint_type__StrategyTransformer, self).__init__(
            'waterpoint_type', 
            override_categories=None,
            pipeline_data_preprocessor=pipeline_data_preprocessor, 
            verbose=verbose
        )

class C__OneHotEncode__waterpoint_type__StrategyTransformer(CCompositeStrategyTransformer):
    def __init__(self, not_used_but_req_for_reflection_instantiation=None, pipeline_data_preprocessor=None, verbose=False):
        super(C__OneHotEncode__waterpoint_type__StrategyTransformer, self).__init__(
            description="required preprocessing to OneHot-Encode waterpoint_type", 
            feat_transformer_sequence=[
                ['waterpoint_type', C__strip_nonalphanumeric__waterpoint_type__StrategyTransformer],
                ['waterpoint_type', _C__OneHotEncode__waterpoint_type__StrategyTransformer] 
                # note that since _C__OneHotEncode__waterpoint_type__StrategyTransformer wraps OneHotEncodingTransformer, 
                #   the original feat that was encoded will obviously not be in the transformed data set;
                #   therefore, there is no need to drop it after this
            ],
            pipeline_data_preprocessor=pipeline_data_preprocessor, 
            verbose=verbose
        )
# ************* StrategyTransformers specific to waterpoint_type: END *************
# ************* StrategyTransformers specific to wpt_type_class__group: END *************