import importlib
from abc import ABC, abstractmethod
from . import impute_TO_nan, impute_TO_lcase, analyze_outliers_detailed, convert_col_to_date_type
from sklearn.preprocessing import FunctionTransformer
from .skl_transformers import fit_target_encoder, target_encoder_transform, DropColumnsTransformer, SimpleValueTransformer
from .submodels import tfidf_kmeans_classify_feature
from IPython.core.display import HTML, Markdown
import numpy as np
import inspect
import json




# note that since these are instantiated through the instantiate_strategy_transfomer() method
#   by specifying the descendant as a string (in the EDA config file), all descendant CBaseStrategyTransformer
#   class constructors MUST have the signature (self, feat, pipeline_data_preprocessor, verbose)
class CBaseStrategyTransformer():
    def __init__(self, feat, pipeline_data_preprocessor, description, verbose=False):
        self.feat = feat
        self.transformer = None
        self.pipeline_data_preprocessor = pipeline_data_preprocessor
        self.pipeline_step = None
        self.description = description
        self.verbose = verbose

    @abstractmethod
    def get_transformer(self, X, y=None): # fitting should occur in the override
        pass

    def fit(self, X, y=None):
        self.transformer = self.get_transformer(X, y)

        if self.pipeline_data_preprocessor is not None:
            self.pipeline_data_preprocessor.steps.append([self.description, self.transformer])
            self.pipeline_step = self.pipeline_data_preprocessor.steps[-1]
            if self.verbose:
                print(f"strategy \"{self.description}\" appended step {self.pipeline_step} to pipeline")

        return self

    def transform(self, X):
        X_transformed = self.pipeline_step[1].transform(X) if self.pipeline_step is not None else self.transformer.fit_transform(X)
        if self.verbose:
            print(f"strategy \"{self.description}\" transformation for feature \"{self.feat}\" is COMPLETE!")
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
            post_encode_null_to_global_mean=self.post_encode_null_to_global_mean
        )
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
                print(f"strategy '{self.description}' appended step {pipeline_step} to pipeline")

        X_transformed = pipeline_step[1].transform(X_transformed) if pipeline_step is not None else dct_after_target_encode.fit_transform(X_transformed)
        if self.verbose:
            print(f"strategy '{self.description}' dropped feature '{self.feat}' after target encoding")
            print(f"strategy '{self.description}' transformation for feature '{self.feat}' is COMPLETE!")

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




class C__tfidf_kmeans_classify__StrategyTransformer(CBaseStrategyTransformer):
    def __init__(self, feat, pipeline_data_preprocessor, verbose=False):
        super(C__tfidf_kmeans_classify__StrategyTransformer, self).__init__(
            feat, 
            pipeline_data_preprocessor, 
            description=f"tfidf kmeans classify high-cardinality string-categorical: {feat}",
            verbose=verbose
        )
        
    def get_transformer(self, X, y=None):
        return FunctionTransformer(lambda X: tfidf_kmeans_classify_feature(
                X, 
                '', 
                'funder', 
                verbosity=0
            )[0], 
            validate=False
        )





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




# ******************* API for string (reflection) based instantiation - used in conjunction with invokinng strategies specified in config file: BEGIN *******************

# used for "reflection" instantation - i.e. instantiation via class name (string)
#   this is integral to being able to dynamically switch to a different strategy via the config file
def strategy_transformer_name_to_class(strategy_transformer_class_name):
    return getattr(importlib.import_module("scjpnlib.utils.strategy_transformers"), strategy_transformer_class_name)

class BadCtorSignature(Exception):
    def __init__(self, class_name):
        self.message = f"class {class_name} ctor does not match required signature: self, feat, pipeline_data_preprocessor, verbose"

def instantiate_strategy_transformer(strategy_composition, description, pipeline):
    feat_transformer_sequence = []
    for strategy_component in strategy_composition:
        # print(f"strategy component feat: {strategy_component[0]}")
        # print(f"strategy component class-name: {strategy_component[1]}")
        StratTransformerClass = strategy_transformer_name_to_class(strategy_component[1])

        # check ctor signature - it must match required sig which is 4 args: self, feat, pipeline_data_preprocessor, verbose
        ctor_argspec = inspect.getargspec(StratTransformerClass.__init__)
        if len(ctor_argspec.args) != 4:
            raise(BadCtorSignature(strategy_component[1]))

        feat_transformer_sequence.append((strategy_component[0], StratTransformerClass))

    return CCompositeStrategyTransformer(description, feat_transformer_sequence, pipeline, verbose=True)

def _html_prettify_strategy_transformer_description(strategy_transformer):
    if isinstance(strategy_transformer, CCompositeStrategyTransformer):
        s_html = f"<b>(composite) strategy name/description: <i><font color='red'>{strategy_transformer.description}</font></i></b>"
        s_html += "<ol>"
        for feat_transformer in strategy_transformer.feat_transformer_sequence:
            s_html += f"<li>{_html_prettify_strategy_transformer_description(feat_transformer)}</li>"
        s_html += "</ol>"
        return s_html
    else:
        return f"<b>strategy description</b>: <i><font color='blue'>{strategy_transformer.description}</font></i>"

def html_prettify_strategy_transformer_description(strategy_transformer):
    display(HTML(_html_prettify_strategy_transformer_description(strategy_transformer)))

# ******************* API for string (reflection) based instantiation - used in conjunction with invokinng strategies specified in config file: END *******************









# Below are strategy transformers that are specific to features

# ************* StrategyTransformers specific to pump_age: BEGIN *************
class C__convert_string_date_to_datetime__date_recorded__StrategyTransformer(C__convert_string_date_to_datetime__StrategyTransformer):
    def __init__(self, not_used_but_req_for_reflection_instantiation=None, pipeline_data_preprocessor=None, verbose=False):
        super(C__convert_string_date_to_datetime__date_recorded__StrategyTransformer, self).__init__(
            'date_recorded',
            from_format="%Y-%m-%d",
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

class C__create_pump_age_feature_from_date_recorded_and_construction_year__StrategyTransformer(CBaseStrategyTransformer):
    def __init__(self, not_used_but_req_for_reflection_instantiation=None, pipeline_data_preprocessor=None, verbose=False):
        super(C__create_pump_age_feature_from_date_recorded_and_construction_year__StrategyTransformer, self).__init__(
            'pump_age', 
            pipeline_data_preprocessor, 
            description=f"create feature from date_recorded and construction_year: pump_age",
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
                ['date_recorded', C__convert_string_date_to_datetime__date_recorded__StrategyTransformer],
                ['construction_year', C__replace_0_construction_year_with_date_recorded__StrategyTransformer],
                ['construction_year', C__convert_string_date_to_datetime__construction_year__StrategyTransformer],
                ['pump_age', C__create_pump_age_feature_from_date_recorded_and_construction_year__StrategyTransformer],
                ['date_recorded', C__drop_it__StrategyTransformer],
                ['construction_year', C__drop_it__StrategyTransformer]
            ],
            pipeline_data_preprocessor=pipeline_data_preprocessor, 
            verbose=verbose
        )
# ************* StrategyTransformers specific to pump_age: END *************


# ************* StrategyTransformers specific to funder: BEGIN *************
class C__impute_lcase__funder__StrategyTransformer(C__impute_lcase__StrategyTransformer):
    def __init__(self, not_used_but_req_for_reflection_instantiation=None, pipeline_data_preprocessor=None, verbose=False):
        super(C__impute_lcase__funder__StrategyTransformer, self).__init__(
            'funder', 
            pipeline_data_preprocessor, 
            verbose=verbose
        )

class C__missing_value_imputer__funder__StrategyTransformer(C__value_replacement__StrategyTransformer):
    def __init__(self, not_used_but_req_for_reflection_instantiation=None, pipeline_data_preprocessor=None, verbose=False, missing_string_value_replacement="none"):
        super(C__missing_value_imputer__funder__StrategyTransformer, self).__init__(
            'funder', 
            [
                {'missing_values': np.nan, 'strategy': 'constant', 'fill_value': missing_string_value_replacement},
                {'missing_values': '0', 'strategy': 'constant', 'fill_value': missing_string_value_replacement}
            ],
            pipeline_data_preprocessor, 
            verbose=verbose
        )

class C__not_known_literal_value_replacement__funder__StrategyTransformer(C__value_replacement__StrategyTransformer):
    def __init__(self, not_used_but_req_for_reflection_instantiation=None, pipeline_data_preprocessor=None, verbose=False, not_known_literal_value_replacement="unknown"):
        super(C__not_known_literal_value_replacement__funder__StrategyTransformer, self).__init__(
            'funder', 
            [{'missing_values': 'not known', 'strategy': 'constant', 'fill_value': not_known_literal_value_replacement}],
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

class C__tfidf_kmeans_classify__funder__StrategyTransformer(C__tfidf_kmeans_classify__StrategyTransformer):
    def __init__(self, not_used_but_req_for_reflection_instantiation=None, pipeline_data_preprocessor=None, verbose=False):
        super(C__tfidf_kmeans_classify__funder__StrategyTransformer, self).__init__(
            'funder', 
            pipeline_data_preprocessor, 
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
    def __init__(self, not_used_but_req_for_reflection_instantiation=None, pipeline_data_preprocessor=None, verbose=False, missing_string_value_replacement="none"):
        super(C__missing_value_imputer__installer__StrategyTransformer, self).__init__(
            'installer', 
            [
                {'missing_values': np.nan, 'strategy': 'constant', 'fill_value': missing_string_value_replacement},
                {'missing_values': '0', 'strategy': 'constant', 'fill_value': missing_string_value_replacement},
                {'missing_values': '-', 'strategy': 'constant', 'fill_value': missing_string_value_replacement}
            ],
            pipeline_data_preprocessor, 
            verbose=verbose
        )

class C__not_known_literal_value_replacement__installer__StrategyTransformer(C__value_replacement__StrategyTransformer):
    def __init__(self, not_used_but_req_for_reflection_instantiation=None, pipeline_data_preprocessor=None, verbose=False, not_known_literal_value_replacement="unknown"):
        super(C__not_known_literal_value_replacement__installer__StrategyTransformer, self).__init__(
            'installer', 
            [{'missing_values': 'not known', 'strategy': 'constant', 'fill_value': not_known_literal_value_replacement}],
            pipeline_data_preprocessor, 
            verbose=verbose
        )

class C__required_proprocessing__installer__StrategyTransformer(CCompositeStrategyTransformer):
    def __init__(self, not_used_but_req_for_reflection_instantiation=None, pipeline_data_preprocessor=None, verbose=False):
        super(C__required_proprocessing__installer__StrategyTransformer, self).__init__(
            description="required preprocessing for funder", 
            feat_transformer_sequence=[
                ['installer', C__impute_lcase__installer__StrategyTransformer],
                ['installer', C__missing_value_imputer__installer__StrategyTransformer],
                ['installer', C__not_known_literal_value_replacement__installer__StrategyTransformer]
            ],
            pipeline_data_preprocessor=pipeline_data_preprocessor, 
            verbose=verbose
        )

class C__tfidf_kmeans_classify__installer__StrategyTransformer(C__tfidf_kmeans_classify__StrategyTransformer):
    def __init__(self, not_used_but_req_for_reflection_instantiation=None, pipeline_data_preprocessor=None, verbose=False):
        super(C__tfidf_kmeans_classify__installer__StrategyTransformer, self).__init__(
            'installer', 
            pipeline_data_preprocessor, 
            verbose=verbose
        )
# ************* StrategyTransformers specific to installer: END *************