from abc import ABC, abstractmethod
from . import impute_TO_nan, analyze_outliers_detailed
from sklearn.preprocessing import FunctionTransformer
from .skl_transformers import fit_target_encoder, target_encoder_transform, DropColumnsTransformer, SimpleValueTransformer

class CBaseStrategyTransformer():
    def __init__(self, feat, pipeline_data_preprocessor, description, verbose=False):
        self.feat = feat
        self.pipeline_data_preprocessor = pipeline_data_preprocessor
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
                print(f"{self.pipeline_step} appended to pipeline")

        return self

    def transform(self, X):
        X_transformed = self.pipeline_step[1].transform(X) if self.pipeline_step is not None else self.transformer.fit_transform(X)
        if self.verbose:
                print(f"strategy transformation for feature {self.feat} COMPLETE")
        return X_transformed

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)




class C__leave_it_as_is__StrategyTransformer(CBaseStrategyTransformer):
    def __init__(self, feat, pipeline_data_preprocessor, verbose=False):
        super(C__leave_it_as_is__StrategyTransformer, self).__init__(
            feat, 
            pipeline_data_preprocessor, 
            description=f"leave feature {feat} as is (do nothing)",
            verbose=verbose
        )

    def get_transformer(self, X, y=None):
        return FunctionTransformer(lambda X: X, validate=False)




class C__drop_it__StrategyTransformer(CBaseStrategyTransformer):
    def __init__(self, feat, pipeline_data_preprocessor, verbose=False):
        super(C__drop_it__StrategyTransformer, self).__init__(
            feat, 
            pipeline_data_preprocessor, 
            description=f"drop feature {feat}",
            verbose=verbose
        )

    def get_transformer(self, X, y=None):
        return DropColumnsTransformer(self.feat)




class C__replace_0_with_nan__StrategyTransformer(CBaseStrategyTransformer):
    def __init__(self, feat, pipeline_data_preprocessor, verbose=False):
        super(C__replace_0_with_nan__StrategyTransformer, self).__init__(
            feat, 
            pipeline_data_preprocessor, 
            description=f"replace {feat} 0-values with nan",
            verbose=verbose
        )

    def get_transformer(self, X, y=None):
        return FunctionTransformer(lambda X: impute_TO_nan(X, self.feat, 0), validate=False)



class C__replace_outliers__StrategyTransformer(CBaseStrategyTransformer):
    def __init__(self, feat, replacement_strategy, pipeline_data_preprocessor, verbose=False):
        super(C__replace_outliers__StrategyTransformer, self).__init__(
            feat, 
            pipeline_data_preprocessor, 
            description=f"replace {feat} outliers with {replacement_strategy}",
            verbose=verbose
        )
        self.replacement_strategy = replacement_strategy
        
    def get_transformer(self, X, y=None):
        _, _, all_replace_outliers_rules = analyze_outliers_detailed(X, '', self.feat, suppress_output=True)
        if self.replacement_strategy in all_replace_outliers_rules:
            replacement_rules = all_replace_outliers_rules[self.replacement_strategy]
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
        dct_after_target_encode = DropColumnsTransformer(self.feat)
        if self.pipeline_data_preprocessor is not None:
            self.pipeline_data_preprocessor.steps.append([f"drop after target encoding: {self.feat}", dct_after_target_encode])
            pipeline_step = self.pipeline_data_preprocessor.steps[-1]
            if self.verbose:
                print(f"{pipeline_step} appended to pipeline")

        X_transformed = pipeline_step[1].transform(X_transformed) if pipeline_step is not None else dct_after_target_encode.fit_transform(X_transformed)
        if self.verbose:
                print(f"dropped feature {self.feat} after target encoding")
                print(f"strategy transformation for feature {self.feat} COMPLETE")

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