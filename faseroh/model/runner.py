import json
import os

import aclick
import numpy as np
import tensorflow as tf
from sklearn.metrics import r2_score
from sympy import sympify

from faseroh.dataset.tokenizers import GeneralTermTokenizer
from faseroh.dataset.utils.sympy_functions import expr_to_func
from faseroh.dataset.utils.expression import generate_histogram

from scipy.stats import poisson

from faseroh.utils import pull_model
from .config import Config
from .model import TransformerWithRegressionAsSeq
from .utils.const_improver import OptimizationType
from .utils.convertor import BestFittingFilter
from .utils.decoding import RandomSampler, TopK

class Runner:
    def __init__(
        self,
        model_path,
        config,
        variables,
        sampler=TopK(20),
        optimization_type: OptimizationType = "gradient",
        use_pool=True,
        early_stopping=False,
        num_equations=256,
    ):
        self.config = config
        self.variables = variables
        self.tokenizer = GeneralTermTokenizer(variables)
        self.model = self.load_model(model_path, config)
        self.search = RandomSampler(
            sampler,
            200,
            self.tokenizer,
            self.model,
            True,
            BestFittingFilter(
                self.tokenizer, optimization_type, use_pool, early_stopping
            ),
            num_equations,
        )

    @classmethod
    def from_checkpoint(cls, path, **kwargs):
        path = pull_model(path)
        with open(os.path.join(path, "config.json")) as f:
            config_dict = json.load(f)
        config_dict["dataset_config"]["path"] = ""
        config_dict["dataset_config"]["valid_path"] = ""
        config: Config = aclick.utils.from_dict(Config, config_dict)
        return cls(
            os.path.join(path, "weights"),
            config,
            variables=config.dataset_config.variables,
            **kwargs
        )

    def load_model(self, model_path, config):
        strategy = tf.distribute.get_strategy()
        transformer = TransformerWithRegressionAsSeq(
            config, self.tokenizer, config.input_regularizer, strategy
        )
        transformer.build(
            input_shape=[
                (
                    None,
                    config.dataset_config.num_points,
                    len(config.dataset_config.variables) + 1,
                ),
                (None, None),
                (None, None),
            ]
        )
        transformer.compile()
        transformer.load_weights(model_path)
        return transformer

    def predict(self, equation, points=None):
        sym_eq = sympify(equation)
        # lam = expr_to_func(sym_eq, self.variables)
        if points is None:
            _ , mean_counts = generate_histogram(f = sym_eq)
            points = poisson.rvs(mean_counts)
            points = tf.convert_to_tensor([points])

        prediction = self.search.batch_decode(points)
        pred_inorder = prediction[-2][0]
        prediction = prediction[-1][0]
        # golden_lambda = expr_to_func(prediction, self.variables)

        _ , mean_counts = generate_histogram(f = sym_eq)
        points = poisson.rvs(mean_counts)
        
        _, mean_pred_y = generate_histogram(f = prediction)
        pred_y = poisson.rvs(mean_pred_y)

        return (
            pred_inorder,
            r2_score(points, pred_y),
            np.mean(
                np.abs(points - pred_y) / np.abs(points)
            ),
        )

    def predict_all(self, equation=None, points=None):
        if points is None:
            assert equation is not None
            sym_eq = sympify(equation)
            # lam = expr_to_func(sym_eq, self.variables)
            if points is None:
                _, mean_counts = generate_histogram(f = sym_eq)
                points = poisson.rvs(mean_counts)
                points = tf.convert_to_tensor([points])

        prediction = self.search.batch_decode(points, return_all=True)
        return prediction