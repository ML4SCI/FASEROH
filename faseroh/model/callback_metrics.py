import enum
import io
import os
from typing import List

import numpy as np

import PIL
import tensorflow as tf
import wandb
from matplotlib import pyplot as plt
from scipy.integrate import trapz
from sklearn import metrics

from scipy.stats import poisson

from ..dataset.utils.sympy_functions import expr_to_func
from ..dataset.utils.expression import generate_histogram

class MetricType(enum.Enum):
    INORDER = "inorder"
    SYMBOLIC = "symbolic"


class CallbackMetric:
    TYPE = None

    def __init__(self, variables: List[str]):
        self.variables = variables

    def reset_states(self, epoch):
        raise NotImplementedError()

    def update(self, golden, prediction):
        raise NotImplementedError()

    def log(self, logs):
        raise NotImplementedError()

    def commit(self, dataset_name: str, search_name: str, model: tf.keras.Model):
        raise NotImplementedError()


class TableCallbackMetric(CallbackMetric):
    TYPE = MetricType.INORDER

    def __init__(self, variables: List[str]):
        super().__init__(variables)
        self.table = None
        self.wandb_log = None

    def reset_states(self, epoch):
        self.table = wandb.Table(columns=["golden", "prediction"])
        self.wandb_log = {
            "epoch": epoch,
        }

    def update(self, golden, prediction):
        if golden is None or prediction is None:
            return {"golden": None, "pred": None}

        return {"golden": str(golden), "pred": str(prediction)}

    def log(self, logs):
        for log in logs:
            self.table.add_data(log["golden"], log["pred"])

    def commit(self, dataset_name: str, search_name: str, model: tf.keras.Model):
        self.wandb_log[f"{dataset_name}/{search_name}/examples"] = self.table
        wandb.log(self.wandb_log, commit=False)


class PointMetrics(CallbackMetric):
    TYPE = MetricType.SYMBOLIC

    def __init__(
        self, eps: List[float], num_points: int, variables: List[str], save_model=True
    ):
        super().__init__(variables)
        self.golden_points = []
        self.eps = eps
        self.save_model = save_model
        self.best_r2 = -np.inf

        self.num_points = num_points
        self.point_dim = len(variables)
        self.golden_hists = []
        self.prediction_hists = []
        self.wandb_log = {}

  

    def reset_states(self, epoch):
        self.golden_hists = []
        self.prediction_hists = []
        self.wandb_log = {"epoch": epoch}

    def update(self, golden, pred):
        results = {
                "valid_length": 0,
                "golden": [],
                "prediction": []
                  }
        if golden is None or pred is None:
            return self.pad(results)

        try:
            str_golden = str(golden)
           
            golden_lambda = golden #expr_to_func(golden, self.variables)
            _, points = generate_histogram(golden_lambda)
            points = poisson.rvs(points)
            self.golden_points = points

            golden_points = self.golden_points

            #pred = expr_to_func(pred, self.variables)
            _, points_pred = generate_histogram(pred)
            points_pred = poisson.rvs(points_pred)
            if golden_points is not None:
                results["golden"] = golden_points
                results["prediction"] = points_pred
                results["valid_length"] = len(
                    golden_points
                )

        except (
            AttributeError,
            KeyError,
            RuntimeWarning,
            NameError,
            TypeError,
            RuntimeError,
            OverflowError,
            ZeroDivisionError,
            MemoryError,
            ValueError,
            FloatingPointError,
        ):
            pass
        return results

    def log(self, logs):
        for log in logs:
            name = "pointmetric"
            length = log["valid_length"]
            if length != 0:
                self.golden_hists.append(log["golden"][:length])
                self.prediction_hists.append(log["prediction"][:length])


    def commit(self, dataset_name: str, search_name: str, model: tf.keras.Model):
        maes = []
        relative_errors = []
        r2s = []
        for pred, golden in zip(
            self.prediction_hists, self.golden_hists
        ):
            pred = tf.reshape(pred, [-1])
            golden = tf.reshape(golden, [-1])
            mask = np.isfinite(pred)
            pred = pred[mask]
            golden = golden[mask]
            if len(pred) == 0 or len(golden) == 0:
                continue
            mae = self.mae(golden, pred)
            relative_error = mae / (np.abs(golden) + 1e-8)
            r2 = metrics.r2_score(golden, pred)
            maes.append(np.nanmean(mae))
            relative_errors.append(np.nanmean(relative_error))
            r2s.append(r2)

        if len(maes) != 0:
            r2median = np.nanmedian(r2s)
            if r2median > self.best_r2 and search_name == "no_teacher":
                model.save_weights(
                    f'{os.getenv("OUTPUT")}/checkpoints/{os.getenv("JOB_ID")}/weights',
                    overwrite=True,
                )
                tf.print(
                    f"New best median MAE {r2median} found. Previously {self.best_r2}. Overwriting."
                )
                self.best_r2 = r2median

            model.save_weights(
                f'{os.getenv("OUTPUT")}/checkpoints/{os.getenv("JOB_ID")}/last',
                overwrite=True,
            )

            self._log(
                maes, relative_errors, r2s, dataset_name, search_name, "pointmetric"
            )

        wandb.log(self.wandb_log, commit=False)

    def _log(self, maes, relative_error, r2, dataset_name, search_name, metric_type):
        if len(maes) != 0:
            self.wandb_log[
                f"{dataset_name}/{search_name}/points_mae_{metric_type}"
            ] = np.nanmedian(maes)
            self.wandb_log[
                f"{dataset_name}/{search_name}/points_mae_{metric_type}_mean"
            ] = np.nanmean(maes)

            self.wandb_log[
                f"{dataset_name}/{search_name}/points_relative_error_{metric_type}"
            ] = np.nanmedian(relative_error)
            self.wandb_log[
                f"{dataset_name}/{search_name}/points_relative_error_{metric_type}_mean"
            ] = np.nanmean(relative_error)

            self.wandb_log[
                f"{dataset_name}/{search_name}/points_accuracy_{metric_type}_image"
            ] = self.plot_image(maes)
            self.wandb_log[
                f"{dataset_name}/{search_name}/points_relative_error_{metric_type}_image"
            ] = self.plot_image(relative_error)

            self.wandb_log[
                f"{dataset_name}/{search_name}/points_accuracy_{metric_type}_auc"
            ] = self.auc(maes)
            self.wandb_log[
                f"{dataset_name}/{search_name}/points_relative_error_{metric_type}_auc"
            ] = self.auc(relative_error)

            self.wandb_log[
                f"{dataset_name}/{search_name}/points_r2_{metric_type}"
            ] = np.nanmedian(r2)
            self.wandb_log[
                f"{dataset_name}/{search_name}/points_r2_{metric_type}_mean"
            ] = np.nanmean(r2)

            for eps in self.eps:
                self.wandb_log[
                    f"{dataset_name}/{search_name}/points_accuracy_{metric_type}_{eps}"
                ] = np.nanmean(np.array(maes) < eps)

                self.wandb_log[
                    f"{dataset_name}/{search_name}/points_relative_error_{metric_type}_{eps}"
                ] = np.nanmean(np.array(relative_error) < eps)

    def mae(self, a, b):
        if tf.is_tensor(a):
            a = a.numpy()

        if tf.is_tensor(b):
            b = b.numpy()
        return np.abs(a.astype(float) - b.astype(float))
