from sklearn import metrics
from sklearn import model_selection
import numpy as np
import pandas as pd

class EvaluationMetrics:
    """
    Benchmark according to Kaggle competition evaluation

    Thanks to https://www.kaggle.com/dborkan/benchmark-kernel
    """
    SUBGROUP_AUC = 'subgroup_auc'
    BPSN_AUC = 'bpsn_auc'  # stands for background positive, subgroup negative
    BNSP_AUC = 'bnsp_auc'  # stands for background negative, subgroup positive

    def compute_auc(y_true, y_pred):
        """
        Computes the AUC.

        Note: this implementation is restricted to the binary classification task
        or multilabel classification task in label indicator format.


        Parameters
        ----------
        y_true : array, shape = [n_samples] or [n_samples, n_classes]
            True binary labels or binary label indicators.

        y_pred : array, shape = [n_samples] or [n_samples, n_classes]
            Target scores, can either be probability estimates of the positive
            class, confidence values, or non-thresholded measure of decisions
            (as returned by "decision_function" on some classifiers). For binary
            y_true, y_score is supposed to be the score of the class with greater
            label.
        """
        try:
            return metrics.roc_auc_score(y_true, y_pred)
        except ValueError:
            return np.nan

    def compute_subgroup_auc(df, subgroup, label, model_name):
        """Computes the AUC for the within-subgroup positive and negative examples."""
        subgroup_examples = df[(df[subgroup] > .5)]
        return EvaluationMetrics.compute_auc(subgroup_examples[label], subgroup_examples[model_name])

    def compute_bpsn_auc(df, subgroup, label, model_name):
        """Computes the AUC of the within-subgroup negative examples and the background positive examples."""
        subgroup_negative_examples = df[(df[subgroup] > 0.5) & ~(df[label] > 0.5)]
        non_subgroup_positive_examples = df[~(df[subgroup] > 0.5) & (df[label] > 0.5)]
        examples = subgroup_negative_examples.append(non_subgroup_positive_examples)
        return EvaluationMetrics.compute_auc(examples[label], examples[model_name])

    def compute_bnsp_auc(df, subgroup, label, model_name):
        """Computes the AUC of the within-subgroup positive examples and the background negative examples."""
        subgroup_positive_examples = df[(df[subgroup] > 0.5) & (df[label] > 0.5)]
        non_subgroup_negative_examples = df[~(df[subgroup] > 0.5) & ~(df[label] > 0.5)]
        examples = subgroup_positive_examples.append(non_subgroup_negative_examples)
        return EvaluationMetrics.compute_auc(examples[label], examples[model_name])

    def compute_bias_metrics_for_model(dataset,
                                       subgroups,
                                       model,
                                       label_col,
                                       include_asegs=False):
        """Computes per-subgroup metrics for all subgroups and one model."""

        records = []
        for subgroup in subgroups:
            record = {
                'subgroup': subgroup,
                'subgroup_size': len(dataset[(dataset[subgroup] > 0.5)])
            }

            record[EvaluationMetrics.SUBGROUP_AUC] = EvaluationMetrics.compute_subgroup_auc(dataset, subgroup, label_col, model)
            record[EvaluationMetrics.BPSN_AUC] = EvaluationMetrics.compute_bpsn_auc(dataset, subgroup, label_col, model)
            record[EvaluationMetrics.BNSP_AUC] = EvaluationMetrics.compute_bnsp_auc(dataset, subgroup, label_col, model)
            records.append(record)

        return pd.DataFrame(records).sort_values('subgroup_auc', ascending=True)

    def calculate_overall_auc(df, model_name, label_column):
        true_labels = df[label_column]
        predicted_labels = df[model_name]
        return metrics.roc_auc_score(true_labels, predicted_labels)

    def power_mean(series, p):
        total = sum(np.power(series, p))
        return np.power(total / len(series), 1 / p)

    def get_final_metric(bias_df, overall_auc, POWER=-5, OVERALL_MODEL_WEIGHT=0.25):
        bias_score = np.average([
            EvaluationMetrics.power_mean(bias_df[EvaluationMetrics.SUBGROUP_AUC], POWER),
            EvaluationMetrics.power_mean(bias_df[EvaluationMetrics.BPSN_AUC], POWER),
            EvaluationMetrics.power_mean(bias_df[EvaluationMetrics.BNSP_AUC], POWER)
        ])
        return (OVERALL_MODEL_WEIGHT * overall_auc) + ((1 - OVERALL_MODEL_WEIGHT) * bias_score)
