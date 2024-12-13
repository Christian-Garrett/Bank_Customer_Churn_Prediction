import numpy as np
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from sklearn.metrics import (roc_auc_score,
                             f1_score,
                             recall_score,
                             confusion_matrix,
                             classification_report)


def get_weighted_ensemble_results(self):

    PRED_THRESH = 0.5
    WEIGHT_PERC = 0.9

    best_model_results = dict()
    weighted_model_results = dict()

    ## Best model predictions
    best_model_pred_probs = \
        self.ensemble_experiment_dict['partial_target_balancing']['val']['pred_probs']
    ensemble_model1_pred_probs = \
        self.ensemble_experiment_dict['no_target_balancing']['val']['pred_probs']
    ensemble_model2_pred_probs = \
        self.ensemble_experiment_dict['full_target_balancing']['val']['pred_probs']

    best_model_preds = np.where(best_model_pred_probs[:, 1] >= PRED_THRESH, 1, 0)

    ## Model averaging predictions (Weighted average)
    ensemble_model_preds = \
        np.where((((1-WEIGHT_PERC)*ensemble_model1_pred_probs[:, 1]) + \
                  (WEIGHT_PERC*ensemble_model2_pred_probs[:, 1])) >= PRED_THRESH, 1, 0)

    ## Partially balanced model (Best model, tuned by GridSearch) performance on validation set
    best_model_results["roc_auc_score"] = \
        roc_auc_score(self.y_val, best_model_preds)
    best_model_results["recall_score"] = \
        recall_score(self.y_val, best_model_preds)
    best_model_results["confusion_matrix"] = \
        confusion_matrix(self.y_val, best_model_preds)
    best_model_results["classification_report"] = \
        classification_report(self.y_val, best_model_preds)
    self.ensemble_experiment_results['solo_best_model'] = best_model_results

    weighted_model_results["roc_auc_score"] = \
        roc_auc_score(self.y_val, ensemble_model_preds)
    weighted_model_results["recall_score"] = \
        recall_score(self.y_val, ensemble_model_preds)
    weighted_model_results["confusion_matrix"] = \
        confusion_matrix(self.y_val, ensemble_model_preds)
    weighted_model_results["classification_report"] = \
        classification_report(self.y_val, ensemble_model_preds)
    self.ensemble_experiment_results['weighted_best_model'] = weighted_model_results

    # compare model averaging ensemble output against the solo best model variation
    class_report1 = best_model_results["classification_report"]
    print(f"Best Model Classification Report:\n{class_report1}\n")
    class_report2 = weighted_model_results["classification_report"]
    print(f"Weighted Model Classification Report:\n{class_report2}\n")


def get_stacked_ensemble_results(self):

    stacked_model_results = dict()
    stacked_model_weights = dict()

    lr = LogisticRegression(C=1.0, class_weight={0:1.0, 1:2.0})

    ## Training - 
    # Concatenating the probability predictions of the 2 models on train set
    X_train_stacked = \
        np.c_[self.ensemble_experiment_dict['no_target_balancing']['train']['pred_probs'][:, 1],  
              self.ensemble_experiment_dict['full_target_balancing']['train']['pred_probs'][:, 1]]

    # Fit stacker model on top of outputs of base model
    lr.fit(X_train_stacked, self.y_train)

    ## Prediction - 
    # Concatenating outputs from both the base models on the validation set
    X_val_stacked = \
        np.c_[self.ensemble_experiment_dict['no_target_balancing']['val']['pred_probs'][:, 1],
              self.ensemble_experiment_dict['full_target_balancing']['val']['pred_probs'][:, 1]]

    # Predict using the stacker model
    stacking_ensemble_preds = lr.predict(X_val_stacked)

    ## Ensemble model prediction on validation set
    stacked_model_results['roc_auc_score'] = roc_auc_score(self.y_val, stacking_ensemble_preds)
    stacked_model_results['recall_score'] = recall_score(self.y_val, stacking_ensemble_preds)
    stacked_model_results['confusion_matrix'] = confusion_matrix(self.y_val, stacking_ensemble_preds)
    stacked_model_results['classification_report'] = classification_report(self.y_val, stacking_ensemble_preds)

    # Stacked model weights
    stacked_model_weights['coef'] = lr.coef_
    stacked_model_weights['intercept'] = lr.intercept_

    # self.ensemble_experiment_results['stacked_best_model'] = stacked_model_results

    class_report = stacked_model_results["classification_report"]
    print(f'Stacked Model Classification Report: \n{class_report}\n')


def initialize_ensemble_model_pipelines(self, scaling=None):

    ## Three versions of the most effective models with best params for F1-score metric
    lgb1 = LGBMClassifier(boosting_type='dart',
                          class_weight={0: 1, 1: 1},
                          min_child_samples=20,
                          n_jobs=-1,
                          importance_type='gain',
                          max_depth=4,
                          num_leaves=31,
                          colsample_bytree=0.6,
                          learning_rate=0.1,
                          n_estimators=21,
                          reg_alpha=0,
                          reg_lambda=0.5)

    lgb2 = LGBMClassifier(boosting_type='dart',
                          class_weight={0: 1, 1: 3.93},
                          min_child_samples=20,
                          n_jobs=-1,
                          importance_type='gain',
                          max_depth=6,
                          num_leaves=63,
                          colsample_bytree=0.6,
                          learning_rate=0.1,
                          n_estimators=201,
                          reg_alpha=1,
                          reg_lambda=1)

    # Best class_weight parameter settings (partial class imbalance correction)
    lgb3 = LGBMClassifier(boosting_type='dart',
                          class_weight={0: 1, 1: 3.0},
                          min_child_samples=20,
                          n_jobs=-1,
                          importance_type='gain',
                          max_depth=6,
                          num_leaves=63,
                          colsample_bytree=0.6,
                          learning_rate=0.1,
                          n_estimators=201,
                          reg_alpha=1,
                          reg_lambda=1)

    if scaling:
        model_pipe_1 = self.add_pipeline_steps(lgb1, True)
        model_pipe_2 = self.add_pipeline_steps(lgb2, True)
        model_pipe_3 = self.add_pipeline_steps(lgb3, True)
    else:
        model_pipe_1 = self.add_pipeline_steps(lgb1)
        model_pipe_2 = self.add_pipeline_steps(lgb2)
        model_pipe_3 = self.add_pipeline_steps(lgb3)
    
    if not self.ensemble_experiment_dict:
        mod_pipe1_dict = {'pipe': model_pipe_1}
        self.ensemble_experiment_dict['no_target_balancing'] = mod_pipe1_dict
        mod_pipe2_dict = {'pipe': model_pipe_2}
        self.ensemble_experiment_dict['full_target_balancing'] = mod_pipe2_dict
        mod_pipe3_dict = {'pipe': model_pipe_3}
        self.ensemble_experiment_dict['partial_target_balancing'] = mod_pipe3_dict
    else:
        self.ensemble_experiment_dict['no_target_balancing']['pipe'] = model_pipe_1
        self.ensemble_experiment_dict['full_target_balancing']['pipe'] = model_pipe_2
        self.ensemble_experiment_dict['partial_target_balancing']['pipe'] = model_pipe_3


def ensemble_model_experiments(self):
    
    self.initialize_ensemble_model_pipelines()
    self.get_best_model_predictions('train')
    self.get_best_model_correlations()  # find least correlated models as ensemble inputs
    self.initialize_ensemble_model_pipelines()
    self.get_best_model_predictions('val')
    self.get_weighted_ensemble_results()
    self.get_stacked_ensemble_results()
