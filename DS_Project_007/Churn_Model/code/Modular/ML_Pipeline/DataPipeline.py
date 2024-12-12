import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import (roc_auc_score,
                             f1_score,
                             recall_score,
                             confusion_matrix,
                             classification_report)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, ComplementNB, BernoulliNB
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from pathlib import Path
import sys
import os

module_path = Path(__file__).parents[1]
sys.path.append(str(module_path))

from ML_Pipeline.CategoricalEncoder import CategoricalEncoder
from ML_Pipeline.AddFeatures import AddFeatures
from ML_Pipeline.CustomScaler import CustomScaler


## Preparing a list of models to try out in the spot-checking process
def initialize_spotcheck_models(self):

    # Tree models
    for n_trees in [21, 1001]:
        self.spotcheck_models_dict['rf_' + str(n_trees)] = \
            RandomForestClassifier(n_estimators=n_trees,
                                   n_jobs=-1,
                                   criterion='entropy',
                                   class_weight=self.train_weight_dict,
                                   max_depth=6,
                                   max_features=0.6,
                                   min_samples_split=30,
                                   min_samples_leaf=20)
    
        self.spotcheck_models_dict['lgb_' + str(n_trees)] = \
            LGBMClassifier(boosting_type='dart',
                           num_leaves=31,
                           max_depth=6,
                           learning_rate=0.1,
                           n_estimators=n_trees,
                           class_weight=self.train_weight_dict,
                           min_child_samples=20,
                           colsample_bytree=0.6,
                           reg_alpha=0.3,
                           reg_lambda=1.0,
                           n_jobs=-1,
                           importance_type='gain')
    
        self.spotcheck_models_dict['xgb_' + str(n_trees)] = \
            XGBClassifier(objective='binary:logistic',
                          n_estimators=n_trees,
                          max_depth=6,
                          learning_rate=0.03,
                          n_jobs=-1,
                          colsample_bytree=0.6,
                          reg_alpha=0.3,
                          reg_lambda=0.1,
                          scale_pos_weight=self.train_weight_dict[1])
    
        self.spotcheck_models_dict['et_' + str(n_trees)] = \
            ExtraTreesClassifier(n_estimators=n_trees,
                                 criterion='entropy',
                                 max_depth=6,
                                 max_features=0.6,
                                 n_jobs=-1,
                                 class_weight=self.train_weight_dict,
                                 min_samples_split=30,
                                 min_samples_leaf=20)

    # kNN models
    for n in [3, 5, 11]:
        self.spotcheck_models_dict['knn_' + str(n)] = \
            KNeighborsClassifier(n_neighbors=n)

    # Naive-Bayes models
    self.spotcheck_models_dict['gauss_nb'] = GaussianNB()
    self.spotcheck_models_dict['multi_nb'] = MultinomialNB()
    self.spotcheck_models_dict['compl_nb'] = ComplementNB()
    self.spotcheck_models_dict['bern_nb'] = BernoulliNB()


def add_pipeline_steps(self, model, scale_features=None):

    if scale_features:
        result = Pipeline(steps = [('categorical_encoding', CategoricalEncoder()),
                                   ('add_new_features', AddFeatures()),
                                   ('standard_scaling', CustomScaler(self.cols_to_scale)),
                                   ('classifier', model)
                                   ])
    else:
        result = Pipeline(steps = [('categorical_encoding', CategoricalEncoder()),
                                   ('add_new_features', AddFeatures()),
                                   ('classifier', model)
                                   ])

    return result


def make_pipeline(self, model):

    if (str(model).find('KNeighborsClassifier') != -1):
        pipe = self.add_pipeline_steps(model, True)
    else:
        pipe = self.add_pipeline_steps(model)
    
    return pipe


def evaluate_spotcheck_models(self, score_metric='recall'):

    # Evaluate model through automated pipelines
    for name, model in self.spotcheck_models_dict.items():
        curr_model = self.make_pipeline(model)
        scores = cross_val_score(curr_model,
                                 self.train_data.drop(columns=self.target_var,
                                                      axis=1),
                                 self.y_train,
                                 cv=5,
                                 scoring=score_metric,
                                 n_jobs=-1)
        # Store results of the evaluated model
        self.spotcheck_results_dict[name] = scores
        mu, sigma = np.mean(scores), np.std(scores)

        # Printing individual model results
        print('Model {}: mean = {}, std_dev = {}' .format(name, mu, sigma))


def rand_tuning(self, params, iters, scoring, scaling=False):

    if scaling:
        model = self.add_pipeline_steps(self.tuning_model, True)
    else:
        model = self.add_pipeline_steps(self.tuning_model)

    search = \
        RandomizedSearchCV(model, params, n_iter=iters, cv=5, scoring=scoring)
    search.fit(self.train_data.drop(columns=self.target_var, axis=1),
               self.y_train)

    result_dictionary = {'best_params': search.best_params_,
                         'best_score': search.best_score_,
                         'cross_val_results': search.cv_results_}

    self.tuning_result_dict['rand_tuning'] = result_dictionary


def grid_tuning(self, params, scoring, scaling=False):

    if scaling:
        model = self.add_pipeline_steps(self.tuning_model, True)
    else:
        model = self.add_pipeline_steps(self.tuning_model)

    grid = GridSearchCV(model, params, cv=5, scoring=scoring, n_jobs=-1)
    grid.fit(self.train_data.drop(columns=self.target_var, axis=1),
               self.y_train)
    
    result_dictionary = {'best_params': grid.best_params_,
                         'best_score': grid.best_score_,
                         'cross_val_results': grid.cv_results_}

    self.tuning_result_dict['grid_tuning'] = result_dictionary


def initialize_tuning_model(self):

    self.tuning_model = \
        LGBMClassifier(boosting_type='dart',
                       num_leaves=31,
                       min_child_samples=20,
                       n_jobs=-1,
                       importance_type='gain')


def hyperparameter_tuning(self):

    TUNING_METRIC = 'f1'

    rand_iters = 20
    rand_scoring = TUNING_METRIC
    rand_params = {'classifier__n_estimators':[10, 21, 51, 100,
                                               201, 350, 501]
                   ,'classifier__max_depth': [3, 4, 6, 9]
                   ,'classifier__num_leaves':[7, 15, 31] 
                   ,'classifier__learning_rate': [0.03, 0.05,
                                                  0.1, 0.5, 1]
                   ,'classifier__colsample_bytree': [0.3, 0.6, 0.8]
                   ,'classifier__reg_alpha': [0, 0.3, 1, 5]
                   ,'classifier__reg_lambda': [0.1, 0.5, 1, 5, 10]
                   ,'classifier__class_weight': [{0:1,1:1.0},
                                                 {0:1,1:1.96},
                                                 {0:1,1:3.0},
                                                 {0:1,1:3.93}]
                 }

    grid_scoring = TUNING_METRIC
    grid_params = {'classifier__n_estimators':[201]
                 ,'classifier__max_depth': [6]
                 ,'classifier__num_leaves': [63]
                 ,'classifier__learning_rate': [0.1]
                 ,'classifier__colsample_bytree': [0.6, 0.8]
                 ,'classifier__reg_alpha': [0, 1, 10]
                 ,'classifier__reg_lambda': [0.1, 1, 5]
                 ,'classifier__class_weight': [{0:1,1:3.0}]
                 }
    
    self.initialize_tuning_model()

    self.rand_tuning(rand_params, rand_iters, rand_scoring)
    self.grid_tuning(grid_params, grid_scoring)

    print(self.tuning_result_dict)


def initialize_ensemble_model_pipelines(self, scaling=None):

    ## Ensembles - Evaluate Best Model Averaging and Stacking
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


def get_best_model_predictions(self, type='train'):

    data_attribute_name = type + '_data'
    dataset = self.__getattribute__(data_attribute_name)
    input_data = dataset.drop(columns=self.target_var, axis=1)

    target_name = 'y_' + type
    target_var = self.__getattribute__(target_name)

    self.ensemble_experiment_dict['no_target_balancing']['pipe'].fit(input_data, target_var)
    mod_preds1_dict = \
        {'pred_probs': 
         self.ensemble_experiment_dict['no_target_balancing']['pipe'].predict_proba(input_data)}
    self.ensemble_experiment_dict['no_target_balancing'][type] = mod_preds1_dict

    self.ensemble_experiment_dict['full_target_balancing']['pipe'].fit(input_data, target_var)
    mod_preds2_dict = \
        {'pred_probs': 
         self.ensemble_experiment_dict['full_target_balancing']['pipe'].predict_proba(input_data)}
    self.ensemble_experiment_dict['full_target_balancing'][type] = mod_preds2_dict

    self.ensemble_experiment_dict['partial_target_balancing']['pipe'].fit(input_data, target_var)
    mod_preds3_dict = \
        {'pred_probs': 
         self.ensemble_experiment_dict['partial_target_balancing']['pipe'].predict_proba(input_data)}
    self.ensemble_experiment_dict['partial_target_balancing'][type] = mod_preds3_dict


def get_best_model_correlations(self):

    ## Checking correlations between the predictions of the 3 models
    ensemble_model_correlations = \
        pd.DataFrame({'unbal_preds': 
                      self.ensemble_experiment_dict['no_target_balancing']['train']['pred_probs'][:, 1],
                      'bal_preds': 
                      self.ensemble_experiment_dict['full_target_balancing']['train']['pred_probs'][:, 1],
                      'semibal_preds': 
                      self.ensemble_experiment_dict['partial_target_balancing']['train']['pred_probs'][:, 1]
                      })

    print(f"Best model variations prediction probability correlation \
matrix: \n{ensemble_model_correlations.corr()}\n")


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


def ensemble_model_experiments(self):
    
    self.initialize_ensemble_model_pipelines()
    self.get_best_model_predictions('train')
    self.get_best_model_correlations()  # find least correlated models as ensemble inputs
    self.initialize_ensemble_model_pipelines()
    self.get_best_model_predictions('val')
    self.get_weighted_ensemble_results()
    self.get_stacked_ensemble_results()


def initialize_final_model(self):

    ## Final model with best params for F1-score metric
    self.final_model = LGBMClassifier(boosting_type='dart',
                                      class_weight={0: 1.0, 1: 3.0},
                                      min_child_samples= 20,
                                      n_jobs=-1,
                                      importance_type='gain',
                                      max_depth=6, 
                                      num_leaves=63,
                                      colsample_bytree=0.6,
                                      learning_rate=0.1,
                                      n_estimators=201,
                                      reg_alpha=1,
                                      reg_lambda=1)


def error_checking_metrics(self, scaling=False):

    if scaling:
        model_pipe = self.add_pipeline_steps(self.final_model, True)
    else:
        model_pipe = self.add_pipeline_steps(self.final_model)

    # Unscaled features will be used since it's a tree model
    X_train = self.train_data.drop(self.target_var, axis=1)
    X_val = self.val_data.drop(self.target_var, axis=1)

    ## Fit best model variation for error analysis
    model_pipe.fit(X_train, self.y_train)

    error_checking_df = self.val_data.copy() # todo: use a mask instead
    error_checking_df['Predictions'] = model_pipe.predict(X_val)
    error_checking_df['Pred_Probs'] = model_pipe.predict_proba(X_val)[:, 1]

    ## Making predictions on the validation set
    print(f"Error Analysis Sample Records: \n{error_checking_df.sample(10)}\n")

    ## Visualizing distribution of predicted probabilities   
    sns.violinplot(error_checking_df.Exited.ravel(), error_checking_df['Pred_Probs'].values)
    plt.savefig(os.path.join(self.output_path, "Error_Analysis/pred_probs_violinplot_.png"))
    plt.clf()

    ## Check churn distribution with respect to age
    sns.boxplot(x='Exited', y='Age', data=error_checking_df)
    plt.savefig(os.path.join(self.output_path, "Error_Analysis/age_churn_boxplot_.png"))
    plt.clf()

    ## Attempting to correctly identify pockets of high-churn customer regions in feature space
    churn_ratio = error_checking_df.Exited.value_counts(normalize=True).sort_index()
    print(f'Churn ratio: \n{churn_ratio}\n')

    target_age_churn_ratio = \
        error_checking_df[(error_checking_df.Age > 42) &
                          (error_checking_df.Age < 53)].Exited.value_counts(normalize=True).sort_index()
    print(f'Targeted age range churn ratio: \n{target_age_churn_ratio}\n')

    target_age_churn_ratio_pred = \
        error_checking_df[(error_checking_df.Age > 42) &
                          (error_checking_df.Age < 53)].Predictions.value_counts(normalize=True).sort_index()
    print(f'Targeted age range churn prediction ratio: \n{target_age_churn_ratio_pred}\n')

    ## Checking correlation between numeric features and target variable vs predicted variable
    feature_target_corr = error_checking_df[self.numeric_feats + ['Predictions', 'Exited']].corr()
    print(feature_target_corr[['Predictions','Exited']])

    ## Extracting the subset of incorrect predictions
    low_recall = error_checking_df[(error_checking_df.Exited == 1) &
                                   (error_checking_df.Predictions == 0)]
    low_prec = error_checking_df[(error_checking_df.Exited == 0) &
                                 (error_checking_df.Predictions == 1)]

    print(f'low recall data: \n{low_recall.head()}\n')
    print(f'low recall shape: {low_recall.shape}\n')

    print(f'low precision data: \n{low_prec.head()}\n')
    print(f'low precision shape: {low_prec.shape}\n')

    ## Visualize prediction probabilty distribution of errors causing low recall and low precision
    sns.displot(low_recall.Pred_Probs, kind='kde')
    plt.savefig(os.path.join(self.output_path,
                             "Error_Analysis/low_recall_pred_prob_distribution_.png"))
    plt.clf()

    sns.displot(low_prec.Pred_Probs, kind='kde')
    plt.savefig(os.path.join(self.output_path,
                             "Error_Analysis/low_precision_pred_prob_distribution_.png"))
    plt.clf()

    ## Tweaking the threshold of the classifier between .4 and .6
    steps = np.arange(.4, .61, .05, dtype=np.float16)
    for step in steps:

        threshold = step

        ## Predict on validation set with adjustable decision threshold
        probs = model_pipe.predict_proba(X_val)[:, 1]
        val_preds = np.where(probs > threshold, 1, 0)

        cm = confusion_matrix(self.y_val, val_preds)
        cr = classification_report(self.y_val, val_preds)

        print(f'error analysis - confusion matrix with\
{threshold} threshold: \n{cm}\n')
        print(f'error analysis - classification report matrix with\
{threshold} threshold: {cr}\n')
        
    ## Checking whether the model has too much dependence on certain features

    '''# Examining low recall and low precision for 'NumOfProducts'
    num_products_ratio = df_ea.NumOfProducts.value_counts(normalize=True).sort_index()
    num_products_low_recall_ratio = low_recall.NumOfProducts.value_counts(normalize=True).sort_index()
    num_products_low_precision_ratio = low_prec.NumOfProducts.value_counts(normalize=True).sort_index()

    print('number of products ratio: \n{}\n' .format(num_products_ratio))
    print('number of products ratio in low recall set: \n{}\n' .format(num_products_low_recall_ratio))
    print('number of products ratio in low precision set: \n{}\n' .format(num_products_low_precision_ratio))

    # Examining low recall and low precision for 'IsActiveMember'
    isActive_ratio = df_ea.IsActiveMember.value_counts(normalize=True).sort_index()
    isActive_low_recall_ratio = low_recall.IsActiveMember.value_counts(normalize=True).sort_index()
    isActive_low_precision_ratio = low_prec.IsActiveMember.value_counts(normalize=True).sort_index()

    print('number of products ratio: \n{}\n' .format(isActive_ratio))
    print('number of products ratio in low recall set: \n{}\n' .format(isActive_low_recall_ratio))
    print('number of products ratio in low precision set: \n{}\n' .format(isActive_low_precision_ratio))


    ## Age distribution comparisons
    output = "Churn_Model/code/Modular/Output/Error_Analysis/age_dist_violinplot.png"     
    sns.violinplot(y = df_ea.Age)
    plt.savefig(output)
    plt.clf()

    output = "Churn_Model/code/Modular/Output/Error_Analysis/low_recall_age_dist_violinplot.png"     
    sns.violinplot(y = low_recall.Age)
    plt.savefig(output)
    plt.clf()

    output = "Churn_Model/code/Modular/Output/Error_Analysis/low_precision_age_dist_violinplot.png"     
    sns.violinplot(y = low_prec.Age)
    plt.savefig(output)
    plt.clf()


    ## Balance distribution comparisons
    output = "Churn_Model/code/Modular/Output/Error_Analysis/balance_dist_violinplot.png"     
    sns.violinplot(y = df_ea.Balance)
    plt.savefig(output)
    plt.clf()

    output = "Churn_Model/code/Modular/Output/Error_Analysis/low_recall_balance_dist_violinplot.png"     
    sns.violinplot(y = low_recall.Balance)
    plt.savefig(output)
    plt.clf()

    output = "Churn_Model/code/Modular/Output/Error_Analysis/low_precision_balance_dist_violinplot.png"     
    sns.violinplot(y = low_prec.Balance)
    plt.savefig(output)
    plt.clf()'''
        

def model_performance_metrics(self, scaling=False):

    if scaling:
        model_pipe = self.add_pipeline_steps(self.final_model, True)
    else:
        model_pipe = self.add_pipeline_steps(self.final_model)

    ## Fitting final model on train dataset
    X = self.data.drop(columns=['Exited'], axis=1)
    y = self.data['Exited'].values
    model_pipe.fit(X, y)

    # Predict target probabilities
    val_probs = model_pipe.predict_proba(X)[:, 1]

    # Predict target values on val data
    val_preds = np.where(val_probs > 0.45, 1, 0) # The probability threshold can be tweaked

    # Churn distribution visualization
    sns.boxplot(y.ravel(), val_probs)
    plt.savefig(os.path.join(self.output_path, "Final_Model/churn_dist_boxplot_.png"))
    plt.clf()

    ## Validation metrics
    ras = roc_auc_score(y, val_preds)
    rs = recall_score(y, val_preds)
    cm = confusion_matrix(y, val_preds)
    cr = classification_report(y, val_preds)

    ##### todo: save the final validation metrics into a class attribute

    '''
    # Perform SHAP Analysis
    shap.initjs()

    ce = CategoricalEncoder()
    af = AddFeatures()

    X = ce.fit_transform(self.X, self.y)
    X = af.transform(X)

    raw_best_model.fit(X, self.y)

    explainer = shap.TreeExplainer(raw_best_model)

    row_num = 7
    shap_vals = explainer.shap_values(X.iloc[row_num].values.reshape(1,-1))

    #base value
    explainer.expected_value

    ## Explain single prediction
    # shap.force_plot(explainer.expected_value[1], shap_vals[1], X.iloc[row_num], link = 'logit')

    ## Check probability predictions through the model
    pred_probs = raw_best_model.predict_proba(X)[:,1]
    pred_probs[row_num]

    ## Explain global patterns/ summary stats
    output = "Churn_Model/code/Modular/Output/Final_Model/SHAP.png"     
    shap_values = explainer.shap_values(X)
    shap.summary_plot(shap_values, X, show=False)
    plt.savefig(output)
    plt.clf()

    return ras, rs, cm, cr'''


def save_final_model(self):

    ## Save model object
    joblib.dump(self.final_model,
                os.path.join(self.output_path,
                             "Final_Model/final_churn_model_v1_0.sav"))
