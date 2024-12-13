import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (roc_auc_score,
                             f1_score,
                             recall_score,
                             confusion_matrix,
                             classification_report)
from lightgbm import LGBMClassifier
import os


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
    sns.violinplot(error_checking_df.Exited.ravel(),
                   error_checking_df['Pred_Probs'].values)
    plt.savefig(os.path.join(self.output_path,
                             "Error_Analysis/pred_probs_violinplot_.png"))
    plt.clf()

    ## Check churn distribution with respect to age
    sns.boxplot(x='Exited', y='Age', data=error_checking_df)
    plt.savefig(os.path.join(self.output_path,
                             "Error_Analysis/age_churn_boxplot_.png"))
    plt.clf()

    ## Attempting to correctly identify pockets of high-churn customer regions in feature space
    churn_ratio = \
        error_checking_df.Exited.value_counts(normalize=True).sort_index()
    print(f'Churn ratio: \n{churn_ratio}\n')

    target_age_churn_ratio = \
        error_checking_df[(error_checking_df.Age > 42) &
                          (error_checking_df.Age < 53)].Exited.value_counts(normalize=True
                                                                            ).sort_index()
    print(f'Targeted age range churn ratio: \n{target_age_churn_ratio}\n')

    target_age_churn_ratio_pred = \
        error_checking_df[(error_checking_df.Age > 42) &
                          (error_checking_df.Age < 53)].Predictions.value_counts(normalize=True
                                                                                 ).sort_index()
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
    num_products_ratio = \
        error_checking_df.NumOfProducts.value_counts(normalize=True).sort_index()
    num_products_low_recall_ratio = \
        low_recall.NumOfProducts.value_counts(normalize=True).sort_index()
    num_products_low_precision_ratio = \
        low_prec.NumOfProducts.value_counts(normalize=True).sort_index()

    print(f'number of products ratio: \n{num_products_ratio}\n')
    print(f'number of products ratio in low recall \
set: \n{num_products_low_recall_ratio}\n')
    print(f'number of products ratio in low precision \
set: \n{num_products_low_precision_ratio}\n')

    # Examining low recall and low precision for 'IsActiveMember'
    isActive_ratio = \
        error_checking_df.IsActiveMember.value_counts(normalize=True).sort_index()
    isActive_low_recall_ratio = \
        low_recall.IsActiveMember.value_counts(normalize=True).sort_index()
    isActive_low_precision_ratio = \
        low_prec.IsActiveMember.value_counts(normalize=True).sort_index()

    print(f'number of products ratio: \n{isActive_ratio}\n')
    print(f'number of products ratio in low recall \
set: \n{isActive_low_recall_ratio}\n')
    print(f'number of products ratio in low precision \
set: \n{isActive_low_precision_ratio}\n')

    ## Age distribution comparisons
    sns.violinplot(y = error_checking_df.Age)
    plt.savefig(os.path.join(self.output_path,
                             "Error_Analysis/age_dist_violinplot_.png"))
    plt.clf()

    sns.violinplot(y = low_recall.Age)
    plt.savefig(os.path.join(self.output_path,
                             "Error_Analysis/low_recall_age_dist_violinplot_.png"))
    plt.clf()

    sns.violinplot(y = low_prec.Age)
    plt.savefig(os.path.join(self.output_path,
                             "Error_Analysis/low_precision_age_dist_violinplot_.png"))
    plt.clf()


    ## Balance distribution comparisons
    sns.violinplot(y = error_checking_df.Balance)
    plt.savefig(os.path.join(self.output_path,
                             "Error_Analysis/balance_dist_violinplot_.png"))
    plt.clf()

    sns.violinplot(y = low_recall.Balance)
    plt.savefig(os.path.join(self.output_path,
                             "Error_Analysis/low_recall_balance_dist_violinplot_.png"))
    plt.clf()

    sns.violinplot(y = low_prec.Balance)
    plt.savefig(os.path.join(self.output_path,
                             "Error_Analysis/low_precision_balance_dist_violinplot_.png"))
    plt.clf()


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
    val_preds = np.where(val_probs > 0.45, 1, 0)

    # Churn distribution visualization
    sns.boxplot(y.ravel(), val_probs)
    plt.savefig(os.path.join(self.output_path,
                             "Final_Model/churn_dist_boxplot_.png"))
    plt.clf()

    ## Validation metrics
    ras = roc_auc_score(y, val_preds)
    rs = recall_score(y, val_preds)
    cm = confusion_matrix(y, val_preds)
    cr = classification_report(y, val_preds)

    self.final_validation_metrics = {'roc_auc_score': ras,
                                     'recall_score': rs,
                                     'confusion_matrix': cm,
                                     'classification_report': cr}

    # Output SHAP Feature Importance Analysis
    shap.initjs()

    X_transformed = \
        model_pipe.named_steps['categorical_encoding'].fit_transform(X, y)
    X_shap = \
        model_pipe.named_steps['add_new_features'].fit_transform(X_transformed, y)

    explainer = shap.TreeExplainer(self.final_model)

    ## Explain global patterns/ summary stats
    shap_values = explainer.shap_values(X_shap)
    shap.summary_plot(shap_values, X_shap, show=False)
    plt.savefig(os.path.join(self.output_path, "Final_Model/SHAP_.png"))
    plt.clf()
