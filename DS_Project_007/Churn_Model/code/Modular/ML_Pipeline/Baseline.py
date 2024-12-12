'''from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, f1_score, recall_score, confusion_matrix, classification_report
from sklearn.decomposition import PCA
from sklearn import tree
#from sklearn.tree import DecisionTreeClassifier
#from sklearn.tree import export_graphviz
#from IPython.display import Image

import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn import tree
from sklearn.decomposition import PCA
from sklearn.metrics import (roc_auc_score,
                             f1_score,
                             recall_score,
                             confusion_matrix,
                             classification_report)

from pathlib import Path
import os
import sys

module_path = Path(__file__).parents[1]
sys.path.append(str(module_path))


def create_weight_dict(self):

    # Obtaining class weights based on the 'num churn class samples' imbalance ratio
    _, num_samples = np.unique(self.y_train, return_counts=True) 
    weights = np.max(num_samples)/num_samples

    weights_dict = dict()
    class_labels = [0, 1]
    for label, weight in zip(class_labels, weights):
        weights_dict[label] = weight

    self.train_weight_dict = weights_dict


def initialize_baseline_models(self):

    ## Defining the baseline LR Model
    self.LR_Baseline_Model = LogisticRegression(C=1.0,
                                                penalty='l2',
                                                class_weight=self.train_weight_dict,
                                                n_jobs=-1)

    ## Defining the baseline SVM Model
    self.SVC_Baseline_Model = SVC(C=1.0,
                                  kernel='linear',
                                  class_weight=self.train_weight_dict)

    ## Transforming the dataset using PCA (linear and nonlinear)
    pca = PCA(n_components=2)

    self.pca_X_train_logreg_selected = \
        pca.fit_transform(self.X_train[self.logreg_selected_feats].values)
    self.pca_X_train_variance_ratio = pca.explained_variance_ratio_
    print(f"Variance captured by 2D PCA w/ log reg selected features: \
{self.pca_X_train_variance_ratio}")

    self.pca_X_train_dectree_selected = \
        pca.fit_transform(self.train_data[self.dectree_selected_feats].values)
    self.pca_X_train_dectree_selected_variance_ratio = pca.explained_variance_ratio_
    print(f"Variance captured by 2D PCA w/ decision tree selected features: \
{self.pca_X_train_variance_ratio}")

    ## Define the Decision Tree model
    self.DT_Baseline_Model = \
        tree.DecisionTreeClassifier(criterion='entropy',
                                    class_weight=self.train_weight_dict,
                                    max_depth=4,
                                    max_features=None,
                                    min_samples_split=25,
                                    min_samples_leaf=15)


def evaluate_baseline_models(self):

    baseline_results_dict =\
          {'LR': {'model_info': {'model': None,
                                 'coefs': None,
                                 'intercepts': None,
                                 'pca_model': None},
                  'train_results': {'roc_auc': None,
                                    'recall_score': None,
                                    'confusion_matrix': None,
                                    'classification_report': None}, 
                  'val_results': {'roc_auc': None,
                                  'recall_score': None,
                                  'confusion_matrix': None,
                                  'classification_report': None}
                   },
           'SVC': {'model_info': {'model': None,
                                  'coefs': None,
                                  'intercepts': None,
                                  'pca_model': None},
                   'train_results': {'roc_auc': None,
                                     'recall_score': None,
                                     'confusion_matrix': None,
                                     'classification_report': None}, 
                   'val_results': {'roc_auc': None,
                                   'recall_score': None,
                                   'confusion_matrix': None,
                                   'classification_report': None}
                   },
           'DT': {'train_results': {'roc_auc': None,
                                    'recall_score': None,
                                    'confusion_matrix': None,
                                    'classification_report': None}, 
                'val_results': {'roc_auc': None,
                                'recall_score': None,
                                'confusion_matrix': None,
                                'classification_report': None}
                  }
                 }
    
    prediction_dict = {'LR': {'prediction_points': None},
                       'SVC': {'prediction_points': None}}

    # Creating a mesh plane region used to illustrate the LR and SVM decision boundaries
    x_min, x_max = self.pca_X_train_logreg_selected[:, 0].min() - 1, self.pca_X_train_logreg_selected[:, 0].max() + 1
    y_min, y_max = self.pca_X_train_logreg_selected[:, 1].min() - 1, self.pca_X_train_logreg_selected[:, 1].max() + 1
    mesh_xx, mesh_yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

    sel_logreg_cont_vars = \
        [feat for feat in self.logreg_selected_feats if feat in self.cont_transformed_vars]
    sel_logreg_cat_vars = \
        [feat for feat in self.logreg_selected_feats if feat in self.cat_transformed_vars]
    print(f"\nmodel parameters selected from log reg: {sel_logreg_cont_vars + sel_logreg_cat_vars}")

    for model in ['LR', 'SVC']:

        model_name = model + '_Baseline_Model'
        baseline_model = self.__getattribute__(model_name)

        ######### logistic regression baseline training results ###########
        baseline_model.fit(self.X_train, self.y_train)
        baseline_results_dict[model]['model_info']['model'] = baseline_model

        print(f"{model} baseline training model coefficients: {baseline_model.coef_}")
        baseline_results_dict[model]['model_info']['coefs'] = baseline_model.coef_

        print(f"{model}  baseline training model intercepts: {baseline_model.intercept_}")
        baseline_results_dict[model]['model_info']['intercepts'] = baseline_model.intercept_

        X_train_baseline_predictions = baseline_model.predict(self.X_train)

        train_rac = roc_auc_score(self.y_train, X_train_baseline_predictions)
        baseline_results_dict[model]['train_results']['roc_auc'] = train_rac
        train_rs = recall_score(self.y_train, X_train_baseline_predictions)
        baseline_results_dict[model]['train_results']['recall_score'] = train_rs
        train_cm = confusion_matrix(self.y_train, X_train_baseline_predictions)
        baseline_results_dict[model]['train_results']['confusion_matrix'] = train_cm
        train_cr = classification_report(self.y_train, X_train_baseline_predictions)
        baseline_results_dict[model]['train_results']['classification_report'] = train_cr
        print(f"train_rac: {train_rac}, train_rs: {train_rs}, train_cm: {train_cm}, train_cr: {train_cr}")

        ######## logistic regression baseline validation results ##########
        baseline_model.fit(self.X_val, self.y_val)
        baseline_results_dict[model]['model_info']['model'] = baseline_model

        print(f"{model}  baseline model validation coefficients: {baseline_model.coef_}")
        baseline_results_dict[model]['model_info']['coefs'] = baseline_model.coef_

        print(f"{model}  baseline model validation intercepts: {baseline_model.intercept_}")
        baseline_results_dict[model]['model_info']['intercepts'] = baseline_model.intercept_

        X_val_baseline_predictions = baseline_model.predict(self.X_val)

        val_rac = roc_auc_score(self.y_val, X_val_baseline_predictions)
        baseline_results_dict[model]['val_results']['roc_auc'] = val_rac
        val_rs = recall_score(self.y_val, X_val_baseline_predictions)
        baseline_results_dict[model]['val_results']['recall_score'] = val_rs
        val_cm = confusion_matrix(self.y_val, X_val_baseline_predictions)
        baseline_results_dict[model]['val_results']['confusion_matrix'] = val_cm
        val_cr = classification_report(self.y_val, X_val_baseline_predictions)
        baseline_results_dict[model]['val_results']['classification_report'] = val_cr
        print(f"{model} val_rac: {val_rac}, \
{model} val_rs: {val_rs}, \
{model} val_cm: {val_cm}, \
{model} val_cr{val_cr}")

        # Fit the logistic regression baseline model on the PCA reduced data
        baseline_model.fit(self.pca_X_train_logreg_selected, self.y_train)
        baseline_results_dict[model]['model_info']['pca_model'] = baseline_model

        ## Capturing logistic regression baseline predictions for the PCA reduced model
        pca_pred_points = baseline_model.predict(np.c_[mesh_xx.ravel(), mesh_yy.ravel()])
        reshaped_pca_pred_points = pca_pred_points.reshape(mesh_xx.shape)
        prediction_dict[model]['prediction_points'] = reshaped_pca_pred_points

    ##################### Linear Baseline Model Output Visualizations #####################
    
    plt.contourf(mesh_xx, mesh_yy, prediction_dict['LR']['prediction_points'], alpha=0.4)
    plt.contour(mesh_xx, mesh_yy, prediction_dict['SVC']['prediction_points'], alpha=0.4,
                colors='blue')
    sns.scatterplot(self.pca_X_train_logreg_selected[:,0],
                    self.pca_X_train_logreg_selected[:,1],
                    hue=self.y_train,
                    s=50,
                    alpha=0.8)
    plt.title('Linear model Decision Boundaries - Logistic Regression and Support Vector Machine')
    plt.savefig(os.path.join(self.output_path, 'Decision_Boundaries/linear_decision_boundaries.png'),
                orientation='landscape')
    plt.clf()


    ################# Decision Tree baseline training results #################

    sel_dectree_cont_vars = \
        [feat for feat in self.dectree_selected_feats if feat in self.cont_transformed_vars]
    sel_dectree_cat_vars = \
        [feat for feat in self.dectree_selected_feats if feat in self.cat_transformed_vars]
    print(f"\nmodel parameters selected from decision tree: \
{sel_dectree_cont_vars + sel_dectree_cat_vars}")
     
    X_train_dt = self.train_data[self.dectree_selected_feats]
    X_train_baseline_model_ = self.DT_Baseline_Model.fit(X_train_dt, self.y_train)
    X_train_baseline_predictions_ = X_train_baseline_model_.predict(X_train_dt)
    X_train_baseline_predictions_list = X_train_baseline_predictions_.tolist()

    # Checking the importance of different features of the model
    feature_importance = pd.DataFrame({'features': self.dectree_selected_feats,
                                       'importance': self.DT_Baseline_Model.feature_importances_
                                       }).sort_values(by = 'importance', ascending=False)
    print(f"Decision Tree Selected Feature Importance: {feature_importance}")

    train_rac_ = roc_auc_score(self.y_train, X_train_baseline_predictions_list)
    baseline_results_dict['DT']['train_results']['roc_auc'] = train_rac_
    train_rs_ = recall_score(self.y_train, X_train_baseline_predictions_list)
    baseline_results_dict['DT']['train_results']['recall_score'] = train_rs_
    train_cm_ = confusion_matrix(self.y_train, X_train_baseline_predictions_list)
    baseline_results_dict['DT']['train_results']['confusion_matrix'] = train_cm_
    train_cr_ = classification_report(self.y_train, X_train_baseline_predictions_list)
    baseline_results_dict['DT']['train_results']['classification_report'] = train_cr_
    print(f"dt train_rac: {train_rac_}, \
dt train_rs: {train_rs_}, \
dt train_cm: {train_cm_}, \
dt train_cr: {train_cr_}")

    ############ Decision Tree baseline validation results #############

    X_val_dt = self.val_data[self.dectree_selected_feats]
    X_val_baseline_model_ = self.DT_Baseline_Model.fit(X_val_dt, self.y_val)
    X_val_baseline_predictions_ = X_val_baseline_model_.predict(X_val_dt)
    X_val_baseline_predictions_list = X_val_baseline_predictions_.tolist()

    val_rac_ = roc_auc_score(self.y_val, X_val_baseline_predictions_list)
    baseline_results_dict['DT']['val_results']['roc_auc'] = val_rac_
    val_rs_ = recall_score(self.y_val, X_val_baseline_predictions_list)
    baseline_results_dict['DT']['val_results']['recall_score'] = val_rs_
    val_cm_ = confusion_matrix(self.y_val, X_val_baseline_predictions_list)
    baseline_results_dict['DT']['val_results']['confusion_matrix'] = val_cm_
    val_cr_ = classification_report(self.y_val, X_val_baseline_predictions_list)
    baseline_results_dict['DT']['val_results']['classification_report'] = val_cr_
    print(f"dt val rac: {val_rac_}, \
dt val rs: {val_rs_}, \
dt val cm: {val_cm_}, \
dt val cr: {val_cr_}")
    

    ############### Non-Linear Baseline Model Output Visualizations ###################

    # Decision Boundaries visualized in two dimensions
    x_min, x_max = self.pca_X_train_dectree_selected[:, 0].min() - 1, self.pca_X_train_dectree_selected[:, 0].max() + 1
    y_min, y_max = self.pca_X_train_dectree_selected[:, 1].min() - 1, self.pca_X_train_dectree_selected[:, 1].max() + 1
    mesh_xx_, mesh_yy_ = np.meshgrid(np.arange(x_min, x_max, 100), np.arange(y_min, y_max, 100))

    X_train_baseline_model_ = self.DT_Baseline_Model.fit(self.pca_X_train_dectree_selected, self.y_train)
    X_train_baseline_predictions_ = \
        X_train_baseline_model_.predict(np.c_[mesh_xx_.ravel(), mesh_yy_.ravel()])
    X_train_baseline_predictions_ = X_train_baseline_predictions_.reshape(mesh_xx_.shape)
    
    plt.contourf(mesh_xx_, mesh_yy_, X_train_baseline_predictions_, alpha=0.4) # DT
    sns.scatterplot(self.pca_X_train_dectree_selected[:,0],
                    self.pca_X_train_dectree_selected[:,1],
                    hue=self.y_train,
                    s=50,
                    alpha=0.8)
    plt.title('Non-Linear Decision Boundaries - Decision Tree')
    plt.savefig(os.path.join(self.output_path, "Decision_Boundaries/nonlinear_decision_boundary.png"))
    plt.clf()

    # Decision Tree Visualization: Change the max-depth to 3 for viz purposes
    clf = tree.DecisionTreeClassifier(criterion='entropy',
                                      class_weight=self.train_weight_dict,
                                      max_depth=3,
                                      max_features=None,
                                      min_samples_split=25,
                                      min_samples_leaf=15)

    clf.fit(self.train_data[self.dectree_selected_feats],
                                     self.y_train)
    plt.figure(figsize=(15, 10))
    tree.plot_tree(clf, filled=True)
    plt.savefig(os.path.join(self.output_path, "Decision_Boundaries/dt_visualization.png"))
    plt.clf()

    # update the baseline model results dictionary
    self.__setattr__('baseline_results_dict', baseline_results_dict)
