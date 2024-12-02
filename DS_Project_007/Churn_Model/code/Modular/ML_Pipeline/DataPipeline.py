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
from sklearn.metrics import roc_auc_score, f1_score, recall_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, ComplementNB, BernoulliNB
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from Churn_Model.code.Modular.ML_Pipeline.CategoricalEncoder import CategoricalEncoder
from Churn_Model.code.Modular.ML_Pipeline.AddFeatures import AddFeatures
from Churn_Model.code.Modular.ML_Pipeline.CustomScaler import CustomScaler




class DataPipeline:


    def __init__(self, data, base):

        ## Preparing data and a few common model parameters
        self.X = data.get_df_train().drop(columns = ['Exited'], axis = 1)
        self.y = data.get_y_train().ravel()

        self.X_val = data.get_df_val().drop(columns = ['Exited'], axis = 1)
        self.y_val = data.get_y_val().ravel()

        self.weights_dict = base.get_weights_dict()
        self.weight = self.get_weight(data)

        self.cols_to_scale = ['CreditScore', 'Age', 'Balance', 'EstimatedSalary'
                                              , 'bal_per_product', 'bal_by_est_salary'
                                              , 'tenure_age_ratio','age_surname_enc']


        ## Spot-checking in action
        self.models = self.model_zoo()
        self.recall_results = self.evaluate_models(self.X, self.y , self.models, metric = 'recall')
        self.f1_results = self.evaluate_models(self.X, self.y , self.models, metric = 'f1')



    def get_weight(self, pre_data):
        _, num_samples = np.unique(pre_data.get_y_train(), return_counts = True)
        return (num_samples[0]/num_samples[1]).round(2)


    ## Preparing a list of models to try out in the spot-checking process
    def model_zoo(self, models = dict()):
        # Tree models
        for n_trees in [21, 1001]:
            models['rf_' + str(n_trees)] = RandomForestClassifier(n_estimators = n_trees, n_jobs = -1, criterion = 'entropy'
                                                                  , class_weight = self.weights_dict, max_depth = 6, max_features = 0.6
                                                                  , min_samples_split = 30, min_samples_leaf = 20)
        
            models['lgb_' + str(n_trees)] = LGBMClassifier(boosting_type='dart', num_leaves=31, max_depth= 6, learning_rate=0.1
                                                           , n_estimators=n_trees, class_weight = self.weights_dict, min_child_samples=20
                                                           , colsample_bytree=0.6, reg_alpha=0.3, reg_lambda=1.0, n_jobs=- 1
                                                           , importance_type = 'gain')
        
            models['xgb_' + str(n_trees)] = XGBClassifier(objective='binary:logistic', n_estimators = n_trees, max_depth = 6
                                                          , learning_rate = 0.03, n_jobs = -1, colsample_bytree = 0.6
                                                          , reg_alpha = 0.3, reg_lambda = 0.1, scale_pos_weight = self.weight)
        
            models['et_' + str(n_trees)] = ExtraTreesClassifier(n_estimators=n_trees, criterion = 'entropy', max_depth = 6
                                                                , max_features = 0.6, n_jobs = -1, class_weight = self.weights_dict
                                                                , min_samples_split = 30, min_samples_leaf = 20)
    
        # kNN models
        for n in [3, 5, 11]:
            models['knn_' + str(n)] = KNeighborsClassifier(n_neighbors=n)
    
        # Naive-Bayes models
        models['gauss_nb'] = GaussianNB()
        models['multi_nb'] = MultinomialNB()
        models['compl_nb'] = ComplementNB()
        models['bern_nb'] = BernoulliNB()
    
        return models


    def add_pipeline_steps(self, model, scaling=False):

        if scaling:
            result =  Pipeline(steps = [('categorical_encoding', CategoricalEncoder())
                                        ,('add_new_features', AddFeatures())
                                        ,('standard_scaling', CustomScaler(self.cols_to_scale))
                                        ,('classifier', model)
                                        ])

        else:
            result =  Pipeline(steps = [('categorical_encoding', CategoricalEncoder())
                                        ,('add_new_features', AddFeatures())
                                        ,('classifier', model)
                                        ])

        return result


    ## Automation of data preparation and model run through pipelines
    def make_pipeline(self, model):
        '''
        Creates pipeline for the model passed as the argument. Uses standard scaling only in case of kNN models. 
        Ignores scaling step for tree/Naive Bayes models
        '''
    
        if (str(model).find('KNeighborsClassifier') != -1):
            pipe = self.add_pipeline_steps(model, True)

        else:
            pipe = self.add_pipeline_steps(model)
       
        return pipe


    ## Run/Evaluate all 15 models using KFold cross-validation (5 folds)
    def evaluate_models(self, X, y, models, folds = 5, metric = 'recall'):
        results = dict()
        for name, model in models.items():
            # Evaluate model through automated pipelines
            curr_model = self.make_pipeline(model)
            scores = cross_val_score(curr_model, X, y, cv = folds, scoring = metric, n_jobs = -1)
        
            # Store results of the evaluated model
            results[name] = scores
            mu, sigma = np.mean(scores), np.std(scores)
            # Printing individual model results
            print('Model {}: mean = {}, std_dev = {}'.format(name, mu, sigma))
    
        return results


    def rand_tuning(self, classifier, params, iters, scoring, scaling=False):

        if scaling:
            model = self.add_pipeline_steps(classifier, True)
        else:
            model = self.add_pipeline_steps(classifier)

        search = RandomizedSearchCV(model, params, n_iter = iters, cv = 5, scoring = scoring)
        search.fit(self.X, self.y)

        return search.best_params_, search.best_score_, search.cv_results_


    def grid_tuning(self, classifier, params, scoring, scaling=False):

        if scaling:
            model = self.add_pipeline_steps(classifier, True)
        else:
            model = self.add_pipeline_steps(classifier)


        grid = GridSearchCV(model, params, cv = 5, scoring = scoring, n_jobs = -1)
        grid.fit(self.X, self.y)

        return grid.best_params_, grid.best_score_, grid.cv_results_


    def initialize_ensemble_data(self, m1, m2, m3, scaling=False):

        if scaling:
            model_1 = self.add_pipeline_steps(m1, True)
            model_2 = self.add_pipeline_steps(m2, True)
            model_3 = self.add_pipeline_steps(m3, True)
        else:
            model_1 = self.add_pipeline_steps(m1)
            model_2 = self.add_pipeline_steps(m2)
            model_3 = self.add_pipeline_steps(m3)

        ## Fitting each of these models
        model_1.fit(self.X, self.y)
        model_2.fit(self.X, self.y)
        model_3.fit(self.X, self.y)

        ## Getting prediction probabilities from each of these models
        m1_pred_probs_trn = model_1.predict_proba(self.X)
        m2_pred_probs_trn = model_2.predict_proba(self.X)
        m3_pred_probs_trn = model_3.predict_proba(self.X)


        return model_1, model_2, model_3, m1_pred_probs_trn, m2_pred_probs_trn, m3_pred_probs_trn


    def get_val_probs(self, ens_mod_1, ens_mod_2, best_model):

        ## Getting prediction probabilities from each of these models
        m1_pred_probs_val = ens_mod_1.predict_proba(self.X_val)
        m2_pred_probs_val = ens_mod_2.predict_proba(self.X_val)
        best_model_pred_probs_val = best_model.predict_proba(self.X_val)

        return m1_pred_probs_val, m2_pred_probs_val, best_model_pred_probs_val


    def model_averaging_correlations(self, probs_1, probs_2, probs_3, scaling=False):

        ## Checking correlations between the predictions of the 3 models
        df_t = pd.DataFrame({'m1_pred': probs_1[:,1], 'm2_pred': probs_2[:,1], 'm3_pred': probs_3[:,1]})

        return df_t.corr()


    def weighted_ensemble(self, ens_m1_pred_probs, ens_m2_pred_probs, best_model_pred_probs, scaling=False):

        best_model_results = dict()
        weighted_model_results = dict()

        threshold = 0.5

        ## Best model predictions
        best_model_preds = np.where(best_model_pred_probs[:,1] >= threshold, 1, 0)

        ## Model averaging predictions (Weighted average)
        ensemble_model_preds = np.where(((0.1*ens_m1_pred_probs[:,1]) + (0.9*ens_m2_pred_probs[:,1])) >= threshold, 1, 0)

        ## Model 3 (Best model, tuned by GridSearch) performance on validation set
        best_model_results["roc_auc_score"] = roc_auc_score(self.y_val, best_model_preds)
        best_model_results["recall_score"] = recall_score(self.y_val, best_model_preds)
        best_model_results["confusion_matrix"] = confusion_matrix(self.y_val, best_model_preds)
        best_model_results["classification_report"] = classification_report(self.y_val, best_model_preds)

        weighted_model_results["roc_auc_score"] = roc_auc_score(self.y_val, ensemble_model_preds)
        weighted_model_results["recall_score"] = recall_score(self.y_val, ensemble_model_preds)
        weighted_model_results["confusion_matrix"] = confusion_matrix(self.y_val, ensemble_model_preds)
        weighted_model_results["classification_report"] = classification_report(self.y_val, ensemble_model_preds)

        return best_model_results, weighted_model_results


    def stacking_ensemble(self, trn_probs_1, trn_probs_2, val_probs_1, val_probs_2):

        stacked_model_results = dict()
        stacked_model_weights = dict()

        ## Training - 
        lr = LogisticRegression(C = 1.0, class_weight =  {0:1, 1:2.0})

        # Concatenating the probability predictions of the 2 models on train set
        X_t = np.c_[trn_probs_1[:,1], trn_probs_2[:,1]] 

        # Fit stacker model on top of outputs of base model
        lr.fit(X_t, self.y)

        ## Prediction - 

        # Concatenating outputs from both the base models on the validation set
        X_t_val = np.c_[val_probs_1[:,1], val_probs_2[:,1]]

        # Predict using the stacker model
        stacking_ensemble_preds = lr.predict(X_t_val)

        ## Ensemble model prediction on validation set
        stacked_model_results['roc_auc_score'] = roc_auc_score(self.y_val, stacking_ensemble_preds)
        stacked_model_results['recall_score'] = recall_score(self.y_val, stacking_ensemble_preds)
        stacked_model_results['confusion_matrix'] = confusion_matrix(self.y_val, stacking_ensemble_preds)
        stacked_model_results['classification_report'] = classification_report(self.y_val, stacking_ensemble_preds)

        # Stacked model weights
        stacked_model_weights['coef'] = lr.coef_
        stacked_model_weights['intercept'] = lr.intercept_

        return stacked_model_weights, stacked_model_results


    def archive_best_model(self, raw_best_model, scaling=False):

        if scaling:
            best_model = self.add_pipeline_steps(raw_best_model, True)
        else:
            best_model = self.add_pipeline_steps(raw_best_model)

        ## Fitting final model on train dataset
        best_model.fit(self.X, self.y)

        # Predict target probabilities
        val_probs = best_model.predict_proba(self.X_val)[:,1]

        # Predict target values on val data
        val_preds = np.where(val_probs > 0.45, 1, 0) # The probability threshold can be tweaked

        # Churn distribution visualization
        output = "Churn_Model/code/Modular/Output/Final_Model/churn_dist_boxplot.png"     
        sns.boxplot(self.y_val.ravel(), val_probs)
        plt.savefig(output)
        plt.clf()

        ## Validation metrics
        ras = roc_auc_score(self.y_val, val_preds)
        rs = recall_score(self.y_val, val_preds)
        cm = confusion_matrix(self.y_val, val_preds)
        cr = classification_report(self.y_val, val_preds)

        ## Save model object
        joblib.dump(best_model, 'Churn_Model/code/Modular/Output/Final_Model/final_churn_model_f1_0_45.sav')


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



        return ras, rs, cm, cr


    def get_recall_results(self):
        return self.recall_results

    def get_f1_results(self):
        return self.f1_results