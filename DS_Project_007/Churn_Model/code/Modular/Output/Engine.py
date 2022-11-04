import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import joblib

from sklearn.metrics import roc_auc_score, f1_score, recall_score, confusion_matrix, classification_report

from Churn_Model.code.Modular.ML_Pipeline.EDA import EDA
from Churn_Model.code.Modular.ML_Pipeline.Baseline import Baseline
from Churn_Model.code.Modular.ML_Pipeline.DataPipeline import DataPipeline

from lightgbm import LGBMClassifier



file_path = "Churn_Model\code\Modular\Input\Churn_Modelling.csv"

def create_model():

    # EDA and Preprocessing
    processed_data = EDA(file_path, full=True)

    # Initialize Baseline Models
    LogReg_Params = [1.0, 'l2', -1]
    SVM_Params = [1.0, 'linear']
    DecTree_Params = ['entropy', 4, None, 25, 15]
    base = Baseline(processed_data, LogReg_Params, SVM_Params, DecTree_Params)

    # Linear Model - Logistic Regression (Baseline)
    logreg_tr_roc_auc_score, logreg_tr_recall_score, logreg_tr_conf_mat, logreg_tr_class_rep = base.get_lr_metrics()
    logreg_val_roc_auc_score, logreg_val_recall_score, logreg_val_conf_mat, logreg_val_class_rep = base.get_lr_metrics(False)

    # Linear Model - Support Vector Machine (Baseline)
    svm_tr_roc_auc_score, svm_tr_recall_score, svm_tr_conf_mat, svm_tr_class_rep = base.get_svm_metrics()
    svm_val_roc_auc_score, svm_val_recall_score, svm_val_conf_mat, svm_val_class_rep = base.get_svm_metrics(False)

    # Get decision boundaries of linear models and save results --> xx & yy = mesh corners | z1 = LR, z2 = SVM
    xx, yy, z1, z2 = base.get_pca_linear_boundaries()

    output = "Churn_Model/code/Modular/Output/Decision_Boundaries/linear_decision_boundaries.png"
    plt.contourf(xx, yy, z1, alpha=0.4) # LR
    plt.contour(xx, yy, z2, alpha=0.4, colors = 'blue') # SVM
    sns.scatterplot(base.get_pca_X_train()[:,0], base.get_pca_X_train()[:,1], hue = base.get_y_train(), s = 50, alpha = 0.8)
    plt.title('Linear models - LogReg and SVM')
    plt.savefig(output)
    plt.clf()


    # Non Linear Model - Decision Tree (Baseline)
    dt_tr_roc_auc_score, dt_tr_recall_score, dt_tr_conf_mat, dt_tr_class_rep = base.get_dt_metrics()
    dt_val_roc_auc_score, dt_val_recall_score, dt_val_conf_mat, dt_val_class_rep = base.get_dt_metrics(False)

    # Get decision boundaries for the non linear model and save the results
    xx, yy, z = base.get_pca_nonlinear_boundary()

    output = "Churn_Model/code/Modular/Output/Decision_Boundaries/nonlinear_decision_boundary.png"
    plt.contourf(xx, yy, z, alpha=0.4) # DT
    sns.scatterplot(base.get_pca_X_train_dt()[:,0], base.get_pca_X_train_dt()[:,1], hue = base.get_y_train(), s = 50, alpha = 0.8)
    plt.title('Decision Tree')
    plt.savefig(output)
    plt.clf()

    # Decision Tree Visualization
    base.create_dt_visualization()



    ## Automating model preparation and implemetation through data pipelines (Production Use Case)
    spot_checks = DataPipeline(processed_data, base)
    
    print('\nSpot Check Recall metrics: \n{}' .format(spot_checks.get_recall_results()))
    print('\nSpot Check F1-score metrics: \n{}' .format(spot_checks.get_f1_results()))



    ## Best Model Hyperparameter Tuning - Randomized and Grid Search
    rand_iters = 20
    rand_scoring = 'f1'
    rand_params = {'classifier__n_estimators':[10, 21, 51, 100, 201, 350, 501]
                   ,'classifier__max_depth': [3, 4, 6, 9]
                   ,'classifier__num_leaves':[7, 15, 31] 
                   ,'classifier__learning_rate': [0.03, 0.05, 0.1, 0.5, 1]
                   ,'classifier__colsample_bytree': [0.3, 0.6, 0.8]
                   ,'classifier__reg_alpha': [0, 0.3, 1, 5]
                   ,'classifier__reg_lambda': [0.1, 0.5, 1, 5, 10]
                   ,'classifier__class_weight': [{0:1,1:1.0}, {0:1,1:1.96}, {0:1,1:3.0}, {0:1,1:3.93}]
                 }


    grid_scoring = 'f1'
    grid_params = {'classifier__n_estimators':[201]
                 ,'classifier__max_depth': [6]
                 ,'classifier__num_leaves': [63]
                 ,'classifier__learning_rate': [0.1]
                 ,'classifier__colsample_bytree': [0.6, 0.8]
                 ,'classifier__reg_alpha': [0, 1, 10]
                 ,'classifier__reg_lambda': [0.1, 1, 5]
                 ,'classifier__class_weight': [{0:1,1:3.0}]
                 }

    lgb = LGBMClassifier(boosting_type = 'dart', min_child_samples = 20, n_jobs = - 1, importance_type = 'gain', num_leaves = 31)

    best_rand_params, rand_score, rand_results = spot_checks.rand_tuning(lgb, rand_params, rand_iters, rand_scoring)
    print('Hyperparameter Tuning - Best Params Randomized Search: \n{}\n' .format(best_rand_params))
    
    best_grid_params, grid_score, grid_results = spot_checks.grid_tuning(lgb, grid_params, grid_scoring)
    print('Hyperparameter Tuning - Best Params Grid Search: \n{}\n' .format(best_grid_params))



    ## Ensembles - Evaluate Best Model Averaging and Stacking

    ## Three versions of the most effective model with best params for F1-score metric

    # Equal weights to both target classes (no class imbalance correction)
    lgb1 = LGBMClassifier(boosting_type = 'dart', class_weight = {0: 1, 1: 1}, min_child_samples = 20, n_jobs = - 1
                         , importance_type = 'gain', max_depth = 4, num_leaves = 31, colsample_bytree = 0.6, learning_rate = 0.1
                         , n_estimators = 21, reg_alpha = 0, reg_lambda = 0.5)

    # Addressing class imbalance completely by weighting the undersampled class by the class imbalance ratio
    lgb2 = LGBMClassifier(boosting_type = 'dart', class_weight = {0: 1, 1: 3.93}, min_child_samples = 20, n_jobs = - 1
                         , importance_type = 'gain', max_depth = 6, num_leaves = 63, colsample_bytree = 0.6, learning_rate = 0.1
                         , n_estimators = 201, reg_alpha = 1, reg_lambda = 1)


    # Best class_weight parameter settings (partial class imbalance correction)
    lgb3 = LGBMClassifier(boosting_type = 'dart', class_weight = {0: 1, 1: 3.0}, min_child_samples = 20, n_jobs = - 1
                         , importance_type = 'gain', max_depth = 6, num_leaves = 63, colsample_bytree = 0.6, learning_rate = 0.1
                         , n_estimators = 201, reg_alpha = 1, reg_lambda = 1)

    # create ensemble data models and get training prediction probabilities
    pipe_1, pipe_2, pipe_3, mod_1_trn_probs, mod_2_trn_probs, mod_3_trn_probs = spot_checks.initialize_ensemble_data(lgb1, lgb2, lgb3)
       
    # check correlations between best model variations from training values
    trn_pred_corr_df = spot_checks.model_averaging_correlations(mod_1_trn_probs, mod_2_trn_probs, mod_3_trn_probs)
    print('Best model variations prediction correlation matrix: \n{}\n' .format(trn_pred_corr_df))

    # models 1 and 2 are the least correlated so they will be used for the ensembles (averaging and stacking)
    best_model = pipe_3
    ensemble_model_1 = pipe_1
    ensemble_model_2 = pipe_2

    # get the validation prediction probabilities for the ensemble models
    ens_val_probs_1, ens_val_probs_2, best_model_val_probs = spot_checks.get_val_probs(ensemble_model_1, ensemble_model_2, best_model)

    # get prediction probabilities from best model variation and weighted combination of remaining variations using validation values
    best_model_results, weighted_model_results = spot_checks.weighted_ensemble(ens_val_probs_1, ens_val_probs_2, best_model_val_probs)

    # compare model averaging ensemble output against the best model variation
    print('Best Model Classification Report: \n{}\n' .format(best_model_results["classification_report"]))
    print('Weighted Model Classification Report: \n{}\n' .format(weighted_model_results["classification_report"]))

    # set the ensemble stack model probabilities 
    ens_trn_probs_1 = mod_1_trn_probs
    ens_trn_probs_2 = mod_2_trn_probs

    stack_model_weights, stack_model_results = spot_checks.stacking_ensemble(ens_trn_probs_1, ens_trn_probs_2, ens_val_probs_1, ens_val_probs_2)
    print('Stacked Model Classification Report: \n{}\n' .format(stack_model_results["classification_report"]))
    

    ## Best model variation error checking -

    ## Final model with best params for F1-score metric
    best_f1_lgb = LGBMClassifier(boosting_type = 'dart', class_weight = {0: 1, 1: 3.0}
                                 , min_child_samples = 20, n_jobs = - 1, importance_type = 'gain'
                                 , max_depth = 6, num_leaves = 63, colsample_bytree = 0.6
                                 , learning_rate = 0.1, n_estimators = 201, reg_alpha = 1
                                 , reg_lambda = 1)

    best_recall_lgb = LGBMClassifier(boosting_type='dart', num_leaves=31, max_depth= 6
                                 , learning_rate=0.1, n_estimators = 21
                                 , class_weight= {0: 1, 1: 3.93}, min_child_samples=2
                                 , colsample_bytree=0.6, reg_alpha=0.3
                                 , reg_lambda=1.0, n_jobs=- 1, importance_type = 'gain')
    
    # print error analysis metrics
    processed_data.error_analysis(best_f1_lgb)


    # save the best model to an output file
    final_roc_auc_score, final_recall_score, final_conf_mat, final_class_rep = spot_checks.archive_best_model(best_f1_lgb)



def test_model():

    ## Load model object
    file = 'Churn_Model/code/Modular/Output/Final_Model/final_churn_model_f1_0_45.sav'
    model = joblib.load(file)

    ## Load the data
    data = EDA(file_path)
    X_test = data.get_df_test().drop(columns = ['Exited'], axis = 1)
    y_test = data.get_y_test()
    
    print('X_test shape: \n{}\n' .format(X_test.shape))
    print('y_test shape: \n{}\n' .format(y_test.shape))

    ## Predict target probabilities
    test_probs = model.predict_proba(X_test)[:,1]

    ## Predict target values on test data
    test_preds = np.where(test_probs > 0.45, 1, 0) # Flexibility to tweak the probability threshold

    output = "Churn_Model/code/Modular/Output/test_churn_distribution_boxplot.png"     
    sns.boxplot(y_test.ravel(), test_probs)
    plt.savefig(output)
    plt.clf()

    ## Test set metrics
    ras = roc_auc_score(y_test, test_preds)
    rs = recall_score(y_test, test_preds)
    cm = confusion_matrix(y_test, test_preds)
    cr = classification_report(y_test, test_preds)

    print('roc_auc_score: \n{}\n' .format(ras))
    print('recall score: \n{}\n' .format(rs))
    print('confusion matrix: \n{}\n' .format(cm))
    print('classification report: \n{}\n' .format(cr))

    ## Adding predictions and their probabilities in the original test dataframe
    test = data.get_df_test().copy()
    test['predictions'] = test_preds
    test['pred_probabilities'] = test_probs


    ## Creating a list of customers who will most likely churn - ptobability greater than 70%
    high_churn_list = test[test.pred_probabilities > 0.7].sort_values(by = ['pred_probabilities'], ascending = False).reset_index().drop(columns = ['index', 'Exited', 'predictions'], axis = 1)

    print('High Churn List Shape: \n{}\n' .format(high_churn_list.shape))
    print('High Churn List Head: \n{}\n' .format(high_churn_list.head()))

    ## Save list to output file
    output = 'Churn_Model/code/Modular/Output/high_churn_list.csv'
    high_churn_list.to_csv(output, index = False)



#create_model()
test_model()




