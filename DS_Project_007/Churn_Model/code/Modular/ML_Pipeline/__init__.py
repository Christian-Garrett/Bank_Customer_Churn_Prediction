from pathlib import Path
import os
import sys
import pandas as pd

module_path = Path(__file__).parents[1]
sys.path.append(str(module_path))


class DataPipeline:
    """
    A class used to create the churn model predictions.
    ...

    Attributes
    ----------
    output_path : str
        Output data text path
    data_path : str
        Input data text path
    data : df
        Input data
    target_var : list
        The variable that is being predicted
    cols_to_remove : list
        Columns that will not be used for modeling
    numeric_feats : list
        Continuous data
    categorical_feats : list
        Non-continuous data
    cols_to_scale : list
        Disproportionate continuous data
    train_data : df
        Training dataset with encoded categorical features
    y_train : list
        Target variable for the training data
    test_data : df
        Test dataset with encoded categorical features
    y_test : list
        Target variable for the testing data
    val_data : df
        Validateion dataset with encoded categorical features
    y_val : list
        Target variable for the validation data
    LR_Baseline_Model : _
        Logistic Regression Baseline Model
    SVC_Baseline_Model : _
        Support Vector Machine Baseline Model
    DT_Baseline_Model : _
        Decision Trees Baseline Model
    cont_transformed_vars : list
        List of continuous variables after feature engineering step
    cat_transformed_vars : list
        List of categorical variables after feature engineering step                      
    logreg_selected_feats : list
        Selected features from logistic regression RFE (recursive feature elimination)
    dectree_selected_feats : list
        Selected features from decision tree RFE (recursive feature elimination)
    train_weight_dict : dict
        Addresses disproportionate outcome levels within the target variable
    X_train : df
        Training dataset with encoded and scaled data
    X_test : df
        Test dataset with encoded and scaled data
    X_val : df
        Validation dataset with encoded and scaled data
    pca_X_train_logreg_selected : list
        PCA reduced data with logistic regression selected features
    pca_X_train_logtree_selected_variance_ratio : list
        The percentage of variance captured by the selected PCA reduced features
    pca_X_train_dectree_selected : list
        PCA reduced data with decision tree selected features
    pca_X_train_dectree_selected_variance_ratio : list
        The percentage of variance captured by the selected PCA reduced features
    baseline_results_dict : dict
        Dictionary that contains the output results from the baseline models
    spotcheck_models_dict : dict
        Dictionary containing the instantiated spot check models and variations
    spotcheck_results_dict : dict
        Dictionary that contains the output results from the spot check models
    tuning_result_dict : _
        Best paramaters and cross validation scores after hyperparameter tuning
    ensemble_experiment_dict : dict
        Dictionary containing ensembles based from variations of target variable weighting
    (ensemble_experiment_results) : dict
        Dictionary containing output results from the stacked/weighted ensemble models
    final_model : _
        Best model after baseline, hyperparameter tuning, and ensemble experiments      
    final_validation_metrics : dict
        Dictionary containing final model evaluation statistics                        

    Methods
    -------
    perform_EDA()
        Descriptive statistics and visualizations
    perform_feature_engineering()
        Data encoding, new feature creation with visualizations
    review_baseline_models()
        Check output of a few baseline models
    perform_model_spotchecks()
        Deeper analysis of more potential models
    build_final_model()
        Hyperparameter tuning, ensemble experiments, error checking,
        performance metrics and saving to an output file.    

    """

    from ML_Pipeline.Preprocessing import (split_dataset,
                                           encode_dataset,
                                           scale_data)
     
    from ML_Pipeline.EDA import (get_descriptive_stats,
                                 univariate_visualizations,
                                 bivariate_visualizations)
    
    from ML_Pipeline.FeatureEngineering import (create_features,
                                                select_features)

    from ML_Pipeline.Baseline import (initialize_baseline_models,
                                      create_weight_dict,
                                      evaluate_baseline_models)

    from ML_Pipeline.DataPipeline import (make_pipeline,
                                          add_pipeline_steps,
                                          save_final_model)

    from ML_Pipeline.ModelTuning import (hyperparameter_tuning,
                                         initialize_tuning_model,
                                         rand_tuning,
                                         grid_tuning)
    
    from ML_Pipeline.SpotCheck import (initialize_spotcheck_models,
                                       evaluate_spotcheck_models)
    
    from ML_Pipeline.Ensembles import (ensemble_model_experiments,
                                       initialize_ensemble_model_pipelines,
                                       get_weighted_ensemble_results,
                                       get_stacked_ensemble_results)
    
    from ML_Pipeline.ModelEvaluation import (get_best_model_predictions,
                                             get_best_model_correlations,
                                             initialize_final_model,
                                             error_checking_metrics,
                                             model_performance_metrics)

    def __init__(self):

        self.output_path=os.path.join(module_path, "Output/")
        self.data_path=os.path.join(module_path, "Input/Churn_Modelling.csv")
        self.data=pd.read_csv(self.data_path)
        self.target_var=['Exited']
        self.cols_to_remove=['RowNumber', 'CustomerId']
        self.numeric_feats=['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts',
                            'EstimatedSalary']
        self.categorical_feats=['Surname', 'Geography', 'Gender', 'HasCrCard',
                                'IsActiveMember']
        self.cols_to_scale = ['CreditScore', 'Age', 'Balance', 'EstimatedSalary',
                              'bal_per_product', 'bal_by_est_salary','tenure_age_ratio',
                              'age_surname_enc']
        self.train_data=None # encoded unscaled data
        self.y_train=None
        self.test_data=None
        self.y_test=None
        self.val_data=None
        self.y_val=None
        self.LR_Baseline_Model=None
        self.SVC_Baseline_Model=None
        self.DT_Baseline_Model=None
        self.cont_transformed_vars=None
        self.cat_transformed_vars=None
        self.logreg_selected_feats=None
        self.dectree_selected_feats=None
        self.train_weight_dict=None 
        self.X_train=None  # scaled and encoded data
        self.X_test=None
        self.X_val=None
        self.pca_X_train_logreg_selected=None
        self.pca_X_train_logtree_selected_variance_ratio=None
        self.pca_X_train_dectree_selected=None
        self.pca_X_train_dectree_selected_variance_ratio=None
        self.baseline_results_dict=None
        self.spotcheck_models_dict=dict()
        self.spotcheck_results_dict=dict()
        self.tuning_result_dict=dict()
        self.ensemble_experiment_dict=dict()
        self.ensemble_experiment_results=dict()
        self.final_model=None
        self.final_validation_metrics=dict()

    def perform_EDA(self):

        self.get_descriptive_stats()
        self.univariate_visualizations()
        self.split_dataset()
        self.encode_dataset()
        self.bivariate_visualizations()

    def perform_feature_engineering(self):

        self.create_weight_dict()
        self.create_features()
        self.scale_data()
        self.select_features()

    def review_baseline_models(self):
        
        self.initialize_baseline_models()
        self.evaluate_baseline_models()

    def perform_model_spotchecks(self):

        self.initialize_spotcheck_models()
        self.evaluate_spotcheck_models('recall')
        self.evaluate_spotcheck_models('f1')

    def build_final_model(self):

        self.hyperparameter_tuning()
        self.ensemble_model_experiments()
        self.initialize_final_model()
        self.error_checking_metrics()
        self.model_performance_metrics()
        self.save_final_model()
