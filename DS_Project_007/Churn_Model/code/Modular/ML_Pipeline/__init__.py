from pathlib import Path
import os
import sys
import pandas as pd

module_path = Path(__file__).parents[1]
sys.path.append(str(module_path))


class DataPipeline:

    from ML_Pipeline.EDA import (get_descriptive_stats,
                                 univariate_visualizations,
                                 bivariate_visualizations)
    
    from ML_Pipeline.Preprocessing import (split_dataset,
                                           encode_dataset,
                                           scale_data)
    
    from ML_Pipeline.FeatureEngineering import (create_features,
                                                select_features)

    from ML_Pipeline.Baseline import (initialize_baseline_models,
                                      create_weight_dict,
                                      evaluate_baseline_models)

    from ML_Pipeline.DataPipeline import (make_pipeline,
                                          add_pipeline_steps,
                                          save_final_model)
    
    from ML_Pipeline.ModelEvaluation import (get_best_model_predictions,
                                             get_best_model_correlations,
                                             initialize_final_model,
                                             error_checking_metrics,
                                             model_performance_metrics)
    
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
        self.scaled_cont_X_train=None 
        self.scaled_cont_X_test=None
        self.scaled_cont_X_val=None
        self.logreg_selected_feats=None
        self.dectree_selected_feats=None
        self.train_weight_dict=None  # target var weights based on the training set
        self.X_train=None  # scaled and encoded data
        self.X_test=None
        self.X_val=None
        self.pca_X_train_logreg_selected=None
        self.pca_X_train_variance_ratio=None
        self.pca_X_train_dectree_selected=None
        self.pca_X_train_dectree_selected_variance_ratio=None
        self.dectree_classifier=None
        self.baseline_results_dict=None
        self.spotcheck_models_dict=dict()
        self.spotcheck_results_dict=dict()
        self.tuning_model_pipe=None
        self.tuning_result_dict=dict()
        self.ensemble_experiment_dict=dict()
        self.ensemble_experiment_results=dict()
        self.final_model=None
        self.final_validation_metrics=dict()

    def perform_EDA(self):

        self.get_descriptive_stats()
        self.univariate_visualizations()
        self.split_dataset()  # preprocessing step
        self.encode_dataset()  # preprocessing step
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
