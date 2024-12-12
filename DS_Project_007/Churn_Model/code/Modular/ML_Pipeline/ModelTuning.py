from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from lightgbm import LGBMClassifier


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
