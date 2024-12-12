import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, ComplementNB, BernoulliNB
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


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
