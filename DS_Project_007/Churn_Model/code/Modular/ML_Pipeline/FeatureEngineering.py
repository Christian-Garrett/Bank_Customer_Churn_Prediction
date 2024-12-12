import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import os


def create_features(self):
    
    eps = 1e-6

    for dataset in [self.train_data, self.test_data, self.val_data]:
        dataset['bal_per_product'] = dataset.Balance/(dataset.NumOfProducts + eps)
        dataset['bal_by_est_salary'] = dataset.Balance/(dataset.EstimatedSalary + eps)
        dataset['tenure_age_ratio'] = dataset.Tenure/(dataset.Age + eps)
        dataset['age_surname_mean_churn'] = np.sqrt(dataset.Age) * dataset.Surname_enc

    ## Ensuring that the new column doesn't have any missing values
    new_cols = ['bal_per_product', 'bal_by_est_salary', 'tenure_age_ratio', 'age_surname_mean_churn']
    print('New Features Missing Values: \n{}\n' .format(self.train_data[new_cols].isnull().sum()))

    ## Linear association of new columns with target variables to judge importance
    sns.heatmap(self.train_data[new_cols + ['Exited']].corr(), annot=True)
    plt.savefig(os.path.join(self.output_path, "Bivariate_Analysis/new_feature_heatmap.png"))
    plt.clf()


def select_features(self):

    #params = {'alpha' : 1, 'kernel' : 'linear', 'gamma': 10}    

    ## Creating feature-set and target for RFE (recursive feature elimination) model
    y_train = self.train_data['Exited'].values

    # Use the encoded unscaled data for feature selection algorithm to help with convergence
    X_train = self.train_data[self.cat_transformed_vars + self.cont_transformed_vars]

    # Create RFE objects with different estimators
    num_features_to_select = 10
    logreg_est = LogisticRegression()
    rfe_logreg_object = RFE(estimator=logreg_est, n_features_to_select=num_features_to_select)

    dectree_est = DecisionTreeClassifier(max_depth=4, criterion='entropy')
    rfe_dectree_object = RFE(estimator=dectree_est, n_features_to_select=num_features_to_select)

    RFE_dict = {'log_reg': {'object': rfe_logreg_object,
                            'selected_feats': 'logreg_selected_feats'},
                'dec_tree': {'object': rfe_dectree_object,
                             'selected_feats': 'dectree_selected_feats'}}
    
    for estimator_name, estimator_dict in RFE_dict.items():
        fitted_estimator = estimator_dict['object'].fit(X_train.values, y_train)
        print(f"\n{estimator_name} Feature Bool Mask: {fitted_estimator.support_}")
        print(f"{estimator_name} Feature Rankings: {fitted_estimator.ranking_}")
        estimator_mask = fitted_estimator.support_.tolist()
        self.__setattr__(estimator_dict['selected_feats'],
                         [feature for keep_feature, feature in zip(estimator_mask, X_train.columns)
                          if keep_feature])
        print(f"{estimator_name} Selected Features: \
{self.__getattribute__(estimator_dict['selected_feats'])}")
