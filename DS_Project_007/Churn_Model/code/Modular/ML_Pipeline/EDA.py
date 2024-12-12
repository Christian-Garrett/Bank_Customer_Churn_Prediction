from pathlib import Path
import os
import sys
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from functools import reduce
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

module_path = Path(__file__).parents[1]
sys.path.append(str(module_path))


def get_descriptive_stats(self):

    print(self.data.info())
    print('\n', self.data.shape)
    print('\n', self.data.head(10).T)
    print('\n', self.data.describe()) # Describe all numerical columns
    print('\n', self.data.describe(include = ['O'])) # Describe all non-numerical/categorical columns
    print('\nThere are {} records and {} unique customers' .format(self.data.shape[0], 
                                                                   self.data.CustomerId.nunique())) ## Checking number of unique customers in the dataset

    surname_stats = \
        self.data.groupby(['Surname']).agg({'RowNumber':'count', 
                                            'Exited':'mean'}).reset_index().sort_values(by='RowNumber', 
                                                                                        ascending=False)
    print('\n', surname_stats.head())
    print('\n', self.data.Geography.value_counts(normalize=True))


def univariate_visualizations(self):
        
    sns.set_theme(style="whitegrid")

    ## CreditScore
    sns.boxplot(y=self.data.CreditScore)
    plt.savefig(os.path.join(self.output_path, "Univariate_Analysis/credit_score_boxplot.png"))
    plt.clf()

    ## Age
    sns.boxplot(y=self.data.Age)
    plt.savefig(os.path.join(self.output_path, "Univariate_Analysis/age_boxplot.png"))
    plt.clf()

    ## Tenure
    sns.violinplot(y=self.data.Tenure)
    plt.savefig(os.path.join(self.output_path, "Univariate_Analysis/tenure_violinplot.png"))
    plt.clf()

    ## Balance
    sns.violinplot(y=self.data.Balance)
    plt.savefig(os.path.join(self.output_path, "Univariate_Analysis/balance_violinplot.png"))
    plt.clf()

    ## NumOfProducts
    sns.set_theme(style='ticks')
    sns.displot(self.data.NumOfProducts, kind='hist')
    plt.savefig(os.path.join(self.output_path, "Univariate_Analysis/numProducts_distplot.png"))
    plt.clf()

    ## EstimatedSalary
    sns.kdeplot(self.data.EstimatedSalary)
    plt.savefig(os.path.join(self.output_path, "Univariate_Analysis/estimatedSalary_kdeplot.png"))
    plt.clf()


def split_dataset(self):

    ## Separating out target variable and removing the non-essential columns
    target_data = self.data[self.target_var].values
    self.data.drop(self.cols_to_remove, axis=1, inplace=True)

    ## Keeping aside a test/holdout set
    df_train_val, self.test_data, y_train_val, self.y_test = \
        train_test_split(self.data, target_data.ravel(), test_size = 0.1, random_state = 42)
    
    ## Splitting into train and validation set
    self.train_data, self.val_data, self.y_train, self.y_val = \
        train_test_split(df_train_val, y_train_val, test_size = 0.12, random_state = 42)     
            
    print('\nThe shape of the training data is: {}' .format(self.train_data.shape))
    print('The mean of the training data is: {}' .format(np.mean(self.y_train)))
    print('The shape of the validation data is: {}' .format(self.val_data.shape))
    print('The mean of the validation data is: {}' .format(np.mean(self.y_val)))
    print('The shape of the test data is: {}' .format(self.test_data.shape))      
    print('The mean of the test data is: {}\n' .format(np.mean(self.y_test)))


def encode_dataset(self):  # todo: loop

    genderLabelEncoder = LabelEncoder()

    ## Label encoding Gender var w/ mapping to accomodate for categorical variations in test/var data
    self.train_data['Gender'] = genderLabelEncoder.fit_transform(self.train_data.Gender)
    genderEncoderNameMap = \
        dict(zip(genderLabelEncoder.classes_,
                 genderLabelEncoder.transform(genderLabelEncoder.classes_)))   
    print('\nGender Encoding Map (training set): {}' .format(genderEncoderNameMap))

    ## Encoding Gender feature for validation and test set - new values tranform to null
    self.val_data['Gender'] = self.val_data.Gender.map(genderEncoderNameMap)
    self.val_data['Gender'].fillna(-1, inplace=True)

    ## Filling missing/NaN values created due to new categorical levels - new values tranform to null
    self.test_data['Gender'] = self.test_data.Gender.map(genderEncoderNameMap)
    self.test_data['Gender'].fillna(-1, inplace=True)

    print('\nGender train set encoding: {}' .format(self.train_data.Gender.unique()))
    print('Gender val set encoding: {}' .format(self.val_data.Gender.unique()))
    print('Gender test set encoding: {}' .format(self.test_data.Gender.unique()))


    ## One Hot Encoding for cat vars w/ multiple levels - reshape b/c not binary
    geographyLabelEncoder = LabelEncoder()
    geography_traindata_le = \
        geographyLabelEncoder.fit_transform(self.train_data.Geography).reshape(self.train_data.shape[0], 1)

    ## One Hot Encoding Training data
    geographyOneHotEncoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
    geography_traindata_ohe = geographyOneHotEncoder.fit_transform(geography_traindata_le)

    ## Create a Mapping to account for any variablity in the Validation and Test Data
    geographyEncoderNameMap = dict(zip(geographyLabelEncoder.classes_,
                                   geographyLabelEncoder.transform(geographyLabelEncoder.classes_)))
    print('\nGeography Encoding Map (training set) ' .format(geographyEncoderNameMap))

    ## Encoding Geography feature for validation and test set
    geography_valdata_le = self.val_data.Geography.map(geographyEncoderNameMap).ravel().reshape(-1, 1)
    geography_testdata_le = self.test_data.Geography.map(geographyEncoderNameMap).ravel().reshape(-1, 1)

    ## Filling missing/NaN values created due to new categorical levels
    geography_valdata_le[np.isnan(geography_valdata_le)] = 9999
    geography_testdata_le[np.isnan(geography_testdata_le)] = 9999

    ## One Hot Encoding Validation and Test data
    geography_valdata_ohe = geographyOneHotEncoder.transform(geography_valdata_le)
    geography_testdata_ohe = geographyOneHotEncoder.transform(geography_testdata_le)

    ## Add One Hot Encoded columns to the repective dataframes and remove original feature
    cols = ['country_' + str(x) for x in geographyEncoderNameMap.keys()]

    self.train_data = \
        pd.concat([self.train_data.reset_index(),
                   pd.DataFrame(geography_traindata_ohe, columns = cols)],
                   axis = 1).drop(['index'], axis=1)
    self.val_data = \
        pd.concat([self.val_data.reset_index(),
                   pd.DataFrame(geography_valdata_ohe, columns = cols)],
                   axis = 1).drop(['index'], axis=1)
    self.test_data = \
        pd.concat([self.test_data.reset_index(),
                   pd.DataFrame(geography_testdata_ohe, columns = cols)],
                   axis = 1).drop(['index'], axis=1)

    ## Drop the Geography column
    self.train_data.drop(['Geography'], axis = 1, inplace=True)
    self.val_data.drop(['Geography'], axis = 1, inplace=True)
    self.test_data.drop(['Geography'], axis = 1, inplace=True)


    ## Target Encoding the Surname Feature w/o data leakage
    means = self.train_data.groupby(['Surname']).Exited.mean()
    print('\nSurname Churn Averages: {}\n' .format(means.head(10)))

    data_target_mean = self.y_train.mean()

    ## Creating new encoded features for surname - Target (mean) encoding
    self.train_data['Surname_mean_churn'] = self.train_data.Surname.map(means)
    self.train_data['Surname_mean_churn'].fillna(data_target_mean, inplace=True)

    ## Calculate frequency of each category
    freqs = self.train_data.groupby(['Surname']).size()
    ## Create frequency encoding - Number of instances of each category in the data
    self.train_data['Surname_freq'] = self.train_data.Surname.map(freqs)
    self.train_data['Surname_freq'].fillna(0, inplace=True)
    ## Create Leave-one-out target encoding for Surname
    self.train_data['Surname_enc'] = \
        ((self.train_data.Surname_freq * self.train_data.Surname_mean_churn) - self.train_data.Exited)/(self.train_data.Surname_freq - 1)
    ## Fill NaNs occuring due to category frequency being 1 or less
    self.train_data['Surname_enc'].fillna((((self.train_data.shape[0] * data_target_mean) - self.train_data.Exited) / (self.train_data.shape[0] - 1)), inplace=True)

    ## Apply the normal Target encoding mapping as obtained from the training set
    self.val_data['Surname_enc'] = self.val_data.Surname.map(means)
    self.val_data['Surname_enc'].fillna(data_target_mean, inplace=True)

    self.test_data['Surname_enc'] = self.test_data.Surname.map(means)
    self.test_data['Surname_enc'].fillna(data_target_mean, inplace=True)

    ## Show that using LOO Target encoding decorrelates features
    print('\nTarget Encoding Correlations:\n {}' .format(self.train_data[['Surname_mean_churn',
                                                                          'Surname_enc',
                                                                          'Exited']].corr()))

    ### Deleting calculation columns
    self.train_data.drop(['Surname_mean_churn'], axis=1, inplace=True)
    self.train_data.drop(['Surname_freq'], axis=1, inplace=True)

    print('\nEncoded Training Data: \n{}' .format(self.train_data.head()))
    print('\nEncoded Val Data: \n{}' .format(self.val_data.head()))
    print('\nEncoded Test Data: \n{}\n' .format(self.test_data.head()))    


def bivariate_visualizations(self):
    
    ## Check linear correlation (rho) between individual features and the target variable
    corr = self.data.corr()
    print(corr)

    ## Heatmap
    sns.heatmap(corr, cmap='coolwarm')
    plt.savefig(os.path.join(self.output_path, "Bivariate_Analysis/train_corr_heatmap.png"))
    plt.clf()   

    ## Churn vs Age Boxplot  
    sns.boxplot(x="Exited", y="Age", data=self.data, palette="Set3")
    plt.savefig(os.path.join(self.output_path, "Bivariate_Analysis/age_churn_boxplot.png"))
    plt.clf()

    ## Churn vs Balance Violinplot
    sns.violinplot(x="Exited", y="Balance", data=self.data, palette="Set3")
    plt.savefig(os.path.join(self.output_path,
                             "Bivariate_Analysis/balance_churn_violinplot.png"))
    plt.clf()

    # Check association of categorical features with target variable
    for col in ['Gender', 'IsActiveMember', 'country_Germany', 'country_France', 'NumOfProducts']:
        print(f"Value Counts (training): {self.train_data.groupby([col]).Exited.mean()}")


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


def scale_data(self):

    ## Feature scaling and normalization the continuous variables (Logistic Rregression and SVM)
    self.cont_transformed_vars = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts',
                      'EstimatedSalary', 'Surname_enc', 'bal_per_product',
                      'bal_by_est_salary', 'tenure_age_ratio', 'age_surname_mean_churn']
               
    self.cat_transformed_vars = ['Gender', 'HasCrCard', 'IsActiveMember', 'country_France',
                     'country_Germany', 'country_Spain']

    ## Scaling only continuous columns for the training set
    scalar_object = StandardScaler()

    update_info = list(zip((self.train_data, self.test_data, self.val_data),
                              ('X_train', 'X_test', 'X_val')))

    for unscaled_data, update_var_name in update_info:
        if(update_var_name == 'X_train'):
            scaled_cont_attribute = \
                scalar_object.fit_transform(unscaled_data[self.cont_transformed_vars])
        else:
            scaled_cont_attribute = \
                scalar_object.transform(unscaled_data[self.cont_transformed_vars])            
        scaled_cont_attribute = scaled_cont_attribute.tolist()
        unscaled_cat_attribute = unscaled_data[self.cat_transformed_vars]
        unscaled_cat_X_train = unscaled_cat_attribute.values.tolist()
        full_values = list(zip(scaled_cont_attribute, unscaled_cat_X_train))
        scaled_cont_full_attribute = [a+b for a,b in full_values]
        updated_df = pd.DataFrame(data=scaled_cont_full_attribute,
                                   columns=self.cont_transformed_vars+self.cat_transformed_vars)
        self.__setattr__(update_var_name, updated_df)

    ## Mapping learned info on the continuous features for the val and test sets
    sc_map = {'mean':scalar_object.mean_, 'std':np.sqrt(scalar_object.var_)}
    print(f"Scaler Transformation Attribute Mappings: {sc_map}")


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
