import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


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
