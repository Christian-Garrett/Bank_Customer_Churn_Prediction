import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, classification_report
from IPython.display import display
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

from Churn_Model.code.Modular.ML_Pipeline.CategoricalEncoder import CategoricalEncoder
from Churn_Model.code.Modular.ML_Pipeline.AddFeatures import AddFeatures
from Churn_Model.code.Modular.ML_Pipeline.CustomScaler import CustomScaler



class EDA:

    def __init__(self, file, full=False):

        self.df = pd.read_csv(file)
        self.y = self.df
        self.target_var = []
        self.cols_to_remove = []
        self.num_feat = []
        self.cat_feats = [] 
        self.df_train_val = [] 
        self.y_train_val = [] 
        self.df_test = []       
        self.y_test = [] 
        self.df_train = [] 
        self.y_train = [] 
        self.df_val = []        
        self.y_val = [] 
        self.cat_vars = []
        self.cont_vars = []
        self.selected_feats = []
        self.selected_feats_dt = []
        self.sc_X_train = self.df
        self.sc_X_val = self.df
        self.sc_X_test = self.df



        self.raw_data_info()
        self.split_data()

        if(full):
            self.univariate_visualizations()

        self.handle_missing_data()
        self.cat_var_encoding()

        if(full):
            self.bivariate_visualizations()

        self.feature_engineering()

        if(full):
            self.feature_selection()


    def raw_data_info(self):

        print(self.df.info())
        print('\n',self.df.shape)
        print('\n',self.df.head(10).T)
        print('\n',self.df.describe()) # Describe all numerical columns
        print('\n',self.df.describe(include = ['O'])) # Describe all non-numerical/categorical columns
        print('\nThere are {} records and {} unique customers' .format(self.df.shape[0], self.df.CustomerId.nunique())) ## Checking number of unique customers in the dataset

        df_t = self.df.groupby(['Surname']).agg({'RowNumber':'count', 'Exited':'mean'}
                                          ).reset_index().sort_values(by='RowNumber', ascending=False)
        print('\n',df_t.head())
        print('\n',self.df.Geography.value_counts(normalize=True))



    def split_data(self):

        ## Separating out different columns into various categories as defined above
        self.target_var = ['Exited']
        self.cols_to_remove = ['RowNumber', 'CustomerId']
        self.num_feats = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']
        self.cat_feats = ['Surname', 'Geography', 'Gender', 'HasCrCard', 'IsActiveMember']

        ## Separating out target variable and removing the non-essential columns
        self.y = self.df[self.target_var].values
        self.df.drop(self.cols_to_remove, axis=1, inplace=True)
        ## Keeping aside a test/holdout set
        self.df_train_val, self.df_test, self.y_train_val, self.y_test = train_test_split(self.df, self.y.ravel(), test_size = 0.1, random_state = 42)
        ## Splitting into train and validation set
        self.df_train, self.df_val, self.y_train, self.y_val = train_test_split(self.df_train_val, self.y_train_val, test_size = 0.12, random_state = 42)     
                

        print('\nThe shape of the training data is: {}' .format(self.df_train.shape))
        print('The mean of the training data is: {}' .format(np.mean(self.y_train)))
        print('The shape of the validation data is: {}' .format(self.df_val.shape))
        print('The mean of the validation data is: {}' .format(np.mean(self.y_val)))
        print('The shape of the test data is: {}' .format(self.df_test.shape))      
        print('The mean of the test data is: {}\n' .format(np.mean(self.y_test)))
      


    def univariate_visualizations(self):

        sns.set(style="whitegrid")

        ## CreditScore
        output = "Churn_Model/code/Modular/Output/Univariate_Analysis/credit_score_boxplot.png"     
        sns.boxplot(y = self.df_train['CreditScore'])
        plt.savefig(output)
        plt.clf()

        ## Age
        output = "Churn_Model/code/Modular/Output/Univariate_Analysis/age_boxplot.png"
        sns.boxplot(y = self.df_train['Age'])
        plt.savefig(output)
        plt.clf()

        ## Tenure
        output = "Churn_Model/code/Modular/Output/Univariate_Analysis/tenure_violinplot.png"
        sns.violinplot(y = self.df_train.Tenure)
        plt.savefig(output)
        plt.clf()

        ## Balance
        output = "Churn_Model/code/Modular/Output/Univariate_Analysis/balance_violinplot.png"
        sns.violinplot(y = self.df_train['Balance'])
        plt.savefig(output)
        plt.clf()

        ## NumOfProducts
        output = "Churn_Model/code/Modular/Output/Univariate_Analysis/numProducts_distplot.png"
        sns.set(style = 'ticks')
        sns.distplot(self.df_train.NumOfProducts, hist=True, kde=False)
        plt.savefig(output)
        plt.clf()

        ## EstimatedSalary
        output = "Churn_Model/code/Modular/Output/Univariate_Analysis/estimatedSalary_kdeplot.png"
        sns.kdeplot(self.df_train.EstimatedSalary)
        plt.savefig(output)
        plt.clf()


    def handle_missing_data(self):

        ## Get Missing Value Info
        print('\nMissing Data Info: {}\n' .format(self.df_train.isnull().sum()))

        ## Making all changes in a temporary dataframe
        #df_missing = self.df_train.copy()
        #df_missing.isnull().sum()/df_missing.shape[0]
        #df_missing['Age'] = df_missing.Age.apply(lambda x: int(np.random.normal(age_mean,3)) if np.isnan(x) else x)
        #output = "Churn_Model/code/Modular/Output/reconstructedData_age.png"
        #sns.distplot(df_missing.Age)
        #plt.savefig(output)

        # Filling nulls in Geography (categorical feature with a high percentage of missing values)
        #geog_fill_value = 'UNK'
        #df_missing.Geography.fillna(geog_fill_value, inplace=True)

        # Filling nulls in HasCrCard (boolean feature) - 0 for few nulls, -1 for lots of nulls
        #df_missing.HasCrCard.fillna(0, inplace=True)

        #df_missing.isnull().sum()/df_missing.shape[0]


    def cat_var_encoding(self):

        le = LabelEncoder()

        ## Label encoding Gender var w/ mapping to accomodate for any variablility in test or validation set.
        self.df_train['Gender'] = le.fit_transform(self.df_train['Gender'])
        le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))   
        print('\nGender Encoding Map (training set): {}' .format(le_name_mapping))

        ## Encoding Gender feature for validation and test set - new values tranform to null
        self.df_val['Gender'] = self.df_val.Gender.map(le_name_mapping)
        self.df_val['Gender'].fillna(-1, inplace=True)
       
        ## Filling missing/NaN values created due to new categorical levels - new values tranform to null
        self.df_test['Gender'] = self.df_test.Gender.map(le_name_mapping)
        self.df_test['Gender'].fillna(-1, inplace=True)

        print('\nGender train set encoding: {}' .format(self.df_train.Gender.unique()))
        print('Gender val set encoding: {}' .format(self.df_val.Gender.unique()))
        print('Gender test set encoding: {}' .format(self.df_test.Gender.unique()))


        ## One Hot Encoding for cat vars w/ multiple levels - reshape b/c not binary
        le_ohe = LabelEncoder()
        enc_train = le_ohe.fit_transform(self.df_train.Geography).reshape(self.df_train.shape[0],1)

        ## One Hot Encoding Training data
        ohe = OneHotEncoder(handle_unknown = 'ignore', sparse=False)
        ohe_train = ohe.fit_transform(enc_train)

        ## Create a Mapping to account for any variablity in the Validation and Test Data
        le_ohe_name_mapping = dict(zip(le_ohe.classes_, le_ohe.transform(le_ohe.classes_)))
        print('\nGeography Encoidng Map (training set) ' .format(le_ohe_name_mapping))

        ## Encoding Geography feature for validation and test set
        enc_val = self.df_val.Geography.map(le_ohe_name_mapping).ravel().reshape(-1,1)
        enc_test = self.df_test.Geography.map(le_ohe_name_mapping).ravel().reshape(-1,1)

        ## Filling missing/NaN values created due to new categorical levels
        enc_val[np.isnan(enc_val)] = 9999
        enc_test[np.isnan(enc_test)] = 9999

        ## One Hot Encoding Validation and Test data
        ohe_val = ohe.transform(enc_val)
        ohe_test = ohe.transform(enc_test)

        ## Add One Hot Encoded columns to the repective dataframes and remove original feature
        cols = ['country_' + str(x) for x in le_ohe_name_mapping.keys()]

        self.df_train = pd.concat([self.df_train.reset_index(), pd.DataFrame(ohe_train, columns = cols)], axis = 1).drop(['index'], axis=1)
        self.df_val = pd.concat([self.df_val.reset_index(), pd.DataFrame(ohe_val, columns = cols)], axis = 1).drop(['index'], axis=1)
        self.df_test = pd.concat([self.df_test.reset_index(), pd.DataFrame(ohe_test, columns = cols)], axis = 1).drop(['index'], axis=1)

        ## Drop the Geography column
        self.df_train.drop(['Geography'], axis = 1, inplace=True)
        self.df_val.drop(['Geography'], axis = 1, inplace=True)
        self.df_test.drop(['Geography'], axis = 1, inplace=True)


        ## Target Encoding the Surname Feature w/o data leakage
        means = self.df_train.groupby(['Surname']).Exited.mean()
        print('\nSurname Churn Averages: {}\n' .format(means.head(10)))

        global_mean = self.y_train.mean()

        ## Creating new encoded features for surname - Target (mean) encoding
        self.df_train['Surname_mean_churn'] = self.df_train.Surname.map(means)
        self.df_train['Surname_mean_churn'].fillna(global_mean, inplace=True)

        ## Calculate frequency of each category
        freqs = self.df_train.groupby(['Surname']).size()
        ## Create frequency encoding - Number of instances of each category in the data
        self.df_train['Surname_freq'] = self.df_train.Surname.map(freqs)
        self.df_train['Surname_freq'].fillna(0, inplace=True)
        ## Create Leave-one-out target encoding for Surname
        self.df_train['Surname_enc'] = ((self.df_train.Surname_freq * self.df_train.Surname_mean_churn) - self.df_train.Exited)/(self.df_train.Surname_freq - 1)
        ## Fill NaNs occuring due to category frequency being 1 or less
        self.df_train['Surname_enc'].fillna((((self.df_train.shape[0] * global_mean) - self.df_train.Exited) / (self.df_train.shape[0] - 1)), inplace=True)


        ## Apply the normal Target encoding mapping as obtained from the training set
        self.df_val['Surname_enc'] = self.df_val.Surname.map(means)
        self.df_val['Surname_enc'].fillna(global_mean, inplace=True)

        self.df_test['Surname_enc'] = self.df_test.Surname.map(means)
        self.df_test['Surname_enc'].fillna(global_mean, inplace=True)

        ## Show that using LOO Target encoding decorrelates features
        print('\nTarget Encoding Correlations: {}\n' .format(self.df_train[['Surname_mean_churn', 'Surname_enc', 'Exited']].corr()))

        ### Deleting the 'Surname' and other redundant column across the three datasets
        self.df_train.drop(['Surname_mean_churn'], axis=1, inplace=True)
        self.df_train.drop(['Surname_freq'], axis=1, inplace=True)
        self.df_train.drop(['Surname'], axis=1, inplace=True)
        self.df_val.drop(['Surname'], axis=1, inplace=True)
        self.df_test.drop(['Surname'], axis=1, inplace=True)


        print('\nEncoded Training Data: \n{}' .format(self.df_train.head()))
        print('\nEncoded Val Data: \n{}' .format(self.df_val.head()))
        print('\nEncoded Test Data: \n{}\n' .format(self.df_test.head()))


    def bivariate_visualizations(self):

        ## Check linear correlation (rho) between individual features and the target variable
        corr = self.df_train.corr()
        print(corr)

        ## Heatmap
        output = "Churn_Model/code/Modular/Output/Bivariate_Analysis/train_corr_heatmap.png" 
        sns.heatmap(corr, cmap = 'coolwarm')
        plt.savefig(output)
        plt.clf()   
        
        ## Churn vs Age Boxplot
        output = "Churn_Model/code/Modular/Output/Bivariate_Analysis/age_churn_boxplot.png" 
        sns.boxplot(x = "Exited", y = "Age", data = self.df_train, palette="Set3")
        plt.savefig(output)
        plt.clf()
      
        ## Churn vs Balance Violinplot
        output = "Churn_Model/code/Modular/Output/Bivariate_Analysis/balance_churn_violinplot.png" 
        sns.violinplot(x = "Exited", y = "Balance", data = self.df_train, palette="Set3")
        plt.savefig(output)
        plt.clf()

        # Check association of categorical features with target variable
        cat_vars_bv = ['Gender', 'IsActiveMember', 'country_Germany', 'country_France']
        for col in cat_vars_bv:
            print('\n{}' .format(self.df_train.groupby([col]).Exited.mean()))
        col = 'NumOfProducts'
        print('\n{}' .format(self.df_train.groupby([col]).Exited.mean()))

        print('\nNum Products Value Counts: \n{}\n' .format(self.df_train[col].value_counts()))



    def feature_engineering(self):

        # Creating some new features based on simple interactions between the existing features.
        eps = 1e-6

        self.df_train['bal_per_product'] = self.df_train.Balance/(self.df_train.NumOfProducts + eps)
        self.df_train['bal_by_est_salary'] = self.df_train.Balance/(self.df_train.EstimatedSalary + eps)
        self.df_train['tenure_age_ratio'] = self.df_train.Tenure/(self.df_train.Age + eps)
        self.df_train['age_surname_mean_churn'] = np.sqrt(self.df_train.Age) * self.df_train.Surname_enc

        new_cols = ['bal_per_product','bal_by_est_salary','tenure_age_ratio','age_surname_mean_churn']
        ## Ensuring that the new column doesn't have any missing values
        print('New Features Missing Values: \n{}\n' .format(self.df_train[new_cols].isnull().sum()))

        ## Linear association of new columns with target variables to judge importance
        output = "Churn_Model/code/Modular/Output/Bivariate_Analysis/new_feature_heatmap.png" 
        sns.heatmap(self.df_train[new_cols + ['Exited']].corr(), annot=True)
        plt.savefig(output)
        plt.clf()


        ## Creating new interaction feature terms for validation set
        self.df_val['bal_per_product'] = self.df_val.Balance/(self.df_val.NumOfProducts + eps)
        self.df_val['bal_by_est_salary'] = self.df_val.Balance/(self.df_val.EstimatedSalary + eps)
        self.df_val['tenure_age_ratio'] = self.df_val.Tenure/(self.df_val.Age + eps)
        self.df_val['age_surname_mean_churn'] = np.sqrt(self.df_val.Age) * self.df_val.Surname_enc

        ## Creating new interaction feature terms for test set
        self.df_test['bal_per_product'] = self.df_test.Balance/(self.df_test.NumOfProducts + eps)
        self.df_test['bal_by_est_salary'] = self.df_test.Balance/(self.df_test.EstimatedSalary + eps)
        self.df_test['tenure_age_ratio'] = self.df_test.Tenure/(self.df_test.Age + eps)
        self.df_test['age_surname_mean_churn'] = np.sqrt(self.df_test.Age) * self.df_test.Surname_enc


        ## Feature scaling and normalization the continuous variables (Logistic Rregression and SVM)
        self.cont_vars = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary', 'Surname_enc', 'bal_per_product',
                     'bal_by_est_salary', 'tenure_age_ratio', 'age_surname_mean_churn']             
        self.cat_vars = ['Gender', 'HasCrCard', 'IsActiveMember', 'country_France', 'country_Germany', 'country_Spain']
        cols_to_scale = self.cont_vars

        sc = StandardScaler()
        ## Scaling only continuous columns for the training set       
        sc_X_train = sc.fit_transform(self.df_train[cols_to_scale])
        ## Converting from array to dataframe and naming the respective features/columns
        self.sc_X_train = pd.DataFrame(data = sc_X_train, columns = cols_to_scale)

        ## Mapping learnt info on the continuous features for the val and test sets
        sc_map = {'mean':sc.mean_, 'std':np.sqrt(sc.var_)}

        ## Scaling validation and test sets by transforming the mapping obtained through the training set
        sc_X_val = sc.transform(self.df_val[cols_to_scale])
        sc_X_test = sc.transform(self.df_test[cols_to_scale])
        ## Converting val and test arrays to dataframes for re-usability
        self.sc_X_val = pd.DataFrame(data = sc_X_val, columns = cols_to_scale)
        self.sc_X_test = pd.DataFrame(data = sc_X_test, columns = cols_to_scale)


    def feature_selection(self):

        ## Creating feature-set and target for RFE (recursive feature elimination) model
        y = self.df_train['Exited'].values
        # Use the unscaled data for feature selection algorithm
        X = self.df_train[self.cat_vars + self.cont_vars]
        X.columns = self.cat_vars + self.cont_vars

        num_features_to_select = 10
        #params = {'alpha' : 1, 'kernel' : 'linear', 'gamma': 10}


        # for logistics regression
        est = LogisticRegression()
        rfe = RFE(estimator = est, n_features_to_select = num_features_to_select)
        rfe = rfe.fit(X.values, y)  
        print('\nLR Feature Bool Mask: {}' .format(rfe.support_))
        print('LR Feature Rankings: {}' .format(rfe.ranking_))

        mask = rfe.support_.tolist()
        self.selected_feats = [b for a,b in zip(mask, X.columns) if a]
        print('Logistic Regression Selected Features: {}' .format(self.selected_feats))

        # for decision trees
        est_dt = DecisionTreeClassifier(max_depth = 4, criterion = 'entropy')
        rfe_dt = RFE(estimator = est_dt, n_features_to_select = num_features_to_select) 
        rfe_dt = rfe_dt.fit(X.values, y)  
        print('\nDT Feature Bool Mask: {}' .format(rfe_dt.support_))
        print('DT Feature Rankings: {}' .format(rfe_dt.ranking_))

        mask = rfe_dt.support_.tolist()
        self.selected_feats_dt = [b for a,b in zip(mask, X.columns) if a]
        print('Decision Tree Selected Features: {}' .format(self.selected_feats_dt))


    def error_analysis(self, ea_model):

        # Unscaled features will be used since it's a tree model
        X_train = self.df_train.drop(columns = ['Exited'], axis = 1)
        X_val = self.df_val.drop(columns = ['Exited'], axis = 1)

        model = Pipeline(steps = [('categorical_encoding', CategoricalEncoder())
                                  ,('add_new_features', AddFeatures())
                                  ,('classifier', ea_model)
                                  ])

        ## Fit best model variation for error analysis
        model.fit(X_train, self.y_train)

        ## Making predictions on a copy of validation set
        df_ea = self.df_val.copy()
        df_ea['y_pred'] = model.predict(X_val)
        df_ea['y_pred_prob'] = model.predict_proba(X_val)[:,1]

        print('Error Analysis Sample Records: \n{}\n' .format(df_ea.sample(10)))

        ## Visualizing distribution of predicted probabilities
        output = "Churn_Model/code/Modular/Output/Error_Analysis/pred_probs_violinplot.png"     
        sns.violinplot(self.y_val.ravel(), df_ea['y_pred_prob'].values)
        plt.savefig(output)
        plt.clf()

        ## Check churn distribution with respect to age
        output = "Churn_Model/code/Modular/Output/Error_Analysis/age_churn_boxplot.png"     
        sns.boxplot(x = 'Exited', y = 'Age', data = df_ea)
        plt.savefig(output)
        plt.clf()

        ## Are we able to correctly identify pockets of high-churn customer regions in feature space?
        churn_ratio = df_ea.Exited.value_counts(normalize=True).sort_index()
        target_age_churn_ratio = df_ea[(df_ea.Age > 42) & (df_ea.Age < 53)].Exited.value_counts(normalize=True).sort_index()
        target_age_churn_ratio_pred = df_ea[(df_ea.Age > 42) & (df_ea.Age < 53)].y_pred.value_counts(normalize=True).sort_index()

        print('Churn ratio: \n{}\n' .format(churn_ratio))
        print('Targeted age range churn ratio: \n{}\n' .format(target_age_churn_ratio))
        print('Targeted age range churn prediction ratio: \n{}\n' .format(target_age_churn_ratio_pred))

        ## Checking correlation between features and target variable vs predicted variable
        x = df_ea[self.num_feat + ['y_pred', 'Exited']].corr()
        print(x[['y_pred','Exited']])

        ## Extracting the subset of incorrect predictions
        low_recall = df_ea[(df_ea.Exited == 1) & (df_ea.y_pred == 0)]
        low_prec = df_ea[(df_ea.Exited == 0) & (df_ea.y_pred == 1)]

        print('low recall data: \n{}\n' .format(low_recall.head()))
        print('low recall shape: {}\n' .format(low_recall.shape))

        print('low precision data: \n{}\n' .format(low_prec.head()))
        print('low precision shape: {}\n' .format(low_prec.shape))


        ## Visualize prediction probabilty distribution of errors causing low recall and low precision
        output = "Churn_Model/code/Modular/Output/Error_Analysis/low_recall_pred_prob_distribution.png"     
        sns.distplot(low_recall.y_pred_prob, hist=False)
        plt.savefig(output)
        plt.clf()

        output = "Churn_Model/code/Modular/Output/Error_Analysis/low_precision_pred_prob_distribution.png"     
        sns.distplot(low_prec.y_pred_prob, hist=False)
        plt.savefig(output)
        plt.clf()


        ## Tweaking the threshold of the classifier between .4 and .6
        steps = np.arange(.4, .61, .05, dtype=np.float16)
        for step in steps:

            threshold = step

            ## Predict on validation set with adjustable decision threshold
            probs = model.predict_proba(X_val)[:,1]
            val_preds = np.where(probs > threshold, 1, 0)

            cm = confusion_matrix(self.y_val, val_preds)
            cr = classification_report(self.y_val, val_preds)

            print('error analysis - confusion matrix with {} threshold: \n{}\n' .format(threshold, cm))
            print('error analysis - classification report matrix with {} threshold: {}\n' .format(threshold, cr))


        
        ## Checking whether the model has too much dependence on certain features

        # Examining low recall and low precision for 'NumOfProducts'
        num_products_ratio = df_ea.NumOfProducts.value_counts(normalize=True).sort_index()
        num_products_low_recall_ratio = low_recall.NumOfProducts.value_counts(normalize=True).sort_index()
        num_products_low_precision_ratio = low_prec.NumOfProducts.value_counts(normalize=True).sort_index()

        print('number of products ratio: \n{}\n' .format(num_products_ratio))
        print('number of products ratio in low recall set: \n{}\n' .format(num_products_low_recall_ratio))
        print('number of products ratio in low precision set: \n{}\n' .format(num_products_low_precision_ratio))

        # Examining low recall and low precision for 'IsActiveMember'
        isActive_ratio = df_ea.IsActiveMember.value_counts(normalize=True).sort_index()
        isActive_low_recall_ratio = low_recall.IsActiveMember.value_counts(normalize=True).sort_index()
        isActive_low_precision_ratio = low_prec.IsActiveMember.value_counts(normalize=True).sort_index()

        print('number of products ratio: \n{}\n' .format(isActive_ratio))
        print('number of products ratio in low recall set: \n{}\n' .format(isActive_low_recall_ratio))
        print('number of products ratio in low precision set: \n{}\n' .format(isActive_low_precision_ratio))


        ## Age distribution comparisons
        output = "Churn_Model/code/Modular/Output/Error_Analysis/age_dist_violinplot.png"     
        sns.violinplot(y = df_ea.Age)
        plt.savefig(output)
        plt.clf()

        output = "Churn_Model/code/Modular/Output/Error_Analysis/low_recall_age_dist_violinplot.png"     
        sns.violinplot(y = low_recall.Age)
        plt.savefig(output)
        plt.clf()

        output = "Churn_Model/code/Modular/Output/Error_Analysis/low_precision_age_dist_violinplot.png"     
        sns.violinplot(y = low_prec.Age)
        plt.savefig(output)
        plt.clf()


        ## Balance distribution comparisons
        output = "Churn_Model/code/Modular/Output/Error_Analysis/balance_dist_violinplot.png"     
        sns.violinplot(y = df_ea.Balance)
        plt.savefig(output)
        plt.clf()

        output = "Churn_Model/code/Modular/Output/Error_Analysis/low_recall_balance_dist_violinplot.png"     
        sns.violinplot(y = low_recall.Balance)
        plt.savefig(output)
        plt.clf()

        output = "Churn_Model/code/Modular/Output/Error_Analysis/low_precision_balance_dist_violinplot.png"     
        sns.violinplot(y = low_prec.Balance)
        plt.savefig(output)
        plt.clf()




    def get_cat_vars(self):
        return self.cat_vars

    def get_cont_vars(self):
        return self.cont_vars

    def get_sel_feats(self):
        return self.selected_feats

    def get_sel_feats_dt(self):
        return self.selected_feats_dt

    def get_df_train(self):
        return self.df_train

    def get_df_val(self):
        return self.df_val

    def get_df_test(self):
        return self.df_test

    def get_sc_X_train(self):
        return self.sc_X_train

    def get_sc_X_test(self):
        return self.sc_X_test

    def get_sc_X_val(self):
        return self.sc_X_val

    def get_y_train(self):
        return self.y_train

    def get_y_test(self):
        return self.y_test

    def get_y_val(self):
        return self.y_val





