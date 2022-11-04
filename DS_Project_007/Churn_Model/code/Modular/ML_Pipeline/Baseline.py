from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, f1_score, recall_score, confusion_matrix, classification_report
from sklearn.decomposition import PCA
from sklearn import tree
#from sklearn.tree import DecisionTreeClassifier
#from sklearn.tree import export_graphviz
#from IPython.display import Image

import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



class Baseline:

    def __init__(self, data, lrP, svmP, dtP, num_pca_comps = 2):

        self.y_train = data.get_y_train()
        self.weight_dict = self.create_weight_dict() #weights are based on the training set
        self.y_test = data.get_y_test()
        self.y_val = data.get_y_val()
        self.selected_cat_vars = [x for x in data.get_sel_feats() if x in data.get_cat_vars()]
        self.selected_cont_vars = [x for x in data.get_sel_feats() if x in data.get_cont_vars()]
        self.X_train = np.concatenate((data.get_df_train()[self.selected_cat_vars].values, data.get_sc_X_train()[self.selected_cont_vars].values), axis = 1)        
        self.X_val = np.concatenate((data.get_df_val()[self.selected_cat_vars].values, data.get_sc_X_val()[self.selected_cont_vars].values), axis = 1)
        self.X_test = np.concatenate((data.get_df_test()[self.selected_cat_vars].values, data.get_sc_X_test()[self.selected_cont_vars].values), axis = 1)
        self.num_pca_comps = num_pca_comps
        self.selected_feats = data.get_sel_feats()
        self.selected_feats_dt = data.get_sel_feats_dt()
        self.df_train = data.get_df_train()
        self.df_val = data.get_df_val()
        self.y_train = data.get_y_train()
        self.y_val = data.get_y_val()
        self.X_train_dt = self.df_train[self.selected_feats_dt].values # use original unscaled features
        self.X_val_dt = self.df_val[self.selected_feats_dt].values # use original unscaled features


        ## Defining the LR Model
        self.lr = LogisticRegression(C = lrP[0], penalty = lrP[1]
                                     , class_weight = self.weight_dict, n_jobs = lrP[2])

        ## Defining the SVM Model
        self.svm = SVC(C = svmP[0], kernel = svmP[1], class_weight = self.weight_dict)

        ## Transforming the dataset using PCA (linear and nonlinear)
        self.pca = PCA(n_components = self.num_pca_comps)

        self.pca_X_train = self.pca.fit_transform(self.X_train)
        self.pca_explained_variance_ratio = self.pca.explained_variance_ratio_   

        self.pca_X_train_dt = self.pca.fit_transform(self.X_train_dt)
        self.pca_explained_variance_ratio_dt = self.pca.explained_variance_ratio_   
        


        ## Define the Decision Tree model
        self.clf = tree.DecisionTreeClassifier(criterion = dtP[0], class_weight = self.weight_dict, max_depth = dtP[1],
                                         max_features = dtP[2], min_samples_split = dtP[3], min_samples_leaf = dtP[4])


    def create_weight_dict(self):

        # Obtaining class weights based on the 'num churn class samples' imbalance ratio
        _, num_samples = np.unique(self.y_train, return_counts = True) 
        weights = np.max(num_samples)/num_samples

        weights_dict = dict()
        class_labels = [0,1]
        for a,b in zip(class_labels,weights):
            weights_dict[a] = b

        return weights_dict


    def get_lr_metrics(self, train = True):

        print('\nlr model parameters: {}' .format(self.selected_cat_vars + self.selected_cont_vars))

        if train:
            
            ## Fitting model
            self.lr.fit(self.X_train, self.y_train)

            ## Fitted model parameters
            print('lr training model coefficients: {}' .format(self.lr.coef_))
            print('lr training model intercepts: {}' .format(self.lr.intercept_))

            ## Training metrics
            rac = roc_auc_score(self.y_train, self.lr.predict(self.X_train))
            rs = recall_score(self.y_train, self.lr.predict(self.X_train))
            cm = confusion_matrix(self.y_train, self.lr.predict(self.X_train))
            cr = classification_report(self.y_train, self.lr.predict(self.X_train))

        else:

            ## Fitting model
            self.lr.fit(self.X_val, self.y_val)

            ## Fitted model parameters
            print('lr validation model coefficients: {}' .format(self.lr.coef_))
            print('lr validation model intercepts: {}' .format(self.lr.intercept_))

            ## Training metrics
            rac = roc_auc_score(self.y_val, self.lr.predict(self.X_val))
            rs = recall_score(self.y_val, self.lr.predict(self.X_val))
            cm = confusion_matrix(self.y_val, self.lr.predict(self.X_val))
            cr = classification_report(self.y_val, self.lr.predict(self.X_val))

        return rac, rs, cm, cr


    def get_svm_metrics(self, train = True):

        print('\nSVM model parameters: {}' .format(self.selected_cat_vars + self.selected_cont_vars))

        if train:

            ## Fitting the model
            self.svm.fit(self.X_train, self.y_train)

            ## Fitted model parameters
            print('svm training model coefficients: {}' .format(self.svm.coef_))
            print('svm training model intercepts: {}' .format(self.svm.intercept_))

            ## Training metrics
            rac = roc_auc_score(self.y_train, self.svm.predict(self.X_train))
            rs = recall_score(self.y_train, self.svm.predict(self.X_train))
            cm = confusion_matrix(self.y_train, self.svm.predict(self.X_train))
            cr = classification_report(self.y_train, self.svm.predict(self.X_train))

        else:

            ## Fitting the model
            self.svm.fit(self.X_val, self.y_val)

            ## Fitted model parameters
            print('svm validation model coefficients: {}' .format(self.svm.coef_))
            print('svm validation model intercepts: {}' .format(self.svm.intercept_))

            ## Training metrics
            rac = roc_auc_score(self.y_val, self.svm.predict(self.X_val))
            rs = recall_score(self.y_val, self.svm.predict(self.X_val))
            cm = confusion_matrix(self.y_val, self.svm.predict(self.X_val))
            cr = classification_report(self.y_val, self.svm.predict(self.X_val))

        return rac, rs, cm, cr


    def get_pca_linear_boundaries(self):

        print('\nThe variance explained by the reduced PCA features in the linear model: {}\n' .format(self.pca_explained_variance_ratio))

        # Creating a mesh region where the boundary will be plotted
        x_min, x_max = self.pca_X_train[:, 0].min() - 1, self.pca_X_train[:, 0].max() + 1
        y_min, y_max = self.pca_X_train[:, 1].min() - 1, self.pca_X_train[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

        ## Fitting LR model on 2 features
        self.lr.fit(self.pca_X_train, self.y_train)

        ## Fitting SVM model on 2 features
        self.svm.fit(self.pca_X_train, self.y_train)

        ## Plotting decision boundary for LR
        z1 = self.lr.predict(np.c_[xx.ravel(), yy.ravel()])
        z1 = z1.reshape(xx.shape)

        ## Plotting decision boundary for SVM
        z2 = self.svm.predict(np.c_[xx.ravel(), yy.ravel()])
        z2 = z2.reshape(xx.shape)

        return xx, yy, z1, z2


    def get_dt_metrics(self, train = True):

        print('\nDecision Tree parameters: {}' .format(self.selected_feats_dt))    
        
        if train:

            self.clf.fit(self.X_train_dt, self.y_train)
           
            ## Training metrics
            rac = roc_auc_score(self.y_train, self.clf.predict(self.X_train_dt))
            rs = recall_score(self.y_train, self.clf.predict(self.X_train_dt))
            cm = confusion_matrix(self.y_train, self.clf.predict(self.X_train_dt))
            cr = classification_report(self.y_train, self.clf.predict(self.X_train_dt))

        else:

            self.clf.fit(self.X_val, self.y_val)
           
            ## Training metrics
            rac = roc_auc_score(self.y_val, self.clf.predict(self.X_val_dt))
            rs = recall_score(self.y_val, self.clf.predict(self.X_val_dt))
            cm = confusion_matrix(self.y_val, self.clf.predict(self.X_val_dt))
            cr = classification_report(self.y_val, self.clf.predict(self.X_val_dt))

        ## Checking the importance of different features of the model
        print('\nDecision Tree Feature Importance: \n{}\n' .format(pd.DataFrame({'features': self.selected_feats_dt,'importance': self.clf.feature_importances_}).sort_values(by = 'importance', ascending=False)))  

        return rac, rs, cm, cr


    def get_pca_nonlinear_boundary(self):

        print('\nThe variance explained by the reduced PCA features in the non linear model: {}\n' .format(self.pca_explained_variance_ratio_dt))

        # Creating a mesh region where the boundary will be plotted
        x_min, x_max = self.pca_X_train_dt[:, 0].min() - 1, self.pca_X_train_dt[:, 0].max() + 1
        y_min, y_max = self.pca_X_train_dt[:, 1].min() - 1, self.pca_X_train_dt[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 100), np.arange(y_min, y_max, 100))

        ## Fitting tree model on 2 features
        self.clf.fit(self.pca_X_train_dt, self.y_train)

        ## Plotting decision boundary for Decision Tree (DT)
        z = self.clf.predict(np.c_[xx.ravel(), yy.ravel()])
        z = z.reshape(xx.shape)

        return xx, yy, z


    def create_dt_visualization(self):

        # Change the max-depth variable of the model to 3 for the viz purposes
        clf = tree.DecisionTreeClassifier(criterion = 'entropy', class_weight = self.weight_dict, max_depth = 3, max_features = None
                            , min_samples_split = 25, min_samples_leaf = 15)

        clf.fit(self.X_train_dt, self.y_train)

        out_image = 'Churn_Model/code/Modular/Output/Decision_Boundaries/dt_visualization.png'
        plt.figure(figsize=(15,10))
        tree.plot_tree(clf, filled=True)
        plt.savefig(out_image)
        plt.clf()


        ## Export as dot file
        #output = 'Churn_Model/code/Modular/Output/Decision_Boundaries/dt_visualization.dot'
        #out_image = 'Churn_Model/code/Modular/Output/Decision_Boundaries/dt_visualization.png'
        #dot_data = export_graphviz(clf, out_file = output
        #                          , feature_names = self.selected_feats_dt
        #                          , class_names = ['Did not churn', 'Churned']
        #                          , rounded = True, proportion = False
        #                          , precision = 2, filled = True)

        ## Convert to png using system command (requires Graphviz)
        #subprocess.run(['dot', '-Tpng', output, '-o', out_image, '-Gdpi=600'], shell=True)
       
          

    def get_y_train(self):
        return self.y_train

    def get_pca_X_train(self):
        return self.pca_X_train

    def get_pca_X_train_dt(self):
        return self.pca_X_train_dt

    def get_weights_dict(self):
        return self.weight_dict


