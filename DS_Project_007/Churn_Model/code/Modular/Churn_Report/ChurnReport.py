

def create_churn_report():

    '''## Load model object
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
    high_churn_list.to_csv(output, index = False)'''


if __name__ == '__main__':
    create_churn_report()