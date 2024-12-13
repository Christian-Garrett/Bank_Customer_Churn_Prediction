from pathlib import Path
import sys
import os
import joblib
import pandas as pd
import numpy as np

module_path = Path(__file__).parents[1]
sys.path.append(str(module_path))

from InferenceClass import Inference

PREDICTION_THRESH = 0.45
PROB_THRESH = 0.70


def create_churn_report():

    model = joblib.load(os.path.join(module_path,
                                     "Output/Final_Model/final_churn_model_v1_0.sav"))
    inference_object = Inference(model)

    mod_pipe = inference_object.get_model_pipeline()
    X, y = inference_object.get_data()
    mod_pipe.fit(X, y)

    print(f'X shape: \n{X.shape}\n')
    print(f'y shape: \n{y.shape}\n')

    probs = mod_pipe.predict_proba(X)[:, 1]
    preds = np.where(probs > PREDICTION_THRESH, 1, 0)

    X['preds'] = preds
    X['pred_probs'] = probs

    ## Creating a list of customers most likely to churn - probability > PROB_THRESH
    high_churn_list = \
        X[X.pred_probs > PROB_THRESH].sort_values(by=['pred_probs'],
                                                      ascending=False
                                                      ).reset_index().drop(columns=['index',
                                                                                    'preds'],
                                                                                    axis=1)
    print(f'High Churn List Shape: \n{high_churn_list.shape}\n')
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(f'High Churn List Head: \n{high_churn_list.head()}\n')

    ## Save list to output file
    high_churn_list.to_csv(os.path.join(module_path, "Output/high_churn_121224.csv"), index = False)


if __name__ == '__main__':
    create_churn_report()
