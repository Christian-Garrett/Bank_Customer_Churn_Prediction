import joblib
from sklearn.pipeline import Pipeline
from pathlib import Path
import sys
import os

module_path = Path(__file__).parents[1]
sys.path.append(str(module_path))

from ML_Pipeline.CategoricalEncoder import CategoricalEncoder
from ML_Pipeline.AddFeatures import AddFeatures
from ML_Pipeline.CustomScaler import CustomScaler


def add_pipeline_steps(self, model, scale_features=None):

    if scale_features:
        result = Pipeline(steps = [('categorical_encoding', CategoricalEncoder()),
                                   ('add_new_features', AddFeatures()),
                                   ('standard_scaling', CustomScaler(self.cols_to_scale)),
                                   ('classifier', model)
                                   ])
    else:
        result = Pipeline(steps = [('categorical_encoding', CategoricalEncoder()),
                                   ('add_new_features', AddFeatures()),
                                   ('classifier', model)
                                   ])

    return result


def make_pipeline(self, model):

    if (str(model).find('KNeighborsClassifier') != -1):
        pipe = self.add_pipeline_steps(model, True)
    else:
        pipe = self.add_pipeline_steps(model)
    
    return pipe


def save_final_model(self):

    ## Save model object
    joblib.dump(self.final_model,
                os.path.join(self.output_path,
                             "Final_Model/final_churn_model_v1_0.sav"))
