from pathlib import Path
import sys
import os

module_path = Path(__file__).parents[1]
sys.path.append(str(module_path))

from ML_Pipeline import DataPipeline


class Inference(DataPipeline):

    def __init__(self, model):
        super().__init__()
        self.churn_model = model

    def get_model_pipeline(self):    
        return self.make_pipeline(self.churn_model)
    
    def get_data(self):
        return self.data.drop(columns=['Exited'], axis=1), self.data['Exited'].values
