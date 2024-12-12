from pathlib import Path
import sys

module_path = Path(__file__).parents[1]
sys.path.append(str(module_path))

from ML_Pipeline import DataPipeline


dp_object = DataPipeline()

dp_object.perform_EDA()
dp_object.perform_feature_engineering()
# dp_object.review_baseline_models()
# dp_object.perform_model_spotchecks()
dp_object.build_final_model()
