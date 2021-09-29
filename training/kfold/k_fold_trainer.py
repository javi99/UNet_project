import sys

sys.path.insert(1, 'D:\\UNet\\vegetation\\analysis')
from training_functions import model_training
from k_fold_evaluation_P_R_F1 import evaluate_model
import os 
import json
import numpy as np

seed = 24

path_folds = "D:\\UNet\\vegetation\\datasets\\folds"
path_weights = "D:\\UNet\\vegetation\\final_weights\\kfold"

results = dict()
precisions_list = []
recalls_list = []
F1_list = []

folds = os.listdir(path_folds)

for fold in folds:

	if len(fold.split(".")) > 1:
		continue

	dataset_name = os.path.join(path_folds, fold)
	
	model = model_training(path_weights,1, False, seed, dataset_name, fold+"_patches_fullDataset")

	print("training of " + fold + " done")

	mean_precision, mean_recall, mean_F1 = evaluate_model(model, True, seed, dataset_name)

	precisions_list.append(mean_precision)
	recalls_list.append(mean_recall)
	F1_list.append(mean_F1)
	
	
	results[fold] = {"precision" : mean_precision,
					"recall" : mean_recall,
					"F1" : mean_F1}


results["final"] = {"precision" : np.mean(np.array(precisions_list)),
					"recall" : np.mean(np.array(recalls_list)),
					"F1" : np.mean(np.array(F1_list))}

with open('results.txt', 'w') as outfile:
    json.dump(results, outfile)
	
