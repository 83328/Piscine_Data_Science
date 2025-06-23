PREDICTIONS_FILE ?= predictions.txt
TRUTH_FILE ?= truth.txt
DATA_FILE ?= Train_knight.csv
TEST_FILE ?= Test_knight.csv
TRAIN_DATA_FILE ?= Ex05_Training_knight.csv
TEST_DATA_FILE ?= Ex05_Validation_knight.csv
SHELL := /bin/bash

.PHONY: clean ex00 ex01 ex02 ex03 ex04 ex05 ex06

clean:
	@echo "nothing to clean so far..."

ex00:
	@echo "Running Exercise 00: Confusion Matrix"
	@echo "Predictions file: $(PREDICTIONS_FILE)"
	@echo "Truth file: $(TRUTH_FILE)"
	python3 ex00/Confusion_Matrix.py $(PREDICTIONS_FILE) $(TRUTH_FILE)

ex01:
	@echo "Running Exercise 01: Heatmap"
	@echo "Data file: $(DATA_FILE)"
	python3 ex01/Heatmap.py $(DATA_FILE)

ex02:
	@echo "Running Exercise 02: Variances"
	@echo "Data file: $(DATA_FILE)"
	python3 ex02/Variances.py $(DATA_FILE)

ex03:
	@echo "Running Exercise 03: Feature Selection"
	@echo "Data file: $(DATA_FILE)"
	python3 ex03/Feature_Selection.py $(DATA_FILE)

ex04:
	@echo "Running Exercise 04: Decision Tree"
	@echo "Training data file: $(TRAIN_DATA_FILE)"
	@echo "Testing data file: $(TEST_DATA_FILE)"
	@echo "Truth labels for F1-score: $(TRUTH_FILE)"
	python3 ex04/Tree.py $(TRAIN_DATA_FILE) $(TEST_DATA_FILE) $(TRUTH_FILE)

ex05:
	@echo "Running Exercise 05: KNN"
	python3 ex05/KNN.py $(TRAIN_DATA_FILE) $(TEST_DATA_FILE)

ex06:
	@echo "Running Exercise 06: Logistic Regression"
	python3 ex06/Democracy.py $(TRAIN_DATA_FILE) $(TEST_DATA_FILE)