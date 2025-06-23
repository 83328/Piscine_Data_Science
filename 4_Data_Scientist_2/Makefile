PREDICTIONS_FILE ?= predictions.txt
TRUTH_FILE ?= truth.txt
DATA_FILE ?= Train_knight.csv
TEST_FILE ?= Test_knight.csv
TRAIN_DATA_FILE ?= Ex05_Training_knight.csv
TEST_DATA_FILE ?= Ex05_Validation_knight.csv
SHELL := /bin/bash

.PHONY: clean ex00 ex01 ex02 ex03 ex04 ex05 ex06

clean:
	@echo "Cleaning up..."
	@rm -f ./ex04/Tree.txt
	@rm -f ./ex05/KNN.txt
	@rm -f ./ex06/Regression.txt
	@rm -f ./ex06/Voting.txt

ex00:
	@echo "Running Exercise 00: Confusion Matrix..."
	python3 ex00/Confusion_Matrix.py $(PREDICTIONS_FILE) $(TRUTH_FILE)

ex01:
	@echo "Running Exercise 01: Heatmap..."
	python3 ex01/Heatmap.py $(DATA_FILE)

ex02:
	@echo "Running Exercise 02: Variances..."
	python3 ex02/Variances.py $(DATA_FILE)

ex03:
	@echo "Running Exercise 03: Feature Selection..."
	python3 ex03/Feature_Selection.py $(DATA_FILE)

ex04:
	@echo "Running Exercise 04: Decision Tree..."
	python3 ex04/Tree.py $(TRAIN_DATA_FILE) $(TEST_DATA_FILE)

ex05:
	@echo "Running Exercise 05: KNN..."
	python3 ex05/KNN.py $(TRAIN_DATA_FILE) $(TEST_DATA_FILE)

ex06:
	@echo "Running Exercise 06 Part 1: Logistic Regression"
	python3 ex06/Regression.py $(TRAIN_DATA_FILE) $(TEST_DATA_FILE)
	@echo "Running Exercise 06 Part 2: Democracy"
	python3 ex06/Democracy.py ./ex04/Tree.txt ./ex05/KNN.txt ./ex06/Regression.txt