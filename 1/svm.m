% This script implements SVM

% clearing screen
clc
% clearing variables
clearvars

% loading data from files
train_data_file = load('ATNT200/trainDataXY.txt');
test_data_file = load('ATNT200/testDataXY.txt');

% creating test and train data from loaded files
train_data = train_data_file(2:end,:);
test_data = test_data_file(2:end,:);

% creating an array of all class labels
training_label_vector = train_data_file(1,:);
testing_label_vector = test_data_file(1,:);

% model = svmtrain(training_label_vector, training_instance_matrix [, 'libsvm_options']);
model = svmtrain(training_label_vector', train_data', '-t 0')
% [predicted_label, accuracy, decision_values/prob_estimates] = svmpredict(testing_label_vector, testing_instance_matrix, model [, 'libsvm_options']);
[predict_label, accuracy, prob_values] = svmpredict(testing_label_vector', test_data', model);

% displayng results
disp('============================Results============================');
disp('Test Samples belong to the following classes:');
disp(predict_label');
disp('============================Results============================');