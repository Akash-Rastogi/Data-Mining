% This script implements kNN Classifier

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

% getting labels
test_labels = test_data_file(1,:);

% finding Column size
[~,col_size] = size(test_labels);

% calculating number of classes
number_of_classes = col_size;

% number of samples in a class
[M,highest_frequency_of_data] = mode(train_data_file(1,1:end));

% creating an array of all class labels
class_label_vector = train_data_file(1,:);

% size of test
[number_of_rows_in_test,number_of_cols_in_test] = size(test_data);

% size of train
[number_of_rows_in_train,number_of_cols_in_train] = size(train_data);

% input from user
k_value = input('Enter value of K : ');

% zeros matrix for storing distances
distance_matrix = zeros(number_of_cols_in_test,number_of_cols_in_train);

% storing all the distances
for i = 1:number_of_cols_in_test
    for j = 1:number_of_cols_in_train
        distance_matrix(i,j) = norm(test_data(:,i) - train_data(:,j));
    end
end

% sorting all distances
[temp, temp_indexes] = sort(distance_matrix');
sorted_matrix = temp';
sorted_indexes = temp_indexes';

for i = 1:number_of_cols_in_test
    for j = 1:k_value
        vote(i, j) = class_label_vector(1, sorted_indexes(i,j));
    end
end

% getting outputs
result = mode(vote,2);

% calculating differences
difference = result' - test_labels;

for i = 1:col_size
    if ne(difference(1,i),0)
        difference(1,i) = 1;
    end
end

% calculating accuracy
accuracy = ((col_size - sum(difference))/col_size ) * 100;

% displaying results
disp('============================Results============================');

disp('Test Samples belong to the following classes:');
disp(result');
val = sprintf('Accuracy for kNN : %d',accuracy);
disp(val);
disp('============================Results============================');