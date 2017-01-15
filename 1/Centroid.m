% This script implements centroid Classifier

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

% number of classes
number_of_classes = col_size;

% number of samples in a class
[M,highest_frequency_of_data] = mode(train_data_file(1,1:end));

% creating training
identity_matrix_for_training = eye(number_of_classes);
number_of_ones = ones(1,highest_frequency_of_data);
training = kron(identity_matrix_for_training,number_of_ones);

% processing data
final_vector = (train_data * training') ./ highest_frequency_of_data;

for i = 1:number_of_classes
    temp = test_data(:,i);
    for j = 1:number_of_classes
        result(i,j) = sqrt(sum((final_vector(:,j) - temp) .^ 2));
    end
end

% calculating outputs
[min_values, result] = min(result,[],2);

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
val = sprintf('Accuracy for Centroid : %d',accuracy);
disp(val);

disp('============================Results============================');