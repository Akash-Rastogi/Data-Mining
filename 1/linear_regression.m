% This script implements Linear Regression Classifier

% clearing screen
clc
% clearing variables
clearvars

% loading data from files
train_data_file = load('ATNT50/trainDataXY.txt');
test_data_file = load('ATNT50/testDataXY.txt');

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
identity_matrix_for_train = eye(number_of_classes);
number_of_ones = ones(1,highest_frequency_of_data);
training = kron(identity_matrix_for_train,number_of_ones);

% learning
learned_data = pinv(train_data') * training';
learned_test = learned_data' * test_data;

% getting outputs
[learned_test_value  result]= max(learned_test,[],1);

% calculating differences
difference = result - test_labels;

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
disp(result);
val = sprintf('Accuracy for Linear Regression : %d',accuracy);
disp(val);

disp('============================Results============================');