% This script implements Linear Discriminant Analysis

% clearing screen
clc
% clearing variables
clearvars

% change accordingly
type = 'gene';  % 'hand 'or 'at&t'

% user input for k value
% k_value = input('Enter the number of folds: ');
k_value = 5;

% initializing average accuracy variables
avg_accuracyK = [];
avg_accuracyL = [];
avg_accuracyC = [];
avg_accuracyS = [];
    
% number of dimensions
num_of_dimensions_labels=[5,10,20,30];

for num_of_dimensions=num_of_dimensions_labels

    % loading data from files
    if type == 'hand'
        data_file = dlmread('HandWrittenLetters.txt');
    elseif type == 'at&t'
        data_file = dlmread('ATNTFaceImage400.txt');
    elseif type == 'gene'
        data_file = dlmread('GeneDataXY.txt');
    end

    % Getting class labels from 1st Row
    class_labels = data_file(1,:);

    % Getting data without labels
    data_without_labels = data_file(2:end,:);

    [mappedX, mapping] = laplacian_eigen(data_without_labels', num_of_dimensions);

    data_file = vertcat(class_labels,mappedX');
    % getting data without labels
    data_without_labels = data_file(2:end,:);
    
    [accuracyK, accuracyL, accuracyC, accuracyS] = k_fold(k_value, data_file);

    % calculating average accuracy
    avg_accuracyK = [avg_accuracyK mean(accuracyK)];
    avg_accuracyL = [avg_accuracyL mean(accuracyL)];
    avg_accuracyC = [avg_accuracyC mean(accuracyC)];
    avg_accuracyS = [avg_accuracyS mean(accuracyS)];

end

% plotting accuracy
hold on;
plotK = plot(num_of_dimensions_labels, avg_accuracyK);
plotL = plot(num_of_dimensions_labels, avg_accuracyL);
plotC = plot(num_of_dimensions_labels, avg_accuracyC);
plotS = plot(num_of_dimensions_labels, avg_accuracyS);
hold off;

% labelling the plot, x-axis, y-axis and title
title('k-Fold Cross Validation Accuracy after Laplacian Embedding');
xlabel('Number of features'); % x-axis label
ylabel('Accuracy Percentage'); % y-axis label

label1 = 'kNN';
label2 = 'Linear Regression';
label3 = 'Centroid';
label4 = 'SVM';

legend([plotK;plotL;plotC;plotS], label1,label2,label3,label4);

% display Results
disp('===================================================');
disp('===================================================');
disp('Results : -')

disp('Accuracy for kNN               :');
disp(avg_accuracyK);

disp('Accuracy for Linear Regression :');
disp(avg_accuracyL);

disp('Accuracy for Centroid          :');
disp(avg_accuracyC);

disp('Accuracy for SVM               :');
disp(avg_accuracyS);

disp('===================================================');
disp('===================================================');