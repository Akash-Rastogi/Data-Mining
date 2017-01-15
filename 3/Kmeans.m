% This script implements k Means Algorithm
clc
% clearing variables
clearvars

% change accordingly
type = 'hand';  % 'hand 'or 'at&t'

% loading data from files
if type == 'hand'
    data_file = load('HandWrittenLetters.txt');
elseif type == 'at&t'
    data_file = load('ATNTFaceImage400.txt');
elseif type == 'gene'
    data_file = load('GeneDataXY.txt');
end

% finding Unique Values
[unique_values,~,index] = unique(data_file(1,:));  
number_of_classes = numel(unique_values);

% class Labels from 1st Row
class_labels = data_file(1,:);

% data without labels
data_load = data_file(2:size(data_file, 1),:);

[~,number_of_columns]=size(data_load);

% K Means call for calculating output
[Kin,C,sumd,D] = kmeans(data_load', number_of_classes);

% getting Confusion Matrix
[confusion_matrix, order] = confusionmat(class_labels, Kin');

% Hungarian/Munkres Algorithm
munkres_values = munkres1(~confusion_matrix);
re_ordered_con_matrix = confusion_matrix(:,munkres_values);

% summing diagonal Elements
sum_of_diagonals = trace(re_ordered_con_matrix);

% calculating accuracy
accuracy = (sum_of_diagonals / number_of_columns) * 100;

% displaying results
disp('============================Results============================');
val_to_display = sprintf('Accuracy for kMeans : %d', accuracy);
disp(val_to_display);
disp('============================Results============================');