% This script implements Laplacian Embedding

% clearing screen
clc
% clearing variables
clearvars

% change accordingly
type = 'gene';  % 'hand 'or 'at&t'

% user input for k value
% k_value = input('Enter the number of folds: ');
k_value = 5;

% loading data from files
if type == 'hand'
    data_file = dlmread('HandWrittenLetters.txt');
elseif type == 'at&t'
    data_file = dlmread('ATNTFaceImage400.txt');
elseif type == 'gene'
    data_file = dlmread('LDAallXY.txt');
end

% finding Unique Values
[unique_values,~,index] = unique(data_file(1,:));
num_of_dimensions = numel(unique_values);

% Getting class labels from 1st Row
class_labels = data_file(1,:);

% Getting data without labels
data_without_labels = data_file(2:end,:);

[mappedX, mapping] = lda(data_without_labels', class_labels, num_of_dimensions-1);

data_file = vertcat(class_labels,mappedX');
% getting data without labels
data_without_labels = data_file(2:end,:);

[accuracyK, accuracyL, accuracyC, accuracyS] = k_fold(k_value, data_file);

% % plotting accuracy
% hold on;
% % plotK = plot([1,2,3,4,5],accuracyK,'-or');
% % plotL = plot([1,2,3,4,5],accuracyL,'-ob');
% % plotC = plot([1,2,3,4,5],accuracyC,'-og');
% % plotS = plot([1,2,3,4,5],accuracyS,'-oc');
% 
% hline1 = refline([0 mean(accuracyK)]);
% hline1.Color = 'r';
% hline2 = refline([0 mean(accuracyL)]);
% hline2.Color = 'b';
% hline3 = refline([0 mean(accuracyC)]);
% hline3.Color = 'g';
% hline4 = refline([0 mean(accuracyS)]);
% hline4.Color = 'c';
% hold off;
% 
% % labelling the plot, x-axis, y-axis and title
% title('k-Fold Cross Validation Accuracy after Linear Discriminant Analysis');
% xlabel('Number of folds'); % x-axis label
% ylabel('Accuracy Percentage'); % y-axis label
% 
% label1 = 'kNN';
% label2 = 'Linear Regression';
% label3 = 'Centroid';
% label4 = 'SVM';
% 
% % legend([plotK;plotL;plotC;plotS], label1,label2,label3,label4);
% legend([hline1;hline2;hline3;hline4], label1,label2,label3,label4);

% display Results
disp('===================================================');
disp('===================================================');
disp('Results : -')

disp('Accuracy for kNN               :');
disp(accuracyK);
disp(mean(accuracyK));
disp('Accuracy for Linear Regression :');
disp(accuracyL);
disp(mean(accuracyL));
disp('Accuracy for Centroid          :');
disp(accuracyC);
disp(mean(accuracyC));
disp('Accuracy for SVM               :');
disp(accuracyS);
disp(mean(accuracyS));
disp('===================================================');
disp('===================================================');