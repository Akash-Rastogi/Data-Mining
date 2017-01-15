% This script implements Feature Selection
clc
% clearing variables
clearvars

% change accordingly
type = 'gene';  % 'hand 'or 'at&t'

% loading data from files
if type == 'hand'
    data_file = load('HandWrittenLetters.txt');
elseif type == 'at&t'
    data_file = load('ATNTFaceImage400.txt');
elseif type == 'gene'
    data_file = load('GenomeTrainXY.txt');
end

% Test Data
test_data = load('GenomeTestX.txt');

% Getting class labels from 1st Row
class_labels = data_file(1,:);

% data from data file, without labels
data = data_file(2:end,:);

% finding Unique Values
[unique_values,~,index] = unique(data(1,:));
number_of_classes = numel(unique_values);

% getting Number of Rows and Columns
[number_of_rows,number_of_columns] = size(data);

% calculating the mean of each row
group_mean = mean(data,2);

% calculating mean of all group means
overall_mean = mean(group_mean);

%% Calculate the F value. The F Value is calculated using the formula 
% F = (SSE1 - SSE2 / m) / SSE2 / n-k, 
% where SSE = residual sum of squares, m = number of restrictions
% and k = number of independent variables.

% calculating rms which is (group mean - overall mean)^2*n
root_mean_square = ((group_mean - overall_mean).^2).*number_of_columns;

% calculating features
features = number_of_rows-1;

% calculating MSb = Sb/fb, where MSb is 'explained variance'
rms_per_feature = root_mean_square./features;

% calculating sw where sw equal to subtract group mean from each element in
%a group than square the diffrence and add all the values together.
sw = zeros(number_of_rows,1);
for i=1:number_of_rows
    sw(i,1) = sum((data(i,1:end)-group_mean(i,1)).^2); 
end

% calculating fw which is a(n-1)
fw = (number_of_columns-1)*number_of_classes;

% calculating Msw = Sw/fw, where Msw is 'unexplained variance'
MSw = sw./fw;

% calculating the F-ratio
% F=(explained variance)/(unexplained variance)
F=rms_per_feature./MSw;
%%

% I stores the index value of the sorted F-ratio
[B,I]= sort(F, 'descend');

% sorting data
sorted_data = data(I,:);

% adding labels
sorted_data = vertcat(class_labels,sorted_data);

% initializing average accuracy variables
avg_accuracyK = [];
avg_accuracyL = [];
avg_accuracyC = [];
avg_accuracyS = [];

% number of features
num_of_features_labels=[100];

for num_of_features=num_of_features_labels
    train_data = sorted_data(1:num_of_features,:);
    
    k_value = 5;

    [accuracyK, accuracyL, accuracyC, accuracyS] = k_fold(k_value, train_data, test_data);
    
    % calculating average accuracy
    avg_accuracyK = [avg_accuracyK mean(accuracyK)]
    avg_accuracyL = [avg_accuracyL mean(accuracyL)]
    avg_accuracyC = [avg_accuracyC mean(accuracyC)]
    avg_accuracyS = [avg_accuracyS mean(accuracyS)]
end

% % plotting accuracy
% hold on;
% plotK = plot(num_of_features_labels, avg_accuracyK);
% plotL = plot(num_of_features_labels, avg_accuracyL);
% plotC = plot(num_of_features_labels, avg_accuracyC);
% plotS = plot(num_of_features_labels, avg_accuracyS);
% hold off;
% 
% % labelling the plot, x-axis, y-axis and title
% title('k-Fold Cross Validation Accuracy after f Statistic');
% xlabel('Number of features'); % x-axis label
% ylabel('Accuracy Percentage'); % y-axis label
% 
% label1 = 'kNN';
% label2 = 'Linear Regression';
% label3 = 'Centroid';
% label4 = 'SVM';
% 
% legend([plotK;plotL;plotC;plotS], label1,label2,label3,label4);
hold on;
hline1 = refline([0 mean(accuracyK)]);
hline1.Color = 'r';
hline2 = refline([0 mean(accuracyL)]);
hline2.Color = 'b';
hline3 = refline([0 mean(accuracyC)]);
hline3.Color = 'g';
hline4 = refline([0 mean(accuracyS)]);
hline4.Color = 'c';
hold off;

label1 = 'kNN';
label2 = 'Linear Regression';
label3 = 'Centroid';
label4 = 'SVM';
legend([hline1;hline2;hline3;hline4], label1,label2,label3,label4);

% displaying results
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

disp('Top 100  indices               :');
disp(I(1:100));

disp('===================================================');
disp('===================================================');