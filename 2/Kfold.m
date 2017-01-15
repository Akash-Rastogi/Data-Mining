% This script implements k-Folds Cross Validation

% clearing screen
clc
% clearing variables
clearvars

% change accordingly
type = 'gene';  % 'hand 'or 'at&t'

% loading data from files
if type == 'hand'
    data = dlmread('HandWrittenLetters.txt');
elseif type == 'at&t'
    data = dlmread('ATNTFaceImage400.txt');
elseif type == 'gene'
    data = dlmread('GeneDataXY.txt');
end

% getting number of rows
col_size = size(data,2);

% user Input for k
% k_value = input('Enter the number of folds: ');
k_value = 5;

[accuracyK, accuracyL, accuracyC, accuracyS] = k_fold(k_value, data);

% plotting accuracy
hold on;
plotK = plot([1,2,3,4,5],accuracyK,'-or');
plotL = plot([1,2,3,4,5],accuracyL,'-ob');
plotC = plot([1,2,3,4,5],accuracyC,'-og');
plotS = plot([1,2,3,4,5],accuracyS,'-oc');

hline = refline([0 mean(accuracyK)]);
hline.Color = 'r';
hline = refline([0 mean(accuracyL)]);
hline.Color = 'b';
hline = refline([0 mean(accuracyC)]);
hline.Color = 'g';
hline = refline([0 mean(accuracyS)]);
hline.Color = 'c';
hold off;

% labelling the plot, x-axis, y-axis and title
title('k-Fold Cross Validation Accuracy');
xlabel('Number of folds') % x-axis label
ylabel('Accuracy Percentage') % y-axis label

label1 = 'kNN';
label2 = 'Linear Regression';
label3 = 'Centroid';
label4 = 'SVM';

% axis([1,5,0,100])
legend([plotK;plotL;plotC;plotS;], label1,label2,label3,label4);

% displaying results
disp('============================Results============================');
disp('Accuracy for kNN :');
disp(accuracyK);
disp(mean(accuracyK));
disp('Accuracy for Centroid :');
disp(accuracyC);
disp(mean(accuracyC));
disp('Accuracy for Linear Regression :');
disp(accuracyL);
disp(mean(accuracyL));
disp('Accuracy for SVM :');
disp(accuracyS);
disp(mean(accuracyS));
disp('============================Results============================');