%% Machine learning project
% We will download the datasets from the .txt files called 
% 'test_set.txt', 'Val_set.txt' and 'train_set.txt' that were generated in
% preprocessing.m and preprocessing2.m
% Then we will run Logistic Regression on the different sets

%% Initialization
clear ; close all; clc
%% =========== Part 1: Loading Data ============= %%
% We start by loading the data

fprintf("Loading data\n");
% load the train set 
range = [0 0 4560 0];
y = csvread('train_set.txt',0,0,range); % y contains the classes
range = [0 1 4560 8];
train = csvread('train_set.txt',0,1,range); % train contains the feature data

% load the validation set 
range = [0 0 1954 0];
y_Val = csvread('Val_set.txt',0,0,range);   % y_val contains the classes
range = [0 1 1954 8];
Val = csvread('Val_set.txt',0,1,range); % Val contains the feature data

% Load the test set
test = csvread('test_set.txt'); % the test_set.txt file doesn't contain classes

fprintf("Train set:\n");
disp(train(1:10,:));

fprintf("validation set:\n");
disp(Val(1:10,:));

fprintf("Test set:\n");
disp(test(1:10,:));

fprintf("Program paused. Press any key to continue\n");
pause;
%% =========== Part 2: Compute Cost and Gradient ============= %%
% Here we will compute the cost function and the gradient

fprintf("\nComputing cost function and gradient\n");

[m, n] = size(train);
[o, q] = size(Val);
[j, k] = size(test);

% Add ones to first column of train, validation and test
train = [ones(m, 1) train];
Val = [ones(o, 1) Val];
test = [ones(j, 1) test];

% Initialize fitting parameters with random parameters between [-10 10]
epsilon = 10;
initial_theta = rand(n + 1, 1)*(2*epsilon) - epsilon;

% Compute and display initial cost and gradient
[cost, grad] = costFunction(initial_theta, train, y);

fprintf("Cost at initial theta: ");
fprintf("%d", cost);
fprintf("\n");

fprintf("Program paused. Press any key to continue\n");
pause;
%% ============= Part 3: Optimizing using fminunc  =============
%  Here we will use fminunc to find the optimal parameters theta.

fprintf("\nOptimizing using fminunc\n");

%  Set options for fminunc
options = optimset('GradObj', 'on', 'MaxIter', 400);

%  Run fminunc to obtain the optimal theta
%  This function will return theta and the cost 
[theta, cost, flag, output] = ...
	fminunc(@(t)(costFunction(t, train, y)), initial_theta, options);

%  Run fminunc to obtain the optimal theta for log regr with regularization
%  This function will return theta_reg and the cost_reg 

lambda = 0.0000001;
[theta_reg, cost_reg, flag_reg, output_reg] = ...
   fminunc(@(t)(costFunctionReg(t, train, y, lambda)), initial_theta, options);

fprintf('Cost at optimal theta found by fminunc: %f\n', cost);
fprintf('Amount of iterations done by fminunc: %f\n', output.iterations);

fprintf('Cost at optimal theta found by fminunc with reg: %f\n', cost_reg);
fprintf('Amount of iterations done by fminunc with reg: %f\n', output_reg.iterations);

Val_cost = costFunctionReg(theta, Val, y_Val, lambda);
fprintf("Cost found for Validation set: %f\n", Val_cost);

fprintf("Program paused. Press any key to continue\n");
pause;
%% ============== Part 4: Accuracies ============== %%
% Now we will use the computed optimal theta to compute
% the accuracy of our algorithm on both the train and validation set.

fprintf("\nComputing results\n");

% Predict classes for train set
threshold = 0.5;
p = predict(theta, train, threshold);
p_reg = predict(theta_reg, train, threshold);

truePos = sum(p == y & y == 1);
falsePos = sum(p == 1 & y ~= 1);
falseNeg = sum(p == 0 & y ~= 0);
trueNeg = sum(p == 0 & y == 0);

fprintf('truePos: %f\n', truePos);
fprintf('falsePos: %f\n', falsePos);
fprintf('trueNeg: %f\n', trueNeg);
fprintf('falseNeg: %f\n', falseNeg);

precision = (truePos) / (truePos + falsePos);
recall = (truePos) / (truePos + falseNeg);
f1_score = (2 * precision * recall) /(precision + recall);

% Compute for log regr w/o regularization
fprintf('Train Accuracy: %f\n', mean(double(p == y)) * 100);
fprintf('Train Precision: %f\n', precision * 100);
fprintf('Train Recall: %f\n', recall * 100);
fprintf('Train f1 score: %f\n', f1_score * 100);

% Compute for log regr with regularization
fprintf('Train Accuracy with reg: %f\n', mean(double(p_reg == y)) * 100);

truePos = sum(p_reg == y & y == 1);
falsePos = sum(p_reg == 1 & y ~= 1);
falseNeg = sum(p_reg == 0 & y ~= 0);

precision = (truePos) / (truePos + falsePos);
recall = (truePos) / (truePos + falseNeg);
f1_score = (2 * precision * recall) /(precision + recall);

fprintf('Train Precision with reg: %f\n', precision * 100);
fprintf('Train Recall with reg: %f\n', recall * 100);
fprintf('Train f1 score: %f\n', f1_score * 100);

fprintf("Program paused. Press any key to calculate for validation set\n");
pause;
% -------------Now for the validation set:------------%
fprintf('Values for validation set\n');
% Compute accuracy on our validation set
threshold = 0.5;
p = predict(theta, Val, threshold);
p_reg = predict(theta_reg, Val, threshold);

% Compute for log regr w/o regularization
fprintf('Validation Accuracy: %f\n', mean(double(p == y_Val)) * 100);

truePos = sum(p == y_Val & y_Val == 1);
falsePos = sum(p == 1 & y_Val ~= 1);
falseNeg = sum(p == 0 & y_Val ~= 0);

precision = (truePos) / (truePos + falsePos);
recall = (truePos) / (truePos + falseNeg);
f1_score = (2 * precision * recall) /(precision + recall);

fprintf('Validation Precision: %f\n', precision * 100);
fprintf('Validation Recall: %f\n', recall * 100);
fprintf('Validation f1 score: %f\n', f1_score * 100);

% Compute for log regr with regularization
fprintf('Validation Accuracy with reg: %f\n', mean(double(p_reg == y_Val)) * 100);

truePos = sum(p_reg == y_Val & y_Val == 1);
falsePos = sum(p_reg == 1 & y_Val ~= 1);
falseNeg = sum(p_reg == 0 & y_Val ~= 0);

precision = (truePos) / (truePos + falsePos);
recall = (truePos) / (truePos + falseNeg);
f1_score = (2 * precision * recall) /(precision + recall);

fprintf('Validation Precision with reg: %f\n', precision * 100);
fprintf('Validation Recall with reg: %f\n', recall * 100);
fprintf('Validation f1 score: %f\n', f1_score * 100);
pause;

%% ============== Part 4: Predict ============== %%
p_t =  predict(theta, test, threshold);
% To try with Logistic Regression with Regularization:
%   p_t =  predict(theta_reg, test, threshold);

fprintf("\nPredictions for the test set: \n");
fprintf("%f", mean(p_t)*100);
fprintf(" percent of candidates have a ");
fprintf("%f", threshold*100);
fprintf(" percent chance of being an exoplanets\n");
fprintf("That means ");
fprintf("%d", sum(p_t==1));
fprintf(" planets\n");