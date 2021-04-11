%% Machine learning project
% We will download the cleaned data from the .csv file called 
%'clean_data.csv'
% 
%% =========== Part 1: Loading Data =============
% We start by loading the data

% Load cleaned data
clean_data = csvread('clean_data.csv', 1);

% Load table version of cleaned data for displaying the column names
table_data = readtable('clean_data.csv');

disp(clean_data(1:10,:));
disp(table_data(1:10,:));
pause;

%% =========== Part 2: Preprocessing Data =============
% We still have some processing to do:
%
%   We need to scale the variables
%
%   We need to create a train and a test set

%% Scaling

temp = (clean_data(:,2:9) - min(clean_data(1,2:9)))./(max(clean_data(:,2:9)) - min(clean_data(:,2:9)));
% Class in column 1 shouldn't be scaled -> (:,2:9)
clean_data = [ clean_data(:,1) temp];

fprintf("Scaled clean_data: \n");
disp(clean_data(1:10,:));

pause;
%% Creating train and test set
% First shuffle the data
clean_data = clean_data( randperm( length(clean_data) ) , : );
fprintf("Shuffled data: \n");
disp(clean_data(1:10,:));

i = 1;
j = 1;
k = 1;
l = size(clean_data);
train = zeros(4561 , l(2));
Val = zeros(1955, l(2));
test = zeros(sum(clean_data(:,1)==2), l(2));
while(i<l(1)+1)
   if(clean_data(i,1)==1 || clean_data(i,1)==0)
       if(j<=4561)
            train(j,:) = clean_data(i,:);   % train set gets 70% or 4561 values
       else
           Val(j - 4561,:) = clean_data(i,:);   % validation set gets the remaining 30%
       end
       j = j + 1;
   else
       test(k,:) = clean_data(i,:);     % test set gets samples with class "CANDIDATE" which corresponds to the value 2
       k = k + 1;
   end  
   i = i + 1;
end    

fprintf("Train: \n");
disp(train(1:10,:));
disp(size(train));

fprintf("Val: \n");
disp(Val(1:10,:));
disp(size(Val));

fprintf("Test: \n");
disp(test(1:10,:));

disp(size(test));

% write sets to .txt files
csvwrite('train_set.txt', train);
csvwrite('Val_set.txt', Val);
csvwrite('test_set.txt', test(:,2:9));
%% Now the preprocessing is done %%