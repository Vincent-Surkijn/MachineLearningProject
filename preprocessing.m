%% Machine learning project
% We will download the data from the .csv file called 
%'Dataset cumulative koi.csv'
% 

%% Initialization
clear ; close all; clc
%% =========== Part 1: Loading Data =============
% We start by loading the data

% Read data
data = readtable('Dataset cumulative koi.csv', 'TextType','string');

% Display headers and first 10 rows
disp(data(1:10,:));

% Print dimensions of data
fprintf("Dimensions of data: %d rows x %d columns\n", size(data));

%% =========== Part 2: Preprocessing Data =============
% Preprocessing is necessary because of:
%       - Missing Values
%       - Bias
%       - Labels need to be redefined as numbers
%       - Only need important columns    
% We will also be creating a training set

% Important columns
%koi_prad 		
%koi_dicco_msky
%koi_fpflag_nt 
%koi_fpflag_ss
%koi_fpflag_ec
%koi_fpflag_co
%koi_fittype
%koi_dikco_msky
% Only keep important columns
clean_data = table(data.koi_disposition, data.koi_prad, data.koi_dicco_msky,... 
                    data.koi_fpflag_nt, data.koi_fpflag_ss,data.koi_fpflag_ec,...
                    data.koi_fpflag_co,data.koi_fittype, data.koi_dikco_msky);
clean_data.Properties.VariableNames = {'koi_disposition' 'koi_prad' 'koi_dicco_msky' 'koi_fpflag_nt' 'koi_fpflag_ss',...
                                        'koi_fpflag_ec' 'koi_fpflag_co' 'koi_fittype' 'koi_dikco_msky'};
                
fprintf("Dimensions of clean data: %d rows x %d columns\n", size(clean_data));
disp(clean_data(1:10,:));

l = size(clean_data);
j = 1;
while(j<l(1)+1)
   if(strcmp(clean_data{j,8},"LS"))
        clean_data{j,8} = 0;              % 'LS'=0
   elseif(strcmp(clean_data{j,8},"MCMC"))
        clean_data{j,8} = 1;              % 'MCMC' = 1
   elseif(strcmp(clean_data{j,8},"DV"))
        clean_data{j,8} = 2;              % 'DV' = 2
   elseif(strcmp(clean_data{j,8},"none"))
        clean_data{j,8} = 3;              % 'none' = 3   
   else
       clean_data{j,8} = 4;               % 'LS+MCMC' = 4
   end
   j = j + 1;
end

% Convert strings to doubles
clean_data.koi_fittype = str2double(clean_data.koi_fittype);

i = 1;
train = zeros(l);
while(i<l(1)+1)
   if(strcmp(clean_data{i,1},"CONFIRMED"))
        clean_data{i,1} = 1;              % 'CONFIRMED'=1
   elseif(strcmp(clean_data{i,1},"FALSE POSITIVE"))
        clean_data{i,1} = 0;              % 'FALSE POSITIVE' = 0
   else
       clean_data{i,1} = 2;               % 'CANDIDATE' = 2
   end
   i = i + 1;
end

% Convert strings to doubles
clean_data.koi_disposition = str2double(clean_data.koi_disposition);

% Delete rows with NaN entries
clean_data = rmmissing(clean_data);

% Write result to new table
writetable(clean_data,'clean_data.csv');

disp(size(clean_data));

%% Next is preprocessing2.m %%