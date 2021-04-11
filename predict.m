function p = predict(theta, X, threshold)
% Predict whether the label is 0 or 1 using learned logistic 
% regression parameters theta
% p = PREDICT(theta, X) computes the predictions for X using a 
% variable threshold (i.e., if sigmoid(theta'*x) >= threshold, predict 1)

m = size(X, 1); % Number of training examples

p = zeros(m, 1);

% Initialize the hypothesis
h = sigmoid(X*theta);

i = 1;
while(i<m+1)
    if(h(i)>=threshold)
        p(i)=1;
    else
        p(i)=0;
    end    
    i=i+1;
end    

end
