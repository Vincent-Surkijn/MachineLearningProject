function [J, grad] = costFunction(theta, X, y)
% Compute cost and gradient for logistic regression
% J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
% parameter for logistic regression and the gradient of the cost
% w.r.t. to the parameters.

m = length(y); % number of training examples

J = 0;
grad = zeros(size(theta));

% Initialize the hypothesis
h = sigmoid(X*theta);

% Compute the grad
l = size(theta);
sum = zeros(l(1),1);
j=1;
while(j<l(1)+1)
    i=1;
    while(i<m+1)
        sum(j)= sum(j) + ((h(i))-y(i))*X(i,j);
        i = i + 1;
    end  
    grad(j) = (1/m)*sum(j);
    j=j+1;
end

% Compute the cost function
csum = 0;
k=1;
while(k<m+1)
    csum = csum + (-y(k)*log(h(k))-(1-y(k))*log(1-h(k)));
    k=k+1;
end   
J = (1/m)*csum;

end
