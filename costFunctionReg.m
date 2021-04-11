function [J, grad] = costFunctionReg(theta, X, y, lambda)
% Compute cost and gradient for logistic regression with regularization
% J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
% theta as the parameter for regularized logistic regression and the
% gradient of the cost w.r.t. to the parameters. 

m = length(y); % number of training examples

J = 0;
grad = zeros(size(theta));

% Initialize the hypothesis
h = sigmoid(X*theta);

% Compute the grad
l = size(theta);
sum = zeros(l(1),1);
j=1;  
sumtheta=0;
while(j<l(1)+1)
    n=0;
    if(j>1)
        n=1;
    end    
    i=1;
    sumtheta = sumtheta + n*theta(j)^2;
    while(i<m+1)
        sum(j)= sum(j) + ((h(i))-y(i))*X(i,j) + n*(lambda/m)*theta(j);
        i = i + 1;
    end  
    grad(j) = (1/m)*sum(j);
    j=j+1;
end

% Compute the cost function
csum = 0;
k=1;
while(k<m+1)
    csum = csum + (-y(k)*log(h(k))-(1-y(k))*log(1-h(k))) + (lambda/(2*m))*sumtheta;
    k=k+1;
end   
J = (1/m)*csum;

end
