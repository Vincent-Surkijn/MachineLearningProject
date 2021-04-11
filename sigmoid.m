function g = sigmoid(z)
%SIGMOID Compute sigmoid function
%   g = SIGMOID(z) computes the sigmoid of z.

l = size(z);
g = zeros(size(z));
j = 1;
while(j < l(1)+1)
    i = 1;
    while(i<l(2)+1)
        g(j,i) = 1/(1 + exp(-z(j,i)));
        i = i + 1;
    end
    j = j + 1;
end

end
