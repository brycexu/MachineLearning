function [J, grad] = costFunction(theta, X, y)
m = length(y);
J = 0;
grad = zeros(size(theta));

for i=1:m
    J=J+(1/m)*(-y(i)*log(sigmoid(X(i,:)*theta))-(1-y(i))*(log(1-sigmoid(X(i,:)*theta))));
end

for j=1:length(theta)
    for i=1:m
        grad(j)=grad(j)+(1/m)*(sigmoid(X(i,:)*theta)-y(i))*X(i,j);
    end
end

end
