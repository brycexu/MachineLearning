function [J, grad] = costFunctionReg(theta, X, y, lambda)
m = length(y);
J = 0;
grad = zeros(size(theta));

for i=1:m
    J=J+(1/m)*(-y(i)*log(sigmoid(X(i,:)*theta))-(1-y(i))*(log(1-sigmoid(X(i,:)*theta))));
end
for j=2:length(theta)
    J=J+(lambda/(2*m))*(theta(j)^2);
end

for i=1:m
    grad(1)=grad(1)+(1/m)*(sigmoid(X(i,:)*theta)-y(i))*X(i,1);
end
for j=2:length(theta)
    for i=1:m
        grad(j)=grad(j)+(1/m)*(sigmoid(X(i,:)*theta)-y(i))*X(i,j);
    end
    grad(j)=grad(j)+(lambda/m)*theta(j);
end

end
