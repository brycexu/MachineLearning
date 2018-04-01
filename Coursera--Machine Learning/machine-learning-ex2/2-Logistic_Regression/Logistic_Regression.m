% Sigmoid Function
% Sigmoid.m

% Normalized Logistic Regression

% Preparations 准备数据
data=load('ex2data1.txt');
X=data(:,[1,2]);
y=data(:,3);
[m,n]=size(X);
X=[ones(m,1),X];
initial_theta=zeros(n+1,1);

% Establish costFunction and gradients 核心
% costFunction.m

% Optimizing using fminunc 使用fminunc来优化数据
options=optimset('GradObj','on', 'MaxIter', 400);
[theta, cost] = ...
	fminunc(@(t)(costFunction(t, X, y)), initial_theta, options);

% Results 结果呈现
fprintf('The optimal theta:\n');
disp(theta);
fprintf('The final linear regression equation:\n')
fprintf('x0+');
for i=1:n-1
fprintf('%.0fx%.0f+',theta(i),i);
end
fprintf('%.0fx%.0f\n',theta(n),n);


% Regularized Logistic Regression

% Preparations 准备数据
data=load('ex2data2.txt');
X=data(:,[1,2]);
y=data(:,3);
initial_theta=zeros(size(X,2),1);
lambda=1;

% Establish costFunctionReg and gradients 核心
% costFunctionReg.m

% Optimizing using fminunc 使用fminunc来优化数据
options = optimset('GradObj', 'on', 'MaxIter', 400);
[theta, J, exit_flag] = ...
	fminunc(@(t)(costFunctionReg(t, X, y, lambda)), initial_theta, options);

% Results 结果呈现
fprintf('The optimal theta:\n');
disp(theta);
