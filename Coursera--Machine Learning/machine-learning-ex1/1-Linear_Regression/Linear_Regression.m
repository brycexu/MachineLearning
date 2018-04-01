% X(加上第一列的1后):储存特征的向量(m*n)=(样本数*特征数）
% y:样本值
data=load('ex1data2.txt');
m=size(data,1);  %样本数
n=size(data,2);  %特征数
% fprintf('The orginal X:\n');
X=data(:,1:n-1);
% disp(X);
% fprintf('The orginal y:\n');
y=data(:,n);
% disp(y);

% Solving with Gradient Descent  第一种方法：梯度下降

fprintf('Solving with Gradient Descent:\n');

% featureNormalize 将X标准化

mu=mean(X);
X_norm=X-mu;
sigma=std(X_norm);
X_norm=X_norm./sigma;
X_norm=[ones(m,1),X_norm];  %在前面增加一个全是1的列
                            
% fprintf('The normalized X:\n');
% disp(X);

% gradient descent

alpha=0.01;
num_iters=400;
theta=zeros(n,1);
J_history=zeros(num_iters,1);
X=[ones(m,1),X];
for iter=1:num_iters
    theta_temp=theta;
    for j=1:length(theta)
        theta_temp(j)=theta(j)-alpha*(1/m)*(X_norm*theta-y)'*X_norm(:,j);
    end
    theta=theta_temp;

    % compute cost
    
    predictions=X*theta;
    Errors=(predictions-y).^2;
    J_history(iter)=(1/(2*m))*sum(Errors);

end

fprintf('The optimal theta:\n');
disp(theta);
fprintf('The final linear regression equation:\n')
fprintf('x0+');
for i=1:n-1
fprintf('%.0fx%.0f+',theta(i),i);
end
fprintf('%.0fx%.0f\n',theta(n),n);

    
% Solving with Normal Equation  第二种方法：一般方程

fprintf('Solving with Normal Equation:\n');

% normal equation

X=data(:,1:n-1);
y=data(:,n);
X=[ones(m,1),X];
theta=zeros(size(X,2),1);
theta=pinv(X'*X)*X'*y;

    % compute cost
    
    predictions=X*theta;
    Errors=(predictions-y).^2;
    J=(1/2*m)*sum(Errors);

fprintf('The optimal theta:\n');
disp(theta);
fprintf('The final linear regression equation:\n')
fprintf('x0+');
for i=1:n-1
fprintf('%.0fx%.0f+',theta(i),i);
end
fprintf('%.0fx%.0f\n',theta(n),n);


fprintf('The comparison between Gradient Descent and Normal Equation:\n');
fprintf('The cost for Gradient Descent:%.0f\n',J_history(n));
fprintf('The cost for Normal Equation:%.0f\n',J);








                            
                            
                            
