function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
% NNCOSTFUNCTION Implements the neural network cost function for a two layer
% neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


% Part1
a1=[ones(m,1),X]; % a1:5000*401

z2=a1*Theta1'; % z2:5000*25

a2=sigmoid(z2); % a2:5000*25

m2=size(a2,1);

a2=[ones(m2,1),a2]; % a2:5000*26

z3=a2*Theta2'; % z3:5000*10

a3=sigmoid(z3);

Y=zeros(m,num_labels); % Y:5000*10

for i=1:m
    Y(i,y(i))=1;
end

for i=1:m
    for k=1:num_labels
        J=J+(1/m)*(-Y(i,k)*log(a3(i,k))-(1-Y(i,k))*log(1-a3(i,k)));
    end
end


% Part2
D_delta_2=zeros(size(Theta2));
D_delta_1=zeros(size(Theta1));
for t=1:m

% Step1
a1=[1,X(t,:)]; % a1:1*401 Theta1:25*401
z2=a1*Theta1'; % z2:1*25
a2=sigmoid(z2); % a2:1*25
a2=[1,a2]; % a2:1*26 Theta2:10*26
z3=a2*Theta2'; % z3:1*10
a3=sigmoid(z3);

% Step2
delta_3=a3-Y(t,:); % delta_3:1*10

% Step3
z2=[1,z2]; % z2:1*26
delta_2=(Theta2)'*delta_3'.*sigmoidGradient(z2'); % delta_2:26*1
% 26*10
delta_2=delta_2';

% Step4
delta_2=delta_2(2:end); % delta_2:1*25
D_delta_2=D_delta_2+(1/m)*delta_3'*a2;
D_delta_1=D_delta_1+(1/m)*delta_2'*a1;

end

Theta2_grad=D_delta_2;
Theta1_grad=D_delta_1;


% Part3
L=0;

[s1,t1]=size(Theta1);
for j=1:s1
    for k=2:t1
        L=L+(lambda/(2*m))*Theta1(j,k)^2;
    end
end

[s2,t2]=size(Theta2);
for j=1:s2
    for k=2:t2
        L=L+(lambda/(2*m))*Theta2(j,k)^2;
    end
end

J=J+L;

RegD_delta_1=zeros(size(Theta1));
RegD_delta_2=zeros(size(Theta2));

RegD_delta_1(:,2:end)=RegD_delta_1(:,2:end)+Theta1(:,2:end);
RegD_delta_2(:,2:end)=RegD_delta_2(:,2:end)+Theta2(:,2:end);

D_delta_1=D_delta_1+(lambda/m)*RegD_delta_1;
D_delta_2=D_delta_2+(lambda/m)*RegD_delta_2;

Theta1_grad=D_delta_1;
Theta2_grad=D_delta_2;


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
