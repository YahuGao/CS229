function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

h_theta = sigmoid(X * theta);
J = sum(-y .* log(h_theta) - (1 - y) .* log(1 - h_theta))/m + lambda * sum(theta([2:end], 1) .^ 2) / 2 / m;

delta = h_theta - y;

% theta[2-end] penalized
grad = (X' * delta)/m + lambda / m * theta;
% theta0 by x0 = 1 and shouldn't be penalized
grad(1, :) = sum(h_theta - y)/m;

% =============================================================

end
