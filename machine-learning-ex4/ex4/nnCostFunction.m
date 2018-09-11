function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
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
a1 = [ones(m, 1) X];   % add a(1)0 5000x401
z2 = a1 * Theta1';     % 5000x401 * 401x25 ==> 5000x25
a2 = sigmoid(z2);
a2 = [ones(size(a2), 1) a2];       % add a(2)0 5000x26
z3 = a2 * Theta2';                 % 5000x26 * 26x10 ==> 5000x10
a3 = sigmoid(z3);                  % h(theta)  5000x10

% Turn labels to vector
labels_0 = eye(num_labels);
y_labels = zeros(1, num_labels);
for i = 1:m
    y_labels = [y_labels; labels_0(y(i), :)];
end
y_labels = y_labels([2: end], :);

% cost without regularization
tmp = -a3 .* y_labels - log(1 - a3).*(1 - y_labels);
J = sum(sum(-log(a3) .* y_labels - log(1 - a3).*(1 - y_labels))) / m;

% Add regularization
J =  J + lambda * (sum(sum(Theta1(:,[2:end]) .^2)) + sum(sum(Theta2(:,[2:end]) .^2))) / 2 / m;


% backpropagation for gradient
D2 = zeros(num_labels, size(hidden_layer_size));
D1 = zeros(size(input_layer_size), hidden_layer_size);
for t = 1:m
%    a1_t = X[1, :]';                     % a1  400x1
%    a1_t = [1; a1];                      % a1  401x1
%    z2_t = Theta1 * a1_t;                % 25x401 * 401x1 ==>25x1
%    a2_t = sigmoid(z2_t);                % 25x1
%    a2_t = [1; a2_t];                    % 26x1
%    z3_t = Theta2 * a2_t;                % 10x26 * 26x1 ==>10x1 
    a1_t = a1(t, :)';					% 401x1
    z2_t = z2(t, :)';					% 25x1
    a2_t = a2(t, :)';					% 26x1
    z3_t = z3(t, :)';					% 10x1
    a3_t = a3(t, :)';					% 10x1

    delta3 = a3_t - y_labels(t, :)';	% 10x1
	% Theta2' * delta3 .* g'(z2)
    delta2 = Theta2(:, [2:end])' * delta3 .* sigmoidGradient(z2_t);   % (10x25)' * 10x1 .* 25x1  ==> 25x1
    Theta2_grad = Theta2_grad + delta3 * (a2_t)';           % 10x1 * 1x26 ==> 10x26
    Theta1_grad = Theta1_grad + delta2 * (a1_t)';           % 25x1 * 1x401 ==> 25x401
end

% Unregularization gradient for theta
% Theta1_grad = Theta1_grad / m;
% Theta2_grad = Theta2_grad / m;

% Add regularization gradient
Theta1_grad = [Theta1_grad(:, 1) Theta1_grad(:, [2:end]) + lambda * Theta1(:, [2:end])] / m;   % [25x1 25x400] ==> 25x401
Theta2_grad = [Theta2_grad(:, 1) Theta2_grad(:, [2:end]) + lambda * Theta2(:, [2:end])] / m;   % [10x1 10x25] ==> 10x26



% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
