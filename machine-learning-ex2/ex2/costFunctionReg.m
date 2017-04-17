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

z = X * theta;
hypo = sigmoid(z);
loghypoplus = log (hypo);
loghypominus = log (1 .- hypo);
results = y .* loghypoplus + (1 .- y) .* loghypominus;
result = sum(results);
J = (- 1 / m) * result;
thetaset = theta .^ 2;
thetaset(1) = 0;
J = J + (lambda / (2 * m)) * sum(thetaset);

difference = hypo - y;
grad = (1 / m) * (difference' * X)'
gradfirst = grad(1); 
grad = grad + (lambda / m) * theta;
grad(1) = gradfirst;




% =============================================================

end
