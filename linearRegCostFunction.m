function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta),1);
J=sum((X*theta-y).*(X*theta-y))/2/m;
for j=1:size(theta,1)
  if j==1
    grad(j)=sum((X*theta-y).*X(:,j))/m;
  else
    grad(j)=sum((X*theta-y).*X(:,j))/m+lambda*theta(j)/m;
  end
end
for i=2:size(theta,1)
  J=J+lambda*(theta(i))^2/2/m;
end


% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%












% =========================================================================

grad = grad(:);

end
