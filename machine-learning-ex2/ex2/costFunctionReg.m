function [J, grad] = costFunctionReg(theta, X, y, lambda)

%   COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

%   Initialize some useful values
        m = length(y); % number of training examples

%   You need to return the following variables correctly 
        J = 0;
        grad = zeros(size(theta));

%   ====================== YOUR CODE HERE ======================
%   Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

        sigterm = X*theta;
        hx = sigmoid(sigterm);
        
%        term = (-y'*log(hx)-(1-y)'*log(1-hx));
%        There is a problem in my cost function regularizatioin part. Focus
%        on it and find the problem
%        J = sum(term)/m + ((lambda/2*m)*sum(theta(2:length(theta)).^2));
       
        H = sigmoid(X*theta);
        T = y.*log(H) + (1 - y).*log(1 - H);
        J = -1/m*sum(T) + lambda/(2*m)*sum(theta(2:end).^2);

        gradterm = hx-y;

        grad(1) = sum(gradterm'*X(:,1))/m;
        
        for i=2:size(grad)
            grad(i) = sum(gradterm'*X(:,i))/m + (lambda/m)*theta(i);
        end


%   =============================================================

end
