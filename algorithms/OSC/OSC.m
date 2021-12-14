function [ Z, funVal, iteration ] = OSC( X, lambda_1, lambda_2, gamma_1, gamma_2, p, maxIterations, diagconstraint)
%% Problem
%
% min L(Z) = 1/2 ||X - XZ||^2_F - lambda_1 * ||Z||_1 + lambda_2 * ||ZR||_2/1
% 
% where ||B||_2/1 = ||b_1||_2 + ||b_2||_2 + ... + ||b_n||_2
%
%% Solution 
% We solve this problem via the ADMM (Alternating direction method of multipliers) variant
% of Augmented Lagrangian method as follows:
% 
% Let S = Z and U = SR then we have
% 
% T(Z,S,U) = 1/2||X - XS||^2_F + lambda_1 * ||Z||_1 + lambda_2 * ||U||_2/1
% + <G,Z - S> + gamma_1/2 * ||Z - S||^2_F + <F,U - SR>
% + gamma_2/2 * ||U - SR||^2_F
%

if (~exist('diagconstraint','var'))
    diagconstraint = 0;
end

funVal = zeros(maxIterations,1);

[~, xn, ~] = size(X);

S = zeros(xn, xn); % S = Z
R = (triu(ones(xn,xn-1),1) - triu(ones(xn, xn-1))) + (triu(ones(xn, xn-1),-1)-triu(ones(xn, xn-1)));
R = sparse(R);

U = zeros(xn, xn-1);

G = zeros(xn, xn);
F = zeros(xn, xn-1); 

Z = zeros(xn, xn);

kron_Xt_X = kron(speye(xn,xn),(X' * X));
kron_R_Rt = kron(R*R',speye(xn,xn));

for iteration=1:maxIterations

    %% Step 1
    V = S - (G/gamma_1);
    [vm, vn, ~] = size(V);
    rolled_v = reshape(V,vm*vn,1);

    rolled_z = shrink_l1(rolled_v, lambda_1/gamma_1);

    Z = reshape(rolled_z, vm, vn);

    % Set Z diag to 0
    if (diagconstraint)
        Z(logical(eye(size(Z)))) = 0;
    end

    %% Step 2
    left = kron_Xt_X  + kron(speye(xn,xn),gamma_1*speye(xn,xn)) + gamma_2 * kron_R_Rt;

    right = X'*X + gamma_2*U*R' + gamma_1*Z + G + F*R';
    right = reshape(right,xn*xn,1);

    s = left \ right;
    S = reshape(s,xn,xn);

    %% Step 3
    V = S*R - (1/gamma_2)*F;

    U = mysolve_l1l2(V, lambda_2/gamma_2);

    %% Step 4

    G = G + gamma_1 * (Z - S);

    %% Step 5

    F = F + gamma_2 * (U - S*R);

    %% Step 6

    gamma_1 = p * gamma_1;
    gamma_2 = p * gamma_2;

    %% Calculate function values
    funVal(iteration) = .5 * norm(X - X*Z,'fro')^2 + lambda_1*norm(Z,1) +lambda_2*l2l1norm(Z*R);

    if iteration > 1
        if funVal(iteration) < 1*10^-3
            break
        end
    end

    if iteration > 100
        if funVal(iteration) < 1*10^-3 || funVal(iteration-1) == funVal(iteration) ...
                || funVal(iteration-1) - funVal(iteration) < 1*10^-3
            break
        end
    end


end

end