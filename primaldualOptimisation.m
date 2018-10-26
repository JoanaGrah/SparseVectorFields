function [u, v] = primaldualOptimisation(f, K, p, lambda)
%% PRIMALDUALOPTIMISATION_LIVE Image Compression with Sparse Vector Fields Regularisation
% _Eva-Maria Brinkmann, Martin Burger, Joana Sarah Grah_
% 
% "Regularization with sparse vector fields: from image compression to TV-type 
% reconstruction." _International Conference on Scale Space and Variational Methods 
% in Computer Vision_. Springer, Cham, 2015.
% 
% This function performs primal-dual optimisation [1,2] for SVF image compression 
% and calculates a solution of the following constrained minimisation problem:
% 
% $$\min_{u,v} \frac{\lambda}{2} \Vert u - f \Vert_2^2 + \Vert v \Vert_1\ 
% \text{subject to}\ \nabla \cdot ( \nabla u - v ) = 0$$
% 
% Adapting the notation in [1], we have: $x = (u,v)^T$, $K = (\Delta, -\nabla 
% \cdot)$, $G(x) = \frac{\lambda}{2} \Vert u - f \Vert_2^2 + \Vert v \Vert_1$, 
% $F(Kx) = \chi_0(\Delta u - \nabla \cdot v)$.
% 
% References
% 
% # Antonin Chambolle and Thomas Pock. "A first-order primal-dual algorithm 
% for convex problems with applications to imaging." _Journal of mathematical 
% imaging and vision_ 40.1 (2011): 120-145.
% # Tom Goldstein, Min Li, Xiaoming Yuan, Ernie Esser and Richard Baraniuk. 
% "Adaptive primal-dual hybrid gradient methods for saddle-point problems." _arXiv 
% preprint arXiv:1305.0546_ (2015).
%% Preallocation
[m,n] = size(f);
f = f(:);
u = zeros(m*n,1);
u_old = u;
v1 = zeros(m*n,1);
v1_old = v1;
v2 = zeros(m*n,1);
v2_old = v2;
x_old = [u_old; v1_old; v2_old];
y_old = zeros(m*n,1);
Kx_old = zeros(m*n,1);
Kx_bar = zeros(m*n,1);
Kadjointy_old = zeros(3*m*n,1);
iter = 0;
stopCrit = 0;
%% Optimisation
while iter < p.maxIter && stopCrit == 0
    
    iter = iter + 1;
%% 
% Update (1): $y = (I + \sigma \partial F^*)^{-1} (y_{\text{old}} + \sigma 
% K\bar{x})$
    y = y_old + p.sigma * Kx_bar;
%% 
% Update (2): $x = (I + \tau \partial G)^{-1} (x_{\text{old}} - \tau K^*y)$
    Kadjointy = K' * y;
    
    u_tilde = u_old - p.tau * Kadjointy(1:m*n,1);
    u = (p.tau * lambda * f + u_tilde)/(1 + (p.tau * lambda));
    
    v1_tilde = v1_old - p.tau * Kadjointy(m*n+1:2*m*n,1); 
    v2_tilde = v2_old - p.tau * Kadjointy(2*m*n+1:3*m*n,1);
    norm_v_tilde = sqrt(v1_tilde.^2 + v2_tilde.^2);
    norm_v_tilde(norm_v_tilde < 1e-8) = 1e-8;
    v1 = max(norm_v_tilde - p.tau, 0) .* (v1_tilde ./ norm_v_tilde);
    v2 = max(norm_v_tilde - p.tau, 0) .* (v2_tilde ./ norm_v_tilde);
    
    x = [u; v1; v2];
%% 
% Update (3): $\bar{x} = x + \theta (x - x_{\text{old}})$
% 
% Included in calculation of $K\bar{x}$
    Kx = K * x;
    Kx_bar = (1 + p.theta) * Kx - p.theta * Kx_old;
%% 
% Stopping Criterion: Primal-Dual-Residual
    if mod(iter,250) == 0
%% 
% Primal Residual: $\frac{x_{\text{old}} - x}{\tau} - (K^* y_{\text{old}} 
% - K^* y )$
        primalRes = (x_old - x)/p.tau - (Kadjointy_old - Kadjointy);    
%% 
% Dual Residual: $\frac{y_{\text{old}} - y}{\sigma} - (K x_{\text{old}} 
% - Kx)$
        dualRes = (y_old - y)/p.sigma - (Kx_old - Kx);   
%% 
% Primal-Dual Residual: $\Vert \text{primalRes} \Vert_1 + \Vert \text{dualRes} 
% \Vert_1$ (scaled)
        pdRes = norm(primalRes,1)/numel(x) + norm(dualRes,1)/numel(y);   
%% 
% Check Stopping Criterion
        disp(['Iteration ',num2str(iter),'. Primal-Dual Residual: ',num2str(pdRes)]);
        if pdRes < p.epsilon
            stopCrit = 1;
        end   
    end
%% 
% Update variables
    u_old = u;
    v1_old = v1;
    v2_old = v2;
    x_old = [u_old; v1_old; v2_old];
    y_old = y;
    Kx_old = Kx;
    Kadjointy_old = Kadjointy;
end
disp(['Total number of iterations: ',num2str(iter),'.']);
%% Return optimal u and v
u = reshape(u, [m,n]);
v = cat(3, reshape(v1, [m,n]), reshape(v2, [m,n]));
end
%%