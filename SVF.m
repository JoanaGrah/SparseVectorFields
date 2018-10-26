%% Image Compression with Sparse Vector Fields Regularisation
% _Eva-Maria Brinkmann, Martin Burger, Joana Sarah Grah_
% 
% "Regularization with sparse vector fields: from image compression to TV-type 
% reconstruction." _International Conference on Scale Space and Variational Methods 
% in Computer Vision_. Springer, Cham, 2015.
% 
% This code performs image compression for a given image by using the SVF 
% model optimised by a primal-dual algorithm.
%% Load Trui test image
%%
load Trui.mat
[m,n] = size(img);
%% Set Parameters
%%
lambda = 10;                                       %weighting parameter (data term)

Dy = spdiags([-ones(m,1) ones(m,1)], [0 1], m, m);
Dy(m,:) = 0;
Dx = spdiags([-ones(n,1) ones(n,1)], [0 1], n, n);
Dx(n,:) = 0;
DX = kron(Dx, speye(m));                           %gradient matrices
DY = kron(speye(n), Dy);
divX = -DX';                                       %divergence matrices
divY = -DY';
lap = divX*DX + divY*DY;                           %Laplace matrix
K = [lap -divX -divY];                             %operator K

algorithmParam.sigma = 1/sqrt(normest(K'*K));      %primal step size
algorithmParam.tau = algorithmParam.sigma;         %dual step size
algorithmParam.theta = 1;                          %primal-dual overrelaxation parameter
algorithmParam.maxIter = 10000;                    %maximum number of primal-dual iterations
algorithmParam.epsilon = 1e-3;                     %primal-dual residual threshold
%% Optimisation
%%
[u, v] = primaldualOptimisation(img, K, algorithmParam, lambda);
%% Visualisation
%%
norm_v = sqrt(v(:,:,1).^2 + v(:,:,2).^2);

figure;
subplot(131);imagesc(img);title('Input Image');axis off;axis image;colormap(gca,'gray');
subplot(132);imagesc(u);title('Compressed Image');axis off;axis image;colormap(gca,'gray');
subplot(133);imagesc(norm_v);title('Norm of Sparse Vector Field');axis off;axis image;colormap(gca,'parula');