function comparison_with_hybrid_identification

% Dependancies:
%         The JSR toolbox, https://nl.mathworks.com/matlabcentral/fileexchange/33202-the-jsr-toolbox
%         CVX
%         GPCA, SSC, SSOMP, EnSc, http://vision.jhu.edu/code/


% % % % system matrices

n = 3;
M = 3;
p = 2;

A(:,:,1) = [-0.6439   -0.3901   -0.2086;
    -0.9682    0.9493    0.2190;
    -0.1140    0.4895    0.2440];
A(:,:,2) = [-0.8866    0.5199   -0.0464;
    0.5097    0.2914    0.3825;
    0.2710    0.8910    0.3508];
A(:,:,3) = [0.6584    0.8439   -0.8955;
    0.5302   -0.6769    0.7137;
    -0.0617    0.7342   -0.8789];


C(:,:,1) = [-0.2982    0.5891    0.2737;
    -0.4818   -0.8210    0.1567];
C(:,:,2) = [-0.2982    0.5891    0.2737;
    -0.4818   -0.8210    0.1567];
C(:,:,3) = [-0.2982    0.5891    0.2737;
    -0.4818   -0.8210    0.1567];


% % % % True jsr
Acell = cell(1,M);
for i = 1:M
    Acell(i) = {A(:,:,i)};
end
[rho_jsr,INFO] = jsr(Acell);


% % % % Data generation
T = 5;
L = n-1;

% % % To reproduce the same results, load('Ybudget.mat')

traj_num = 2000;
Ybudget = [];
for iter = 1:traj_num
    x0 = randn(n,1);
    x0 = x0/norm(x0);
    x = x0;
    sw = randi([1,M],[1,T]);
    y = C(:,:,sw(1))*x0;
    for t = 1:T-1
        x = A(:,:,sw(t))*x;
        y = [y;C(:,:,sw(t+1))*x];
    end
    Ybudget = [Ybudget y];
end


% % % % Start the simulation
rho_jsr_data_vec = [];
rho_jsr_gpca_vec = [];
rho_jsr_ssc_vec = [];
rho_jsr_ssc_omp_vec = [];
rho_jsr_ssc_ensc_vec = [];

for N = 100:100:traj_num
    
    Y = Ybudget(:,1:N);
    
    % % % % data-driven stability analysis
    [rho_jsr_data,timejsr] = stability_analysis_scenario(Y,p,L,T);
    
    rho_jsr_data_vec = [rho_jsr_data_vec rho_jsr_data];
    
    % % % % Identification data set
    Ydata = [];
    for i = 1:T-L
        Ydata = [Ydata Y((i-1)*p+1:(i+L)*p,:)];
    end
    
    % % % Identification using GPCA
    M_gpca = M^L;
    [rho_jsr_gpca,idtime_gpca,jsrtime_gpca] = hit_gpca(Ydata,M_gpca,p);
    
    rho_jsr_gpca_vec = [rho_jsr_gpca_vec rho_jsr_gpca(2)];
    
    % % % Identification using sparse subspace clustering
    M_ssc = M^L;
    [rho_jsr_ssc,idtime_ssc,jsrtime_ssc] = hit_ssc(Ydata,M_ssc,p);
    [rho_jsr_ssc_omp,idtime_ssc_omp,jsrtime_ssc_omp] = hit_ssc_omp(Ydata,M_ssc,p);
    [rho_jsr_ssc_ensc,idtime_ssc_ensc,jsrtime_ssc_ensc] = hit_ssc_ensc(Ydata,M_ssc,p);
    
    rho_jsr_ssc_vec = [rho_jsr_ssc_vec rho_jsr_ssc(2)];
    rho_jsr_ssc_omp_vec = [rho_jsr_ssc_omp_vec rho_jsr_ssc_omp(2)];
    rho_jsr_ssc_ensc_vec = [rho_jsr_ssc_ensc_vec rho_jsr_ssc_ensc(2)];
    
end

figure
hold on
plot(100:100:2000,rho_jsr_data_vec)
plot(100:100:2000,rho_jsr_gpca_vec)
plot(100:100:2000,rho_jsr_ssc_vec)
plot(100:100:2000,rho_jsr_ssc_omp_vec)
plot(100:100:2000,rho_jsr_ssc_ensc_vec)




end


function [rho_jsr_data,jsrtime] = stability_analysis_scenario(Y,p,L,T)
N = size(Y,2);

X0 = Y(1:p*L,:);
X1 = Y((T-L)*p+1:end,:);

n = p*L;

eta_u = 100;
eta_l = 0;
dis = eta_u-eta_l;
tStart = cputime;
while dis > 1e-2
    eta = (eta_u+eta_l)/2;
    cvx_solver mosek
    cvx_begin sdp quiet
    variable P(n,n) symmetric
    minimize(trace(P))
    P >= eye(n);
    for i = 1:N
        X1(:,i)'*P*X1(:,i)-eta^2*X0(:,i)'*P*X0(:,i) <= 0;
    end
    sum_square_abs(vec(P)) <= 1e6;
    cvx_end
    if strcmp(cvx_status,'Solved')
        eta_u = (eta_u+eta_l)/2;
    else
        eta_l = (eta_u+eta_l)/2;
    end
    dis = eta_u-eta_l;
    disp(['Upper bound: ', num2str(eta_u),...
        ' Lower bound: ', num2str(eta_l),...
        ' Gap:',num2str(dis)])
end

jsrtime = cputime - tStart;

% cvx_solver mosek
% cvx_begin sdp quiet
% variable P(n,n) symmetric
% minimize(sum_square_abs(vec(P)))
%     P >= eye(n);
%     for i = 1:N
%         X1(:,i)'*P*X1(:,i)-eta_u^2*X0(:,i)'*P*X0(:,i) <= 0;
%     end
% %     sum(sum_square_abs(P)) <= 1e20;
% cvx_end

rho_jsr_data = eta_u^(1/(T-L));

end


function [rho_jsr_gpca,idtime,jsrtime] = hit_gpca(Ydata,M_gpca,p)

addpath('CodeGPCAPDASpectral');
addpath('CodeGPCAPDASpectral/helper_functions');

N = size(Ydata,2);
L = size(Ydata,1)/p-1;
Adj = zeros(N);
tStart = cputime;
for i = 1:p
    Yi = [Ydata(1:L*p,:);Ydata(i+L*p,:)];
    groups_Yi=gpca_pda_spectralcluster(Yi,M_gpca);
    Adj_i = zeros(N);
    for j = 1:N
        g_num_j = groups_Yi(j);
        Adj_i(j,groups_Yi==g_num_j) = 1;
        %         Adj_i(j,j) = 0;
    end
    Adj = Adj + Adj_i;
end
Adj = Adj >= 1;

closedset = [];
% Adj = Adj + eye(N);
AY = [];
AYindex = 1;
for i = 1:N
    if ~isempty(closedset)
        if sum(closedset == i) == 0
            Yi = Ydata(:,Adj(i,:) == 1);
            if size(Yi,2) > L*p
                coeff_i = Yi(1+L*p:end,:)/Yi(1:L*p,:);
                AYi = [zeros(p*(L-1),p) eye(p*(L-1));coeff_i];
                AY(:,:,AYindex) = AYi;
                closedset = [closedset find(Adj(i,:) == 1)];
                AYindex = AYindex + 1;
            end
        end
    else
        Yi = Ydata(:,Adj(i,:) == 1);
        if size(Yi,2) > L*p
            coeff_i = Yi(1+L*p:end,:)/Yi(1:L*p,:);
            AYi = [zeros(p*(L-1),p) eye(p*(L-1));coeff_i];
            AY(:,:,AYindex) = AYi;
            closedset = [closedset find(Adj(i,:) == 1)];
            AYindex = AYindex + 1;
        end
    end
    
end
idtime = cputime-tStart;
AYcell = cell(1,AYindex-1);
for i = 1:AYindex-1
    AYcell(i) = {AY(:,:,i)};
end

% AY = [];
% AYcell = cell(1,M_gpca);
% for i = 1:M_gpca
%     Yi = Y(:,group_gpca==i);
%     coeff_i = Yi(1+L*p:end,:)/Yi(1:L*p,:);
%     AYi = [zeros(p*(L-1),p) eye(p*(L-1));coeff_i];
%     AY(:,:,i) = AYi;
%     AYcell(i) = {AYi};
% end
tStart = cputime;
[rho_jsr_gpca,INFO_gpca] = jsr(AYcell);
jsrtime = cputime-tStart;
rmpath('CodeGPCAPDASpectral');
rmpath('CodeGPCAPDASpectral/helper_functions');
end


function [rho_jsr_ssc,idtime,jsrtime] = hit_ssc(Ydata,M_ssc,p)

addpath('SSC_1.0');

Cst = 0; %Enter 1 to use the additional affine constraint sum(c) == 1
OptM = 'Lasso'; %OptM can be {'L1Perfect','L1Noise','Lasso','L1ED'}
lambda = 0.001; %Regularization parameter in 'Lasso' or the noise level for 'L1Noise'
K = 0;
L = size(Ydata,1)/p-1;


tStart = cputime;
CMat = SparseCoefRecovery(Ydata,Cst,OptM,lambda);
CKSym = BuildAdjacency(CMat,K);
group_ssc = SpectralClustering(CKSym,M_ssc);

AY = [];
AYcell = cell(1,M_ssc);
for i = 1:M_ssc
    Yi = Ydata(:,group_ssc==i);
    coeff_i = Yi(1+L*p:end,:)/Yi(1:L*p,:);
    AYi = [zeros(p*(L-1),p) eye(p*(L-1));coeff_i];
    AY(:,:,i) = AYi;
    AYcell(i) = {AYi};
end
idtime = cputime-tStart;
tStart = cputime;
[rho_jsr_ssc,INFO_ssc] = jsr(AYcell);
jsrtime = cputime-tStart;
rmpath('SSC_1.0');

end

function [rho_jsr_ssc,idtime,jsrtime] = hit_ssc_ADMM(Ydata,M_ssc)


addpath('SSC_ADMM_v1.1');

affine = 0; %Enter 1 to use the additional affine constraint sum(c) == 1
% OptM = 'Lasso'; %OptM can be {'L1Perfect','L1Noise','Lasso','L1ED'}
% lambda = 0.001; %Regularization parameter in 'Lasso' or the noise level for 'L1Noise'
K = 0;

tStart = cputime;
CMat = admmLasso_mat_func(Ydata,affine);
CKSym = BuildAdjacency(CMat,K);
group_ssc = SpectralClustering(CKSym,M_ssc);

AY = [];
AYcell = cell(1,M_ssc);
for i = 1:M_ssc
    Yi = Ydata(:,group_ssc==i);
    coeff_i = Yi(1+L*p:end,:)/Yi(1:L*p,:);
    AYi = [zeros(p*(L-1),p) eye(p*(L-1));coeff_i];
    AY(:,:,i) = AYi;
    AYcell(i) = {AYi};
end
idtime = cputime-tStart;
tStart = cputime;
[rho_jsr_ssc,INFO_ssc] = jsr(AYcell);
jsrtime = cputime-tStart;
rmpath('SSC_ADMM_v1.1');

end


function [rho_jsr_ssc,idtime,jsrtime] = hit_ssc_omp(Ydata,M_ssc,p)

addpath('SSCOMP_Code');
N = size(Ydata,2);
L = size(Ydata,1)/p-1;

tStart = cputime;
genLabel = @(affinity, nCluster) SpectralClustering(affinity, nCluster, 'Eig_Solver', 'eigs');

% Ydatanorm = normalizeColumn(Ydata);
% generate representation
%     fprintf('Representation...\n')
R = OMP_mat_func(Ydata, 5, 1e-6);
% generate affinity
%     fprintf('Affinity...\n')
R(1:N+1:end) = 0;
% R = cnormalize(R, Inf);
A = abs(R) + abs(R)';
% generate label
%     fprintf('Generate label...\n')
group_ssc = genLabel(A, M_ssc);

AY = [];
AYcell = cell(1,M_ssc);
for i = 1:M_ssc
    Yi = Ydata(:,group_ssc==i);
    coeff_i = Yi(1+L*p:end,:)/Yi(1:L*p,:);
    AYi = [zeros(p*(L-1),p) eye(p*(L-1));coeff_i];
    AY(:,:,i) = AYi;
    AYcell(i) = {AYi};
end
idtime = cputime - tStart;
tStart = cputime;
[rho_jsr_ssc,INFO_ssc] = jsr(AYcell);
jsrtime = cputime-tStart;
rmpath('SSCOMP_Code');

end



function [rho_jsr_ssc,idtime,jsrtime] = hit_ssc_ensc(Ydata,M_ssc,p)

addpath('EnSC_Code');
addpath('EnSC_Code/Tools');

N = size(Ydata,2);
L = size(Ydata,1)/p-1;


nu0 = 20;
lambda = 0.5;

tStart = cputime;
Ydatanorm = cnormalize_inplace(Ydata);
EN_solver =  @(X, y, lambda, nu) rfss( X, y, lambda / nu, (1-lambda) / nu );
R = ORGEN_mat_func(Ydatanorm, EN_solver, 'nu0', nu0, 'nu_method', 'nonzero', 'lambda', lambda, ...
    'Nsample', N, 'maxiter', 100, 'outflag', true);

R(1:N+1:end) = 0;
A = abs(R) + abs(R)';
group_ssc = SpectralClustering(A, M_ssc, 'Eig_Solver', 'eigs');

AY = [];
AYcell = cell(1,M_ssc);
for i = 1:M_ssc
    Yi = Ydata(:,group_ssc==i);
    coeff_i = Yi(1+L*p:end,:)/Yi(1:L*p,:);
    AYi = [zeros(p*(L-1),p) eye(p*(L-1));coeff_i];
    AY(:,:,i) = AYi;
    AYcell(i) = {AYi};
end
idtime = cputime-tStart;
tStart = cputime;
[rho_jsr_ssc,INFO_ssc] = jsr(AYcell);
jsrtime = cputime-tStart;
rmpath('EnSC_Code');
rmpath('EnSC_Code/Tools');


end





