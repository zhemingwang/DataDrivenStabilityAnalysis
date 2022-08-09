function interface_parc
% % % Identification using piecewise affine models

N = 200;
Y = Ybudget(:,1:N);
Ydata = [];
for i = 1:T-L
    Ydata = [Ydata Y((i-1)*p+1:(i+L)*p,:)];
end

load('group_parc.mat')
rho_jsr_parc = hit_parc(Ydata,group_parc,p);


end

function [rho_jsr_parc,idtime,jsrtime] = hit_parc(Ydata,group_parc,p)
AY = [];

N = length(group_parc);
M_max = max(group_parc);
modes = [];
dim = size(Ydata,1)-p;

tStart = cputime;
for i = 0:M_max
    if length(find(group_parc==i)) > dim*p
        modes = [modes i];
    end
end
M_parc = length(modes);

L = dim/p;
AYcell = cell(1,M_parc);
for i = 1:M_parc
    Yi = Ydata(:,group_parc==modes(i));
    coeff_i = Yi(1+L*p:end,:)/Yi(1:L*p,:);
    AYi = [zeros(p*(L-1),p) eye(p*(L-1));coeff_i];
    AY(:,:,i) = AYi;
    AYcell(i) = {AYi};
end
idtime = cputime-tStart;
tStart = cputime;
[rho_jsr_parc,INFO_parc] = jsr(AYcell);
jsrtime = cputime-tStart;

end
