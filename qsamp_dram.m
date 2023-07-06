%function V = poly_dram(x,spec,int)
spec = bayesinit_script;
int = model_init(spec);
% C0 = int.cov; C0(3,3) = 0.77;
% int.cov = C0;
nhyp = prior_nhyp(spec,int);
ndim = int.ndim;

rng(1)
tries = 1; 
%q_com = zeros(tries,1); m_com = zeros(tries,1); mp_com = zeros(tries,1);

%int.cov = (spec.pm * abs(diag(int.mu)) + int.cov).^2;
int.cov = 9 * eye(ndim);
%int.mu = int.mu + 1;

data = int.data;
 t_data = data.xdata;
y_data = data.ydata;

%entr = x_ran(q);    

% C0 = int.cov; C0(3,3) = 0.77;
% int.cov = C0;

x = .5 * ones(nhyp,1);

%x = z;
priorfun = prior_select(spec,int,x);
%%
 burnin = 1e3;
iter = 1e4; 
cutoff = 1e5;

Ad = 100; count = 1;
DR = 3;
sc = 1 ./ ([2 2 2 2]);
sd = 1^2 / ndim;
ep = 1e-8; beta = 1;
%%

%% initialize
data = int.data;
qoifun = int.qoi;


%G =  sd  * int.G / 6;
%G = .1 * ones(6,6) + eye(6);
%G = int.G / 1e9;
G = (.1 * ones(ndim,ndim) + eye(ndim)) / 1e6;
gamma =  chol(G)'; gamma0 = gamma / 1; 
gamma_n = gamma0;
%m_old = int.mu; 
%%
ss= int.ssfun; ssfun = @(th) ss(th);


%Pi_old = rss - 2 * log(priorfun(m_old));
qoi_set = zeros(iter,1);
rng(1)
chain = zeros(iter,ndim);
ss_chain = zeros(iter,1);

 accept = 0; 
target = @(p) (ssfun(p))- 2 * log(priorfun(p));
   options1 = optimoptions(@fminunc,'MaxFunctionEvaluations',10000,'Display','off');
[m_old,Pi_old] = fminunc(target,int.mu,options1);
ss_old = ssfun(m_old);
%mp_com(q) = qoifun(m_old);
%% True Chain
mu = int.mu; V = diag(int.cov);

params = {
%      name,  init,        min, max, mu,  sig, target?, local?
    {'mu', m_old(1),        -Inf,  Inf,  mu(1), V(1)}
    {'beta'    , m_old(2),       -Inf,  Inf,  mu(2), V(2)}
    {'sigma', m_old(3),        -Inf,  Inf,  mu(3), V(3)}
    {'gamma',     m_old(4),       -Inf,  Inf,  mu(4), V(4)}
         };



data = int.data;

options.nsimu = iter;
%model.modelfun   = @(dat,th ) SEIR_dram(exp(th),dat);
model.ssfun = @(th,data) SEIR_ss(exp(th),data) ;
options.waitbar     = 1; 

[results, chain, s2chain]= mcmcrun(model,data,params,options);



%%
%chain(1:1e3,:) = [];
%q = sum(chain(burn:end,:).^2,2);
% V = var(qoi);
% q_com(q) = V;
% m_com(q) = mean(qoi);
accept / (iter)

%end
[acf,lags] = autocorr(chain(:,1),NumLags=100);
% 
 figure;
% plot(chain(:,1))
plot(qoi_set)
set(gca,'FontSize',15)
xlabel('Iteration')
ylabel('R_0')
set(gca,'FontSize',15)

mu_D = int.mu;
cov_D = int.cov;

%% m transform
%M2 = int.mu(2); pm = spec.pm;
%x_com = M2 + (x_com - .5) * pm * M2;  

% 
% figure;
% [f1,xi1] = ksdensity(q_com);
% hold on;
% %plot(xi,f,'linewidth',3)
% plot(xi1,f1,'-.','linewidth',3)
% xlabel('\gamma')
% ylabel('Density')
% set(gca,'FontSize',15)
% %xlim([1.35,2.6])
% %legend('V_{prior}=1','V_{prior}=0.01')
% hold off;
% %  scatter(x_ran,q_com)
% %  set(gca,'FontSize',15)
% %  xlabel('V_{log(\gamma)}')
%  ylabel('F(\xi)')