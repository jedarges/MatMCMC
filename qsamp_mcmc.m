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
 burnin = 1e2;
iter = 1e5; 
cutoff = 1e5;

Ad = 100; count = 1;
DR = 3;
sc = 1 ./ ([2 2 2 2]);
sd = 2.3^2 / ndim;
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
   options = optimoptions(@fminunc,'MaxFunctionEvaluations',10000,'Display','off');
[m_old,Pi_old] = fminunc(target,int.mu,options);
ss_old = ssfun(m_old);
%mp_com(q) = qoifun(m_old);
%% True Chain

for tot = 1:iter
    %p = beta * (m_old+gamma*randn(ndim,1)) + (1-beta)*(m_old+gamma_n*randn(ndim,1));
    p = m_old + (beta*gamma + (1-beta)*gamma_n) * randn(ndim,1);
 
    ss = ssfun(p); 

    Pi_new = ss - 2 * log(priorfun(p));
  
    alpha = Pi_new - Pi_old;
  
    rando = rand;
    if alpha < -2*log(rando)
        Pi_old = Pi_new;
        m_old = p;
        accept = accept + 1;
        ss_old = ss;
    else
        N = 1; Y_list = p; Pi_list = Pi_new;
        for k = 1:DR

            Yi = beta * (Y_list(k)+sc(k)*gamma*randn(ndim,1)) + (1-beta) * (Y_list(k)+sc(k)*gamma_n*randn(ndim,1));
            Yi = Y_list(:,k) + sc(k)*(beta*gamma + (1-beta)*gamma_n) * randn(ndim,1);
            Y_list = [Y_list Yi];
            ss = ssfun(Yi); Pi_new = ss - 2 * log(priorfun(Yi));
            Pi_list = [Pi_list Pi_new];
            if Pi_list(k+1) < Pi_old
                Pi_old = Pi_list(k+1);
                m_old = Yi; accept = accept + 1;
                ss_old = ss;
                break
            elseif Pi_list(k+1) > Pi_old
                break
            else
                Pi_st = min(Pi_list);
                pb = (exp(-.5 * Pi_list(k+1)) - exp(-.5 * Pi_st)) / (exp(-.5 * Pi_old) - exp(-.5 * Pi_st));
                rando = rand;
                if pb > rando
                    Pi_old = Pi_list(k+1);
                    accept = accept + 1;
                    ss_old = ss;
                    m_old = Yi; break
                end
            end
        end
     end
    
    chain(tot,:) = m_old;
    ss_chain(tot) = ss_old;
%    qoi_set(tot) = qoifun(m_old);

    if mod(tot,Ad) == 0 && tot > burnin && tot < cutoff

        batch = count * Ad;
        beta = 1 / ( batch);
       Sign = sd * cov(chain(1:tot,:)) + sqrt(eps) * eye(ndim);

         gamma_n = chol(Sign)';

        count = count + 1;
    end
end

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
 plot(chain(:,1))
%plot(qoi_set)
set(gca,'FontSize',15)
xlabel('Iteration')
%ylabel('R_0')
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