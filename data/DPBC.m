function [rho,delta]=DPBC(Y,Y_index,groundtruth1D,show)

if (nargin<3)
    show=0;
end

[no_bands,no_pixels] = size(Y);

[~,score,latent,tsquare] = pca(Y');
pc = 10;
data = score(:,1:pc);

D = pdist(data, 'euclidean');
dist = squareform(D);

ND=size(dist,2);
N=ND*(ND-1)/2;

percent=2.0;

position=round(N*percent/100);
sda=sort(D);
dc=sda(position);


for i=1:ND
  rho(i)=0.;
end

for i=1:ND-1
  for j=i+1:ND
     rho(i)=rho(i)+exp(-(dist(i,j)/dc)*(dist(i,j)/dc));
     rho(j)=rho(j)+exp(-(dist(i,j)/dc)*(dist(i,j)/dc));
  end
end

maxd=max(max(dist));

[rho_sorted,ordrho]=sort(rho,'descend');
delta(ordrho(1))=-1.;
nneigh(ordrho(1))=0;

for ii=2:ND
   delta(ordrho(ii))=maxd;
   for jj=1:ii-1
     if(dist(ordrho(ii),ordrho(jj))<delta(ordrho(ii)))
        delta(ordrho(ii))=dist(ordrho(ii),ordrho(jj));
        nneigh(ordrho(ii))=ordrho(jj);
     end
   end
end
delta(ordrho(1))=max(delta(:));

rho = (rho-min(rho))./(max(rho)-min(rho));
delta = (delta-min(delta))./(max(delta)-min(delta));

ind = find(delta>=0.005);
rho1 = rho(ind);
delta1 = delta(ind);
gamma1 = rho1.*delta1.^2;
[a,b] = sort(gamma1,'descend');
% 
if show==1    
    figure
    set(gca,'Fontsize',12,'Fontname','times new roman')
    tt=plot(rho(:),delta(:),'o','MarkerSize',5,'MarkerFaceColor','k','MarkerEdgeColor','k');
    title ('Decision Graph','FontSize',12.0)
    xlabel ('\rho')
    ylabel ('\delta')
    
    
end


end