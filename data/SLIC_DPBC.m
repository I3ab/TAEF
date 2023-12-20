function [cluster,index,rho_local,delta_local]=SLIC_DPBC(data,groundtruth,N,show,W)

if (nargin<5)
    W=8e-3;
end

[R,C,B]=size(data);
Y=D3_to_2D(data,1);
n=R*C;

N2=N*N;
cluster0(:,1)=1:N2;
SR=round(0.5*R/N);
Rows=SR:2*SR:R;
CR=round(0.5*C/N);
Cols=CR:2*CR:C;

cluster0(:,2)=reshape(repmat(Rows,N,1),N2,1);
cluster0(:,3)=reshape(repmat(Cols',1,N),N2,1);
cluster=cluster0;

index_d=inf*ones(R,C,2);
iter=0;
maxiter=30;

while iter<maxiter
    iter=iter+1;
    for k=1:N2
        i0=cluster(k,2);
        j0=cluster(k,3);
        ck=data(i0,j0,:);
        for i=max(1,i0-2*SR):min(i0+2*SR,R)
            for j=max(1,j0-2*CR):min(j0+2*CR,C)
                yij=data(i,j,:);    
                dijN=norm(reshape(yij-ck,B,1),2)^2+W*norm([i-i0,j-j0],2)^2;
                if dijN<index_d(i,j,2)
                    index_d(i,j,1)=k;
                    index_d(i,j,2)=dijN;
                end
            end
        end
    end
    

    index=index_d(:,:,1);
    for k=1:N2
        [Rs,Cs]=find(index==k);
        ck=zeros(B,1);
        for i=1:size(Rs,1)
            ck=ck+reshape(data(Rs(i),Cs(i),:),B,1);
        end
        ck_mean=ck/size(Rs,1);        
        dk=inf;
        for i=1:size(Rs,1)
            dki=norm(reshape(data(Rs(i),Cs(i),:),B,1)-ck_mean,2);
            if dki<dk
                cluster(k,2)=Rs(i);
                cluster(k,3)=Cs(i);
                dk=dki;
            end
        end       
    end
    
    if norm(cluster0-cluster,2)==0
        break
    end
    cluster0=cluster;
end

if show==1
    figure
    imshow(index/(N*N))
end


GT=groundtruth(:);
rho_local=zeros(1,n);
delta_local=zeros(1,n);
for i=1:N*N
    indexi=index;
    indexi(index~=i)=0;
    indexi(index==i)=1;
    
    indexi_1D=reshape(indexi,1,n);
    indexi_1D_index=find(indexi_1D==1);
    
    Y_SLICi=Y(:,indexi_1D_index);
    if size(Y_SLICi,2)<10
        continue
    end
    groundtruth1D=GT(indexi_1D_index');
    [rhoi,deltai]=DPBC(Y_SLICi,indexi_1D_index',GT,show);
    rho_local(indexi_1D_index)=rhoi;
    delta_local(indexi_1D_index)=deltai;
    
end
end
        
            
    
    












