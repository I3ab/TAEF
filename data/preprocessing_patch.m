clear
clc
tic
%% 
load pavia.mat
% load MUUFL.mat

show=0;
N=3;
[cluster,index,rho_local,delta_local]=SLIC_DPBC(data,groundtruth,N,show);

clearvars -except data groundtruth rho_local delta_local index

[row,col,band]=size(data);
pixels=size(data,1)*size(data,2);

thresh=100;
zonghe1D=(delta_local./rho_local)';
zonghe1D_index=[zonghe1D,[1:pixels]'];
zonghe1D_index_des=sortrows(zonghe1D_index,1,'descend');

pre_ano_num=length(find(zonghe1D_index_des(:,1)>thresh));

datause=data;
for i=1:pre_ano_num
    indexi=zonghe1D_index_des(i,2);
    if mod(indexi,row)==0
        row_index=row;
        col_index=floor(indexi/row);
    else
        row_index=mod(indexi,row);
        col_index=floor(indexi/row)+1;
    end
    
    clusteri=index(row_index,col_index);
    indexi=index;
    indexi(index~=clusteri)=0;
    indexi(index==clusteri)=1;
    
    indexi_1D=reshape(indexi,1,pixels);
    indexi_1D_index=find(indexi_1D==1);
    
    zonghe_i=[zonghe1D(indexi_1D_index),indexi_1D_index'];
    zonghe_i_des=sortrows(zonghe_i,1,'descend');
    
    background_choosed_index= zonghe_i_des(length(find(zonghe_i_des(:,1)>=thresh))+unidrnd(10),2);
    
    if mod(background_choosed_index,row)==0
        row_index2=row;
        col_index2=floor(background_choosed_index/row);
    else
        row_index2=mod(background_choosed_index,row);
        col_index2=floor(background_choosed_index/row)+1;
    end
    
    datause(row_index,col_index,:)=datause(row_index2,col_index2,:);    
    
    
end

%%
clearvars -except data groundtruth datause

win=10;
step=5;

[h,w,b]=size(data);
num_windows_h=(floor(h/step)-1);
num_windows_w=(floor(w/step)-1);

data_train=zeros(num_windows_h*num_windows_w,win^2,b);
data_test=zeros(num_windows_h*num_windows_w,win^2,b);


for i=1:num_windows_h
    for j=1:num_windows_w
        windowij=datause(step*(i-1)+1:step*(i-1)+10,step*(j-1)+1:step*(j-1)+10,:);
        windowij_2D=D3_to_2D(windowij,1);

        
        data_train(num_windows_w*(i-1)+j,:,:)=windowij_2D';
        
        
        windowij_test=data(step*(i-1)+1:step*(i-1)+10,step*(j-1)+1:step*(j-1)+10,:);
        windowij_2D_test=D3_to_2D(windowij_test,1);
        
        data_test(num_windows_w*(i-1)+j,:,:)=windowij_2D_test';
    end
end

time=toc;

save pavia_split.mat time win step groundtruth data data_train data_test
% save MUUFL_split.mat time win step groundtruth data data_train data_test




