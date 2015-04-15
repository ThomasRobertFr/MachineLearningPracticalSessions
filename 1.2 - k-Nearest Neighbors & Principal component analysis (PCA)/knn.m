function [ypred,Dist]=knn(xtest,xapp,yapp,k,Dist)


% USAGE
% [ypred,Dist]=knn(xtest,xapp,yapp,k,Dist)
%

% DECISION

classcode=(unique(yapp))';
nbclasse=length(classcode);



nX=size(xtest,1);
nXi=size(xapp,1);
ypred=0*ones(nX,1);
%nbclasse=2;
if nargin < 5 | isempty(Dist) % on a deja calcule la matrice de distance
    for i=1:nX;
        D=0*ones(nXi,1);
        for j=1:nXi;
            
            Dist(i,j)=(xtest(i,:)-xapp(j,:))*(xtest(i,:)-xapp(j,:))';
        end;
    end;
end;

for i=1:nX
    [aux,I]=sort(Dist(i,:));
    C=yapp(I);
    classeppv=C(1:k);
    nc=0*ones(nbclasse,1);
    for j=1:k;
        ind=find(classcode==classeppv(j));
        nc(ind)=nc(ind)+1;    
    end;
    [val,aff]=max(nc);
    ypred(i)=classcode(aff);
end;




