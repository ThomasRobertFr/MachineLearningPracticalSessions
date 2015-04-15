function [err, varargout]=ASINETcriterion(TARGET,YE,criterion)
%%
% Calcul le critère du modèle
%
% TARGET: cible voulue
% YE: estimation du modèle
% criterion: critère utilisé ('mse')
% 
% err: erreur du modèle
% varargout{1: gradient du critère

nX=size(TARGET,1);

eps=10e-9;

% Calcul du critère
% (ne pas oublier de moyenner par le nombre d'exemples)
switch lower(criterion)
case 'mse'
  err=mean(sum((TARGET-YE).^2));
case 'nll'
  err=0;
  for x=1:nX
    i0=TARGET(x,:)==1;
    err=err-log(YE(x,i0)+eps);
  end
case 'lnll'
    err=mean(-sum(TARGET.*YE,2));
otherwise
      error('Unknown criterion');
end

if nargout>1
  switch lower(criterion)
  case 'mse'
    gradOut = -(TARGET-YE);
  case 'nll'
    n0=size(TARGET,2);
    i0=find(TARGET==1);
    gradOut=zeros(1,n0);
    gradOut(i0)=-1./(YE(i0)+eps);
  case 'lnll'
    n0=size(TARGET,2);
    i0=find(TARGET==1);
    gradOut=zeros(1,n0);
    gradOut(i0)=-YE(i0);
  otherwise
	error('Unknown criterion');
  end
 varargout{1}=gradOut;
end