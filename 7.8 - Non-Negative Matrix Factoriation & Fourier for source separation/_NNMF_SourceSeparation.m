%% Load data
% First, let's load the audio file and compute the spectrogram of it.

[data, fs] = audioread('Mary.wav');

nfft = 1024;

Xfull = myspectrogram(data,nfft,fs,hann(512),-256);
X = abs(Xfull(1:(nfft/2),:));

%% Compute the NNMF
%
% Let's iterate to compute the NNMF of the frequency analysis matrix using
% proximal gradient method.

[n,p] = size(X);

% Params
K = 3;
lambda = 1/5;

% Init
D = 1 + rand(n,K);
A = 1 + rand(K,p);

% Iterate
for i=1:100
    rho = 1/norm(D'*D);
    for j = 1:20
        A = A + rho * D'*(X - D*A);
        A = A - lambda;
        A = A .* (A > 0);
    end
    
    rho = 1/norm(A*A');
    for j = 1:20
        D = D + rho * (X - D*A)*A';
        D = D .* (D > 0);
        % ||d|| < 1
        normw = sqrt(sum(D.^2));
        D = D ./ (ones(n,1)*normw);
    end
    
end

%% Reconstruct original
% Let's reconstruct the original signal.

phi = angle(Xfull);

Xhat = D*A;
Xfullhat = [Xhat;Xhat(end:-1:1, :)];
Xfullhat = Xfullhat.*exp(1i*phi);
datahat = real(invmyspectrogram(Xfullhat, 256));

%% Plot results
% Let's see the result.

subplot(2,3,2);
imagesc(db(X))
title('Spectrum')
xlabel('Time');
ylabel('Frequency');

subplot(2,3,5);
imagesc(db(D*A))
title('Reconstructed spectrum')
xlabel('Time');
ylabel('Frequency');

subplot(2,3,3);
Dplot = D/max(max(D))/1.1 + ones(n,1)*[1 2 3];
plot(Dplot);
xlim([0 200]);
title('D');
xlabel('Frequency (axis cut)');
ylabel('Component');

subplot(2,3,6);
Aplot = A'/max(max(A))/1.1 + ones(p,1)*[1 2 3];
plot(Aplot);
title('A^T');
xlabel('Time');
ylabel('Component');

subplot(2,3,1);
plot(data);
title('Signal');

subplot(2,3,4);
plot(datahat);
title('Reconstructed signal');



