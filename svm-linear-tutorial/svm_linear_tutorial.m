clear all; clc;

%% requires that you download libsvm

% open octave or matlab to the matlab subdirectory for libsvm
% for matlab only ensure that it is set to use a c++ compiler ( mex -setup ) 
% for octave and matlab type make

%% add matlab libsvm directory to path 
addpath('~/libsvm-3.17/matlab');

%% load data

[labels,data] = libsvmread('twofeature.txt');


%% visualize data
Xplus = full(data(labels==1,:));
Xminus = full(data(labels==-1,:));

for i = 1:2
	figure(i)
	plot(Xplus(:,1),Xplus(:,2),'.b','markersize',25); 
	hold on;
	plot(Xminus(:,1),Xminus(:,2),'.g','markersize',25);
end

%% SVM train

% -s 0  ( SVM Classification ), 
% -t 0 ( Linear Kernel ), 
% -c 1 ( Cost factor 1)

model = svmtrain(labels,data, '-s 0 -t 0 -c 1');

w = model.SVs' * model.sv_coef;
b = -model.rho;
if (model.Label(1) == -1)
	w = -w; b = -b;
end

%% visualize decision boundary

figure(1);
hold on
xp = linspace(min(min(data)),max(max(data)),100);
yp = - (w(1)*xp + b)/w(2);
plot(xp,yp,'-b');
title('SVM Linear Classifier - C = 1')
hold off

%% model with cost factor higher (C = 100)

model = svmtrain(labels,data, '-s 0 -t 0 -c 100');

w = model.SVs' * model.sv_coef;
b = -model.rho;
if (model.Label(1) == -1)
	w = -w; b = -b;
end

%% visualize decision boundary

figure(2);
hold on;
yp = - (w(1)*xp + b)/w(2);
plot(xp,yp,'-b');
title('SVM Linear Classifier - C = 100')
hold off
