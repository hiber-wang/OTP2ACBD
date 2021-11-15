function [] = transition_slx()
%clear;clc;
%Path = './success/';
Path = './';
Dir = dir(strcat([Path,'*.mdl']));
% print(namelist);
disp('文件数：');
disp(length(Dir));
for i = 1:length(Dir)
    open_system(Dir(i).name);
    disp(class(Dir(i).name));
    S = strsplit(Dir(i).name,'.');
    save_system(S(1),strcat(S(1),'.slx'));
    close_system(strcat(S(1),'.slx'));
end 
disp('结束');
end