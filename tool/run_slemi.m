function [] = run_slemi()
setenv('COVEXPEXPLORE','D:\github\OTP2ACBD\workspace_for_emi')
setenv('SLSFCORPUS','D:\github\OTP2ACBD\workspace_for_emi')
covexp.covcollect() % 预处理收集到的模型死块的覆盖信息
emi.go() % 进行差分测试
emi.report() % 以表格形式输出变体和差分测试的结果
end