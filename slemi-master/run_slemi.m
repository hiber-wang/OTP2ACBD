function [] = run_slemi()
setenv('COVEXPEXPLORE','D:\github\OTP2ACBD\workspace_for_emi')
setenv('SLSFCORPUS','D:\github\OTP2ACBD\workspace_for_emi')
covexp.covcollect() % 预处理收集到的模型死块的覆盖信息
emi.go() % 进行差分测试
end