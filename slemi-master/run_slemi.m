function [] = run_slemi()
setenv('COVEXPEXPLORE','D:\workspace\simulink\simulink2\test_for_emi')
setenv('SLSFCORPUS','D:\workspace\simulink\simulink2\test_for_emi')
covexp.covcollect() % 预处理收集到的模型死块的覆盖信息
emi.go() % 进行差分测试
end