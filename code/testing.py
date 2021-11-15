import matlab
import matlab.engine
import io
import os
import shutil
import path


def detect_crash_with_1_testcase(model_path=None):
    if model_path is None or '.mdl' not in model_path:
        print('model_path is illegal')
    engine = matlab.engine.start_matlab()
    out = io.StringIO()
    err = io.StringIO()

    try:
        engine.sim(model_path, stdout=out, stderr=err)
        engine.exit(stdout=out, stderr=err)
    except:
        return 1
    else:
        return 0


def detect_emi(filename=None, file_path=None, folder_path='../workspace_for_emi',
               eng_path=path.ENG_PATH,
               emi_path=path.EMI_PATH):
    old_path = file_path + '/' + filename
    new_path = folder_path + '/' + filename
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
    os.mkdir(folder_path)
    shutil.copy(old_path, new_path)
    shutil.copy('../tool/transition_slx.m', folder_path + '/transition_slx.m')

    eng = matlab.engine.start_matlab()
    out = io.StringIO()
    err = io.StringIO()
    eng.cd(eng_path)
    eng.transition_slx(nargout=0, stdout=out, stderr=err)
    eng.cd(emi_path)
    eng.run_slemi(nargout=0, stdout=out, stderr=err)
    eng.exit(stdout=out, stderr=err)

