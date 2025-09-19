import os
import stat
import sys
import shutil
import sipconfig
import platform
import optparse

import myfbxconfig

lib_ext = ''

def build():
    build_file = "fbx_module.sbf"
    config = sipconfig.Configuration()
    build_macros = config.build_macros()
    
    if platform.system() == 'Linux':
        build_macros['CXXFLAGS_WARN_ON'] = ''
        config.platform = 'linux-g++'
    elif platform.system() == 'Windows':
        build_macros['CXXFLAGS_WARN_ON'] = ''
        config.platform = 'win32-msvc2005'
        
    #if platform.system() == "Windows":
    #    platform_tag = "FBX_WIN"
    #elif platform.system() == "Linux":
    #    platform_tag = "FBX_LINUX"
    #else:
    #    platform_tag = "FBX_MACX"

    if platform.machine() == 'x86_64':
        platform_tag = "FBX_X64"
    else:
        platform_tag = "FBX_X86"
    
    config.set_build_macros(build_macros)
    result = os.system(" ".join(["\"" + config.sip_bin + "\"", "-t", platform_tag, "-c", "./bin", "-b", "./bin/"+build_file, "./sip/fbx_module.sip"]))
    
    if result == 1:
        return False;
    
    makefile = sipconfig.SIPModuleMakefile(config, build_file, dir="./bin")
    makefile.extra_include_dirs = myfbxconfig.EXTRA_INCLUDE_DIRS
    makefile.extra_lib_dirs = myfbxconfig.EXTRA_LIB_DIRS
    
    if platform.system() == 'Linux':
        if platform.machine() == 'x86':
            makefile.extra_libs = myfbxconfig.EXTRA_LIBS_LINUX32
        else:
            makefile.extra_libs = myfbxconfig.EXTRA_LIBS_LINUX64
    elif platform.system() == 'Windows':
        makefile.extra_libs = myfbxconfig.EXTRA_LIBS_WIN32
    
    makefile.generate()
    
    # build with MAKEFILE    
    os.chdir('bin')
    
    if platform.system() == 'Linux':
        os.system('make -j 4')
    elif platform.system() == 'Windows':        
        # process Makefile
        mf = open("Makefile")
        ofiles = mf.readlines()
        mf.close()
        os.remove("Makefile")
        tmpmf = open("Makefile.template")
        tmpcontents = tmpmf.readlines()
        tmpmf.close()
        mf = open("Makefile", 'w')
        for i in range(len(ofiles)):
            if ofiles[i].startswith('.SUFFIXES:'):
                start = i
            if ofiles[i].startswith('install:'):
                end = i
                break
        for i in range(start):
            mf.write(ofiles[i])
        for line in tmpcontents:
            mf.write(line)
        mf.write('\n')
        for i in range(end, len(ofiles)):
            mf.write(ofiles[i])
        mf.close()
       
        os.system('RunMakePyFbx.bat')
    
    os.chdir(os.pardir)
    
    return True

def run(only_unittest):
    if os.path.exists('./bin') == False:
        os.mkdir("bin")
    
    if not only_unittest:
        if not build():
            return;

    if not 'PYTHONPATH' in os.environ:
        os.environ.setdefault('PYTHONPATH', '')
    if platform.system() == 'Linux':
        os.environ['PYTHONPATH'] += ':' + os.getcwd() + '/bin' + ':' + os.getcwd()
    elif platform.system() == 'Windows':
        os.environ['PYTHONPATH'] += ';' + os.getcwd() + '/bin' + ';' + os.getcwd()
    
    # Run unittests & examples     
    if os.path.exists('./temp') == False:
        os.mkdir("temp")
    os.chdir("temp")
    os.system("\"" + sys.executable + "\" ../unittest/fbx_testsuite.py")    
    #os.system("\"" + sys.executable + "\" ../examples/ExportScene01/ExportScene01.py")
    #os.system("\"" + sys.executable + "\" ../examples/ExportScene02/ExportScene02.py")
    #os.system("\"" + sys.executable + "\" ../examples/ExportScene03/ExportScene03.py")
    #os.system("\"" + sys.executable + "\" ../examples/ExportScene04/ExportScene04.py")
    #os.system("\"" + sys.executable + "\" ../examples/UserProperties/UserProperties.py")
    #os.system("\"" + sys.executable + "\" ../examples/AnimLayersAndStack/AnimLayersAndStack.py")
    #os.system("\"" + sys.executable + "\" ../examples/ImportScene/ImportScene.py ./ExportScene01.fbx")
    #os.system("\"" + sys.executable + "\" ../examples/ImportScene/ImportScene.py ./ExportScene02.fbx")
    #os.system("\"" + sys.executable + "\" ../examples/ImportScene/ImportScene.py ./ExportScene03_MC.fbx")
    #os.system("\"" + sys.executable + "\" ../examples/ImportScene/ImportScene.py ./ExportScene04.fbx")
    #os.system("\"" + sys.executable + "\" ../examples/ImportScene/ImportScene.py ./UserProperties.fbx")
    #os.system("\"" + sys.executable + "\" ../examples/ImportScene/ImportScene.py ./AnimLayersAndStack.fbx")
    
if __name__ == '__main__':
    if platform.system() == 'Linux':
        lib_ext = '.so'
    elif platform.system() == 'Windows':
        lib_ext = '.pyd'
    
    p = optparse.OptionParser()
    p.add_option('--OnlyUnitest', '-u', default=False, action='store_true', help='skip build and run unittests only')
    options, arguments = p.parse_args()
    run(only_unittest = options.OnlyUnitest)
