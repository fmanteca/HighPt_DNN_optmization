import os
import shutil
import glob


for myfile in glob.glob("send_*sh"):

    subcom = 'sbatch -o logfile.log -e errfile.err --qos=gridui_sort --partition=cloudcms -N 20 ' + myfile
    os.system(subcom)


 
 


                    


