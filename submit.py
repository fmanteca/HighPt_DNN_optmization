import os
import shutil
import glob


for myfile in glob.glob("send_*sh"):

    subcom = 'sbatch -o logfile.log -e errfile.err --qos=gridui_sort --partition=cloudcms ' + myfile
    os.system(subcom)


 
 


                    


