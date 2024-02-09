from os import listdir
from os.path import isfile, join
import os

models_folder = 'models'
views_folder = 'views'

files = [f for f in listdir(models_folder) if isfile(join(models_folder, f))]

for file in files:
    if file.rsplit('.', 1)[1] != 'scad':
        continue
    print(file)
    fp = join(models_folder, file)
    outpath = join(views_folder, file.rsplit('.', 1)[0])
    if not os.path.exists(outpath):
        os.makedirs(outpath)
        
    for x in range(-1,2):
        for y in range(-1,2):
            for z in range(-1,2):
                if x==0 and y==0 and z==0:
                    continue
                outfile = join(outpath, str(x+1)+str(y+1)+str(z+1)+'.png')
                if os.path.exists(outfile):
                    continue
                
                cmd = 'openscad -o '+outfile+' --camera '+str(x)+','+str(y)+','+str(z)+',0,0,0 --viewall --autocenter --imgsize=2048,2048 '+fp
                print(cmd)
                os.system(cmd)
