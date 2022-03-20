import os
import pdb
import shutil


main_directory = './logs/final-baseline-v2-backup/Tanksworld Baseline v2'
portable_directory = './logs/final-baseline-v2-portable'
for folder in os.listdir(main_directory):
    if not folder.endswith('.json'):
        for seed_folder in os.listdir(os.path.join(main_directory, folder)):
            try:
                ckpt_folder = os.path.join(main_directory, folder, seed_folder, 'checkpoints')
                if not os.path.exists(ckpt_folder): continue
                subfolder1 = os.listdir(ckpt_folder)[0]
                ckpt_folder = os.path.join(ckpt_folder, subfolder1)
                subfolder2 = os.listdir(ckpt_folder)[0]
                ckpt_folder = os.path.join(ckpt_folder, subfolder2)
                if not os.path.exists(ckpt_folder): continue
                folder_ = folder.replace('cons', '')
                folder_ = folder_.replace('ff=0.0__', '')
                folder_ = folder_.replace('ff=0.5__', '')
                folder_ = folder_.replace('ff=1.0__', '')
                folder_ = folder_.replace('H=64__', 'H=64')
                folder_ = folder_.replace('H=8__', 'H=8')
                folder_ = folder_.replace('H=16__', 'H=16')
                folder_ = folder_.replace('H=32__', 'H=32')
                folder_ = folder_.replace('H=128__', 'H=128')
                portable_folder = os.path.join(portable_directory, folder_, seed_folder, 'checkpoints')
                ckpt_file = os.path.join(ckpt_folder, '999999.pth')
                if not os.path.exists(ckpt_file): continue
                os.makedirs(portable_folder, exist_ok=True)
                shutil.copy(ckpt_file, os.path.join(portable_folder, '999999.pth'))
            except:
                continue
