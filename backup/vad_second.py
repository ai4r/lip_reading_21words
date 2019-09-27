from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from moviepy.editor import *
import os

dir_name = '../dataset_lipread/old'
save_dir = '../dataset_lipread/test1'

file_name='005_old_d_hi1.mp4'
sub_number =6
start=6.2
end=7.35
full_path = os.path.join(dir_name, file_name)


targetname='{}/{}_{}.mp4'.format(save_dir,file_name[:-4],sub_number)

#ffmpeg_extract_subclip(full_path, 10, 11.2, targetname=targetname)
clip=VideoFileClip(full_path).subclip(start,end)
clip.write_videofile(targetname)
#[1.38, 3.0, 5.25, 7.5, 9.48]
#[2.1, 4.11, 6.12, 8.1, 10.11]
