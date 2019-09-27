import os
import time
import numpy as np
import librosa
import librosa.display
import scipy
import csv
from pyvad import trim, vad
import matplotlib.pyplot as plt
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from moviepy.editor import *

pre_emphasis = 0.97  # Pre-Emphasis filter coefficient
dir_name = '../dataset_lipread/new'
save_dir_name = '../dataset_lipread/test_new'
file_name_list=os.listdir(dir_name)
file_name_list =  [file for file in file_name_list if '099_new_u' in file]

#twiceTF=True
twiceTF=False
# Main Code
def generate():
	for file_name in file_name_list:
		full_path = os.path.join(dir_name, file_name)
		if twiceTF:
			if 'u' in file_name:
				file_name2=file_name.replace(file_name[8],'d')
				full_path2 = os.path.join(dir_name, file_name2)
			if 'd' in file_name:
				file_name2=file_name.replace(file_name[8],'u')
				full_path2 = os.path.join(dir_name, file_name2)
		y, sr = librosa.load(full_path)
		filter_value=0.2
		power_value=0.015
		filter_mode=0
		vad_mode=2
		firstornot=1
		
		# vad filtering
		while(True):

			start_i, end_i = preprocessing(file_name, y, sr, filter_value, power_value, filter_mode,vad_mode, firstornot)		# Preprocessing

			start_i=[round(i/16000,2) for i in start_i]
			end_i=[round(i/16000,2) for i in end_i]
			len_i=[]
			for i in range(len(start_i)):
				len_i.append(end_i[i]-start_i[i])
			min_len_i = min(len_i)
			if len(start_i)==7 and min_len_i>0.2 and firstornot==1:
				break
			else:			
				firstornot +=1
				key_input=input('Press option:')
				key_input=int(key_input)

				if key_input == 1:
					filter_value*= 0.8
				elif key_input== 2:
					filter_value*= 1.2

				elif key_input ==3:
					power_value*= 0.8
				elif key_input ==4:
					power_value*= 1.2

				elif key_input ==5:
					filter_mode=1
				elif key_input ==6:
					filter_mode=0
				elif key_input ==7:
					filter_mode=2	
				
				elif key_input ==8:
					vad_mode=1
				elif key_input ==9:
					vad_mode=3
				elif key_input >9:
					side=key_input%10
					if side==0:
						del start_i[-1]
						del end_i[-1]
						break
					else:
						del start_i[side-1]
						del end_i[side-1]
						break
				elif key_input==0:
					break
				else:
					vad_mode=2

		print(file_name)
		print(start_i)
		print(end_i)
		for i in range(len(start_i)):
			targetname='{}/{}_{}.mp4'.format(save_dir_name,file_name[:-4],i+1)
			clip=VideoFileClip(full_path).subclip(max(0.01, start_i[i]-0.2), end_i[i]+0.2)
			clip.write_videofile(targetname)
			if twiceTF:
				targetname2='{}/{}_{}.mp4'.format(save_dir_name,file_name2[:-4],i+1)
				#clip2=VideoFileClip(full_path2).subclip(start_i[i], end_i[i])
				clip2=VideoFileClip(full_path2).subclip(max(0.01, start_i[i]-0.2), end_i[i]+0.2)
				clip2.write_videofile(targetname2)

def preprocessing(file_name, y, sr, filter_value, power_value, filter_mode, vad_mode, firstornot):    

	# 1. Resampling to 16kHz    
	if sr != 16000:        
		sr_re = 16000  # sampling rate of resampling        
		y = librosa.resample(y, sr, sr_re)        
		sr = sr_re

	# 2. Denoising 
	y[np.argwhere(y == 0)] = 1e-10  
	y_denoise = scipy.signal.wiener(y, mysize=None, noise=None)
	
	# 3. Pre Emphasis filter    
	y_Emphasis = np.append(y_denoise[0], y_denoise[1:] - pre_emphasis * y_denoise[:-1])

	# 4. Normalization (Peak)    
	y_max = max(y_Emphasis)
	if filter_mode ==2:
		y_Emphasis[np.argwhere(y_Emphasis > y_max*0.1)] = 0
		y_max = max(y_Emphasis)
	y_Emphasis = y_Emphasis / y_max *0.9 	# VAD 인식을 위해 normalize
	
	plt.figure(figsize=(14,9))
	plt.subplot(4,1,1)
	librosa.display.waveplot(y_Emphasis,sr=sr)

	# 5. Additional Filtering
	if filter_mode>=1:
		i=2000
		while i<len(y_Emphasis)-1:
			y_max=max(abs(y_Emphasis[max(i-2000,0):min(i+2001,len(y_Emphasis))]))
			if abs(y_max)<filter_value:
				y_Emphasis[i-2000:i+2001]=1e-10
				i+=2000
			else:
				i+=2000
				
	plt.subplot(4,1,2)
	librosa.display.waveplot(y_Emphasis,sr=sr)

	# 6. Voice Activity Detection (VAD)    
	y_vad = vad(y_Emphasis, sr, vad_mode=vad_mode)  ## VAD 사용하여 trim 수행
	if y_vad is None:        
		y_vad = y_Emphasis

	plt.subplot(4,1,3)
	librosa.display.waveplot(y_vad,sr=sr)

	# 7. Filtering for VAD
	y_diff = np.diff(y_vad).astype('int')
	start_i = np.where(y_diff == 1)[0]
	end_i =   np.where(y_diff == -1)[0]
	start_i=start_i.tolist()
	end_i=end_i.tolist()

	if start_i[0]>end_i[0]:
		start_i=[0]+start_i
	if len(start_i)>len(end_i):
		end_i.append(len(y_Emphasis)-1)

	for i, (s, e) in enumerate(zip(start_i, end_i)):
		power = (np.mean(y_Emphasis[s:e]**2))**0.5
		if s==0:
			y_vad[s:e+1]=0
			start_i.remove(s)
			end_i.remove(e)
		if i>0 and s-end_i[i-1]<3300:
			y_vad[end_i[i-1]-1:s+1]=1
			start_i.remove(s)
			end_i.remove(end_i[i-1])
		if e-s <3201 or power < power_value:
			y_vad[s-1:e+1]=0
			if s in start_i:
				start_i.remove(s)
				end_i.remove(e)

	plt.subplot(4,1,4)
	librosa.display.waveplot(y_vad,sr=sr)

	# 8. Return start_i/end_i
	if len(start_i)==7 and firstornot==1:
		plt.close()
		return start_i, end_i
	else:
		plt.show(block=False)
		plt.pause(2)
		plt.close()
		return start_i, end_i



def main():
	result=generate()

if __name__ == '__main__': 
	main()