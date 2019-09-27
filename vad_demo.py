import os
import time
import numpy as np
import librosa
import librosa.display
import scipy
import csv
from pyvad import trim, vad
import matplotlib.pyplot as plt

# Main Code
def generate(file_name, frame=24):
	y, sr = librosa.load(file_name)
	start_i, end_i = preprocessing(y, sr)		# Preprocessing
	start_i=[round((i/16000-0.2)*frame) for i in start_i]
	#start_i=[round((i/16000)*frame) for i in start_i]
	end_i=[round((i/16000+0.2)*frame) for i in end_i]
	#end_i=[round((i/16000)*frame) for i in end_i]
	mid_i = [end_i[i]-start_i[i] for i in range(len(start_i))]
	print(start_i)
	print(end_i)
	print(mid_i)
	print(len(start_i))


	return start_i, end_i, mid_i

def preprocessing(y, sr):    

	# 1. Resampling to 16kHz    
	if sr != 16000:        
		sr_re = 16000  # sampling rate of resampling        
		y = librosa.resample(y, sr, sr_re)        
		sr = sr_re

	# 2. Denoising 
	y[np.argwhere(y == 0)] = 1e-10  
	y_denoise = scipy.signal.wiener(y, mysize=None, noise=None)
	
	# 3. Pre Emphasis filter    
	y_Emphasis = np.append(y_denoise[0], y_denoise[1:] - 0.97 * y_denoise[:-1])

	# 4. Normalization (Peak)    
	y_max = max(y_Emphasis)
	y_Emphasis = y_Emphasis / y_max *0.9 	# VAD 인식을 위해 normalize

	plt.figure(figsize=(14,9))
	plt.subplot(4,1,1)
	librosa.display.waveplot(y_Emphasis,sr=sr)

	i=1000
	while i<len(y_Emphasis)-1:
		y_max=max(abs(y_Emphasis[max(i-1000,0):min(i+1001,len(y_Emphasis))]))
		if abs(y_max)<0.15:
			y_Emphasis[i-1000:i+1001]=1e-10
			i+=2000
		else:
			i+=2000
	plt.subplot(4,1,2)
	librosa.display.waveplot(y_Emphasis,sr=sr)
	# 6. Voice Activity Detection (VAD) 
	y_vad = vad(y_Emphasis, sr, vad_mode=2)  ## VAD 사용하여 trim 수행
	if y_vad is None:        
		y_vad = y_Emphasis

	plt.subplot(4,1,3)
	librosa.display.waveplot(y_vad,sr=sr)

	# 7. Filtering for VAD
	y_diff = np.diff(y_vad).astype('int')
	start_i = np.where(y_diff == 1)[0]
	end_i = np.where(y_diff == -1)[0]
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
		if e-s <3201 or power < 0.010:
			y_vad[s-1:e+1]=0
			if s in start_i:
				start_i.remove(s)
				end_i.remove(e)
	
	plt.subplot(4,1,4)
	librosa.display.waveplot(y_vad,sr=sr)
	
	'''
	plt.show(block=False)
	plt.savefig('vad_lhw.jpg')
	plt.pause(10)
	plt.close()
	
	'''
	return start_i, end_i

def main():
	result=generate('input_avi/demo_new_pretrained.mp4')
	#result=generate('lhw_1.mp4')

if __name__ == '__main__': 
	main()