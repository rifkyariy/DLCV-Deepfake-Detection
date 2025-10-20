from glob import glob
import os
import pandas as pd
import cv2
from tqdm import tqdm
import numpy as np
import shutil
import json
import sys
import argparse
from imutils import face_utils
from retinaface.pre_trained_models import get_model
from retinaface.utils import vis_annotations
import torch


def facecrop(model,org_path,save_path,period=1,num_frames=10):
	cap_org = cv2.VideoCapture(org_path)
	croppedfaces=[]
	frame_count_org = int(cap_org.get(cv2.CAP_PROP_FRAME_COUNT))
	
	frame_idxs = np.linspace(0, frame_count_org - 1, num_frames, endpoint=True, dtype=int)
	
	for cnt_frame in range(frame_count_org): 
		ret_org, frame_org = cap_org.read()
		height,width=frame_org.shape[:-1]
		if not ret_org:
			tqdm.write('Frame read {} Error! : {}'.format(cnt_frame,os.path.basename(org_path)))
			continue
		
		if cnt_frame not in frame_idxs:
			continue
		
		frame = cv2.cvtColor(frame_org, cv2.COLOR_BGR2RGB)
		faces = model.predict_jsons(frame)
		try:
			if len(faces)==0:
				print(faces)
				tqdm.write('No faces in {}:{}'.format(cnt_frame,os.path.basename(org_path)))
				continue
			face_s_max=-1
			landmarks=[]
			size_list=[]
			for face_idx in range(len(faces)):
				
				x0,y0,x1,y1=faces[face_idx]['bbox']
				landmark=np.array([[x0,y0],[x1,y1]]+faces[face_idx]['landmarks'])
				face_s=(x1-x0)*(y1-y0)
				size_list.append(face_s)
				landmarks.append(landmark)
		except Exception as e:
			print(f'error in {cnt_frame}:{org_path}')
			print(e)
			continue
		landmarks=np.concatenate(landmarks).reshape((len(size_list),)+landmark.shape)
		landmarks=landmarks[np.argsort(np.array(size_list))[::-1]]
			

		save_path_=save_path+'frames/'+os.path.basename(org_path).replace('.mp4','/')
		os.makedirs(save_path_,exist_ok=True)
		image_path=save_path_+str(cnt_frame).zfill(3)+'.png'
		land_path=save_path_+str(cnt_frame).zfill(3)

		land_path=land_path.replace('/frames','/retina')
		os.makedirs(os.path.dirname(land_path),exist_ok=True)
		np.save(land_path, landmarks)

		if not os.path.isfile(image_path):
			cv2.imwrite(image_path,frame_org)

	cap_org.release()
	return

def facecrop_batched(model, org_path, save_path, period=1, num_frames=32, batch_size=16):
    cap_org = cv2.VideoCapture(org_path)
    if not cap_org.isOpened():
        print(f"Error opening video file: {org_path}")
        return

    frame_count_org = int(cap_org.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_idxs = np.linspace(0, frame_count_org - 1, num_frames, endpoint=True, dtype=int)
    
    frames_to_process = []
    original_frames = {} 
    
    for i, frame_idx in enumerate(frame_idxs):
        cap_org.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret_org, frame_org = cap_org.read()
        
        if not ret_org:
            tqdm.write(f'Frame read Error at index {frame_idx} in {os.path.basename(org_path)}')
            continue
        
        frame_rgb = cv2.cvtColor(frame_org, cv2.COLOR_BGR2RGB)
        frames_to_process.append(frame_rgb)
        original_frames[frame_idx] = frame_org

        if len(frames_to_process) == batch_size or i == len(frame_idxs) - 1:
            if not frames_to_process:
                continue

            all_faces_batch = [model.predict_jsons(frame) for frame in frames_to_process]
            
            batch_frame_indices = list(original_frames.keys())[-len(frames_to_process):]

            for j, faces in enumerate(all_faces_batch):
                current_frame_idx = batch_frame_indices[j]
                
                if not faces or len(faces) == 0:
                    tqdm.write(f'No faces in frame {current_frame_idx} of {os.path.basename(org_path)}')
                    continue

                landmarks = []
                size_list = []
                for face_idx in range(len(faces)):
                    # --- CORRECTED PART ---
                    bbox = faces[face_idx]['bbox']
                    # Skip if the bounding box is empty or malformed
                    if not bbox or len(bbox) < 4:
                        continue
                    
                    x0, y0, x1, y1 = bbox
                    # ----------------------

                    landmark = np.array([[x0, y0], [x1, y1]] + faces[face_idx]['landmarks'])
                    face_s = (x1 - x0) * (y1 - y0)
                    size_list.append(face_s)
                    landmarks.append(landmark)

                # If all faces in the frame were skipped, continue to the next
                if not landmarks:
                    continue

                landmarks = np.array(landmarks)
                sorted_landmarks = landmarks[np.argsort(np.array(size_list))[::-1]]
                
                base_name = os.path.basename(org_path).replace('.mp4', '')
                frame_str = str(current_frame_idx).zfill(3)
                
                land_path = os.path.join(save_path, 'retina', base_name, f'{frame_str}.npy')
                os.makedirs(os.path.dirname(land_path), exist_ok=True)
                np.save(land_path, sorted_landmarks)

                image_path = os.path.join(save_path, 'frames', base_name, f'{frame_str}.png')
                if not os.path.isfile(image_path):
                    os.makedirs(os.path.dirname(image_path), exist_ok=True)
                    cv2.imwrite(image_path, original_frames[current_frame_idx])

            frames_to_process = []
            original_frames = {}

    cap_org.release()

if __name__=='__main__':
	parser=argparse.ArgumentParser()
	parser.add_argument('-d',dest='dataset',choices=['DeepFakeDetection_original','DeepFakeDetection','FaceShifter','Face2Face','Deepfakes','FaceSwap','NeuralTextures','Original','Celeb-real','Celeb-synthesis','YouTube-real','DFDC','DFDCP'])
	parser.add_argument('-c',dest='comp',choices=['raw','c23','c40'],default='raw')
	parser.add_argument('-n',dest='num_frames',type=int,default=32)
	args=parser.parse_args()
	if args.dataset=='Original':
		dataset_path='data/FaceForensics++/original_sequences/youtube/{}/'.format(args.comp)
	elif args.dataset=='DeepFakeDetection_original':
		dataset_path='data/FaceForensics++/original_sequences/actors/{}/'.format(args.comp)
	elif args.dataset in ['DeepFakeDetection','FaceShifter','Face2Face','Deepfakes','FaceSwap','NeuralTextures']:
		dataset_path='data/FaceForensics++/manipulated_sequences/{}/{}/'.format(args.dataset,args.comp)
	elif args.dataset in ['Celeb-real','Celeb-synthesis','YouTube-real']:
		dataset_path='data/Celeb-DF-v2/{}/'.format(args.dataset)
	elif args.dataset in ['DFDC','DFDCVal']:
		dataset_path='data/{}/'.format(args.dataset)
	else:
		raise NotImplementedError

	device=torch.device('cuda')

	model = get_model("resnet50_2020-07-20", max_size=2048,device=device)
	model.eval()


	movies_path=dataset_path+'videos/'

	movies_path_list=sorted(glob(movies_path+'*.mp4'))
	print("{} : videos are exist in {}".format(len(movies_path_list),args.dataset))


	n_sample=len(movies_path_list)
	
	for i in tqdm(range(n_sample)):
		folder_path=movies_path_list[i].replace('videos/','frames/').replace('.mp4','/')
		if len(glob(folder_path.replace('/frames/','/retina/')+'*.npy'))<args.num_frames:
			# facecrop(model,movies_path_list[i],save_path=dataset_path,num_frames=args.num_frames)
			facecrop_batched(model, movies_path_list[i], save_path=dataset_path, num_frames=args.num_frames, batch_size=16)
	

	
