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
import dlib
from imutils import face_utils

# --- NEW IMPORTS for multiprocessing ---
import multiprocessing
from functools import partial


def facecrop(org_path,save_path,face_detector,face_predictor,num_frames=10):
    """
    This function is the original one, no changes are needed here.
    It processes a single video file.
    """
    cap_org = cv2.VideoCapture(org_path)
    
    frame_count_org = int(cap_org.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count_org == 0:
        return # Skip empty or corrupted videos

    
    frame_idxs = np.linspace(0, frame_count_org - 1, num_frames, endpoint=True, dtype=int)
    for cnt_frame in range(frame_count_org): 
        ret_org, frame_org = cap_org.read()
        
        # Check if frame_org is None, which can happen with corrupted videos
        if not ret_org or frame_org is None:
            tqdm.write('Frame read {} Error! : {}'.format(cnt_frame,os.path.basename(org_path)))
            break
        
        if cnt_frame not in frame_idxs:
            continue
        
        frame = cv2.cvtColor(frame_org, cv2.COLOR_BGR2RGB)


        faces = face_detector(frame, 1)
        if len(faces)==0:
            # Using f-string for slightly cleaner output
            tqdm.write(f'No faces in frame {cnt_frame}:{os.path.basename(org_path)}')
            continue
            
        landmarks=[]
        size_list=[]
        for face_idx in range(len(faces)):
            landmark = face_predictor(frame, faces[face_idx])
            landmark = face_utils.shape_to_np(landmark)
            x0,y0=landmark[:,0].min(),landmark[:,1].min()
            x1,y1=landmark[:,0].max(),landmark[:,1].max()
            face_s=(x1-x0)*(y1-y0)
            size_list.append(face_s)
            landmarks.append(landmark)
            
        landmarks_np = np.array(landmarks)
        landmarks_sorted = landmarks_np[np.argsort(np.array(size_list))[::-1]]
            

        save_path_=save_path+'frames/'+os.path.basename(org_path).replace('.mp4','/')
        os.makedirs(save_path_,exist_ok=True)
        image_path=save_path_+str(cnt_frame).zfill(3)+'.png'
        land_path=save_path_+str(cnt_frame).zfill(3)

        land_path=land_path.replace('/frames','/landmarks')

        os.makedirs(os.path.dirname(land_path),exist_ok=True)
        np.save(land_path, landmarks_sorted)

        if not os.path.isfile(image_path):
            cv2.imwrite(image_path,frame_org)

    cap_org.release()
    return


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
    elif args.dataset in ['DFDC']:
        dataset_path='data/{}/'.format(args.dataset)
    else:
        raise NotImplementedError

    # Load models once in the main process before creating the pool
    print("Loading dlib models...")
    face_detector = dlib.get_frontal_face_detector()
    predictor_path = 'src/preprocess/shape_predictor_81_face_landmarks.dat'
    face_predictor = dlib.shape_predictor(predictor_path)
    
    movies_path=dataset_path+'videos/'
    movies_path_list=sorted(glob(movies_path+'*.mp4'))
    print(f"{len(movies_path_list)} videos are exist in {args.dataset}")

    # --- START of MULTIPROCESSING LOGIC ---

    # 1. Use 'partial' to create a new function with some arguments already filled in.
    #    This is necessary because the multiprocessing map function only accepts one iterable argument.
    task_function = partial(facecrop, 
                            save_path=dataset_path, 
                            face_detector=face_detector, 
                            face_predictor=face_predictor, 
                            num_frames=args.num_frames)

    # 2. Set the number of parallel processes (usually the number of CPU cores)
    num_cores = multiprocessing.cpu_count()
    print(f"Starting parallel processing with {num_cores} cores.")

    # 3. Create a pool of worker processes and distribute the work
    with multiprocessing.Pool(processes=num_cores) as pool:
        # pool.imap_unordered is efficient for tasks that take varying amounts of time.
        # We wrap the call with tqdm to get a progress bar over the list of videos.
        # The list() wrapper is important to ensure the main process waits for all tasks to complete.
        list(tqdm(pool.imap_unordered(task_function, movies_path_list), total=len(movies_path_list)))
    
    print("All videos processed successfully.")
    # --- END of MULTIPROCESSING LOGIC ---