'''Improved mapGaze script with robust painting localization, preprocessing, and tuned matching'''
import os
import sys
import time
import logging
import shutil

import cv2
import numpy as np
import pandas as pd

# Global matcher settings
DIST_RATIO = 0.7     # Lowe's ratio (0-1) #lower is stricter
MIN_GOOD_MATCHES = 12  # require at least this many inliers # higher is stricter
RANSAC_THRESH = 8.0   # px reprojection threshold

###############################################################################

def adjust_gamma(frame, gamma=1.5):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(frame, table)


def local_brightness_norm(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)



################################################################################

def preprocess_frame(frame):
    # Brightness enhancement
    frame = adjust_gamma(frame, gamma=1.5)
    frame = local_brightness_norm(frame)

    # Convert to grayscale and apply bilateral filter and CLAHE
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, d=7, sigmaColor=50, sigmaSpace=7)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4,4)) # (clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)

    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    max_area = 0
    quad = None
    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4:
            area = cv2.contourArea(approx)
            if area > max_area:
                max_area = area
                quad = approx.reshape(4, 2)
    mask = None
    if quad is not None and max_area > 0.1 * frame.shape[0] * frame.shape[1]:
        mask = np.zeros_like(gray)
        cv2.fillPoly(mask, [quad], 255)
        gray = cv2.bitwise_and(gray, mask)
    return gray, mask


def find_matches(kp1, des1, kp2, des2):
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = matcher.knnMatch(des1, des2, k=2)
    good = [m for m, n in matches if m.distance < DIST_RATIO * n.distance]
    if len(good) >= MIN_GOOD_MATCHES:
        pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good])
        return pts1, pts2
    return None, None


def compute_homography(pts_ref, pts_frame):
    H, mask = cv2.findHomography(
        pts_ref.reshape(-1,1,2), pts_frame.reshape(-1,1,2),
        cv2.RANSAC, RANSAC_THRESH
    )
    return H, mask

def is_homography_reasonable(H, ref_shape, min_area=2000, max_aspect_ratio=4.0):
    h, w = ref_shape
    corners = np.float32([[0,0], [w,0], [w,h], [0,h]]).reshape(-1,1,2)
    projected = cv2.perspectiveTransform(corners, H).reshape(-1, 2)

    area = cv2.contourArea(projected.astype(np.float32))
    if area < min_area:
        return False

    side_lengths = [np.linalg.norm(projected[i] - projected[(i+1)%4]) for i in range(4)]
    ratio = max(side_lengths) / (min(side_lengths) + 1e-6)
    if ratio > max_aspect_ratio:
        return False

    return True

def process_frame(frame, ref_kp, ref_des, detector, ref_shape):
    gray, _ = preprocess_frame(frame)
    kp, des = detector.detectAndCompute(gray, None)
    if des is None or len(kp) < MIN_GOOD_MATCHES:
        return None

    pts_ref, pts_frame = find_matches(ref_kp, ref_des, kp, des)
    if pts_ref is None:
        return None

    H_ref2frame, _ = compute_homography(pts_ref, pts_frame)
    if H_ref2frame is None or not is_homography_reasonable(H_ref2frame, ref_shape):
        return None

    _, H_frame2ref = cv2.invert(H_ref2frame)
    return {'H_ref2frame': H_ref2frame, 'H_frame2ref': H_frame2ref}


def map_coords(pt, H):
    arr = np.array(pt, dtype=np.float32).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(arr, H)
    x, y = dst.ravel()
    return float(x), float(y)



def main(gaze_csv, video_path, ref_image_path, out_dir, ordered_tablo_path=None, ref2world_on=False, n_frames=None, print_every=100):
    os.makedirs(out_dir, exist_ok=True)
    shutil.copy(ref_image_path, out_dir)

    gazeWorld_df = pd.read_table(gaze_csv, sep=',', header=0)
    directory = os.path.dirname(gaze_csv)
    fixationData = os.path.join(directory, 'fixations.csv')
    fixations_df = pd.read_csv(fixationData)
    fixations_df['start_frame_index'] = fixations_df['start_frame_index'].astype(int)
    fixations_df['end_frame_index'] = fixations_df['end_frame_index'].astype(int)

    min_frame_index = min(
        gazeWorld_df['world_index'].min(),
        fixations_df['start_frame_index'].min())

    gazeWorld_df['world_index'] -= min_frame_index
    fixations_df['start_frame_index'] -= min_frame_index
    fixations_df['end_frame_index'] -= min_frame_index

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    ref_img = cv2.imread(ref_image_path)
    ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
    detector = cv2.AKAZE_create()
    ref_kp, ref_des = detector.detectAndCompute(ref_gray, None)

    h_ref, w_ref = ref_gray.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    #out_ref_vid = cv2.VideoWriter(os.path.join(out_dir, 'ref_gaze.mp4'), fourcc, fps, (w_ref, h_ref))

    out_world_vid = None
    if ref2world_on:
        h_frame = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        w_frame = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        out_world_vid = cv2.VideoWriter(os.path.join(out_dir, 'painting_in_world.mp4'), fourcc, fps, (w_frame, h_frame))

    idx = 0
    gaze_mapped_records = []  # Changed from mapped_records to be more specific
    fixations_mapped_records = []  # New list for fixation data
    
    print(f"Processing video frames... Total: {total_frames}")
    while True:
        ret, frame = cap.read()
        if not ret or (n_frames is not None and idx >= n_frames):
            break

        Hs = process_frame(frame, ref_kp, ref_des, detector, ref_gray.shape)
        frame_gaze = gazeWorld_df[gazeWorld_df['world_index'] == idx]
        ref_overlay = ref_img.copy()

        if Hs:
            # Process gaze data
            for _, row in frame_gaze.iterrows():
                wx = row['norm_pos_x'] * frame.shape[1]
                wy = (1 - row['norm_pos_y']) * frame.shape[0]
                rx, ry = map_coords((wx, wy), Hs['H_frame2ref'])
                
                gaze_mapped_records.append({
                    'worldFrame': idx,
                    'gaze_ts': row['gaze_timestamp'],
                    'confidence': row['confidence'],
                    'world_gazeX': wx,
                    'world_gazeY': wy,
                    'ref_gazeX': rx,
                    'ref_gazeY': ry
                })
                
                cv2.circle(ref_overlay, (int(rx), int(ry)), 8, (0,255,0), -1)

            # Process fixations that span this frame
            this_frame_fixations = fixations_df[
                (fixations_df['start_frame_index'] <= idx) &
                (fixations_df['end_frame_index'] >= idx)
            ]
            
            for _, fix_row in this_frame_fixations.iterrows():
                fix_wx = fix_row['norm_pos_x'] * frame.shape[1]
                fix_wy = (1 - fix_row['norm_pos_y']) * frame.shape[0]
                fix_rx, fix_ry = map_coords((fix_wx, fix_wy), Hs['H_frame2ref'])
                
                fixations_mapped_records.append({
                    'fixation_id': fix_row['id'],
                    'start_ts': fix_row['start_timestamp'],
                    'duration': fix_row['duration'],
                    'confidence': fix_row['confidence'],
                    'worldFrame': idx,
                    'world_fixX': fix_wx,
                    'world_fixY': fix_wy,
                    'ref_fixX': fix_rx,
                    'ref_fixY': fix_ry
                })
                
                # Draw fixation points if needed
                cv2.circle(ref_overlay, (int(fix_rx), int(fix_ry)), 16, (255,0,0), -1)  # Blue for fixations

            if ref2world_on:
                warped = cv2.warpPerspective(ref_img, Hs['H_ref2frame'], (frame.shape[1], frame.shape[0]))
                overlaid = cv2.addWeighted(frame, 0.5, warped, 0.5, 0)
                
        else:
            if ref2world_on:
                overlaid = frame.copy()
                cv2.putText(overlaid, "No match", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

        if ref2world_on:
            out_world_vid.write(overlaid)

        # out_ref_vid.write(ref_overlay)
        if idx % print_every == 0:
            print(f"Processed frame {idx}/{total_frames}")
        idx += 1

    cap.release()
    # out_ref_vid.release()
    if out_world_vid:
        out_world_vid.release()

    # Add anchor rows to gaze data
    first_entry = gazeWorld_df.iloc[0]
    last_entry = gazeWorld_df.iloc[-1]
    anchor_rows = pd.DataFrame([
        {
            'worldFrame': int(first_entry['world_index']),
            'gaze_ts': first_entry['gaze_timestamp'],
            'confidence': pd.NA,
            'world_gazeX': pd.NA,
            'world_gazeY': pd.NA,
            'ref_gazeX': pd.NA,
            'ref_gazeY': pd.NA
        },
        {
            'worldFrame': int(last_entry['world_index']),
            'gaze_ts': last_entry['gaze_timestamp'],
            'confidence': pd.NA,
            'world_gazeX': pd.NA,
            'world_gazeY': pd.NA,
            'ref_gazeX': pd.NA,
            'ref_gazeY': pd.NA
        }
    ])
    gaze_mapped_df = pd.DataFrame(gaze_mapped_records)
    gaze_mapped_df = pd.concat([gaze_mapped_df, anchor_rows], ignore_index=True)

    # Save the dataframes
    # gaze_mapped_df = pd.DataFrame(gaze_mapped_records)
    gaze_mapped_df.to_csv(os.path.join(out_dir, 'gazeData_mapped.tsv'), sep='\t', index=False)
    
    fixations_mapped_df = pd.DataFrame(fixations_mapped_records)
    # Add anchor rows to fixation data
    if not fixations_df.empty:
        fix_anchor_rows = pd.DataFrame([
            {
                'fixation_id': pd.NA,
                'start_ts': first_entry['gaze_timestamp'],
                'duration': pd.NA,
                'confidence': pd.NA,
                'worldFrame': int(first_entry['world_index']),
                'world_fixX': pd.NA,
                'world_fixY': pd.NA,
                'ref_fixX': pd.NA,
                'ref_fixY': pd.NA
            },
            {
                'fixation_id': pd.NA,
                'start_ts': last_entry['gaze_timestamp'],
                'duration': pd.NA,
                'confidence': pd.NA,
                'worldFrame': int(last_entry['world_index']),
                'world_fixX': pd.NA,
                'world_fixY': pd.NA,
                'ref_fixX': pd.NA,
                'ref_fixY': pd.NA
            }
        ])
        fixations_mapped_df = pd.concat([fixations_mapped_df, fix_anchor_rows], ignore_index=True)

    fixations_mapped_df.to_csv(os.path.join(out_dir, 'fixations_mapped.tsv'), sep='\t', index=False)


#####################################################################################################


def selectBestReferenceImage(videoPath, referenceImagesFolder, featureDetect, numFrames=10):
    """
    Select the best reference image based on the highest number of matches.

    Parameters
    ----------
    videoPath : str
        Path to the world camera video.
    referenceImagesFolder : str
        Path to the folder containing reference images.
    featureDetect : object
        Instance of cv2 SIFT class.
    numFrames : int
        Number of frames to sample from the beginning, middle, and end.

    Returns
    -------
    bestReferenceImage : str
        Path to the reference image with the highest matches.
    """
    # logger = logging.getLogger()
    # print("Selecting the best reference image...")

    # Load all image files from the reference images folder
    referenceImages = [
        os.path.join(referenceImagesFolder, f)
        for f in os.listdir(referenceImagesFolder)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]

    if not referenceImages:
        raise ValueError(f"No valid reference images found in folder: {referenceImagesFolder}")


    # Load the video
    vid = cv2.VideoCapture(videoPath)
    totalFrames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    # Sample frames: 10 frames from every 1/6th part of the video
    numSegments = 5
    numFramesPerSegment = 10
    sampleFrames = []
    for i in range(1, numSegments - 2):
        startFrame = int((i / numSegments) * totalFrames)  # Calculate the starting frame for the segment
        segmentFrames = np.arange(startFrame, startFrame + numFramesPerSegment)  # Select 10 frames
        sampleFrames.extend(segmentFrames)
    # Ensure unique and sorted frame indices
    sampleFrames = np.unique(np.clip(sampleFrames, 0, totalFrames - 1))  # Clip to valid frame range


    # Initialize match counts for each reference image
    matchCounts = {refImg: 0 for refImg in referenceImages}

    # Process each reference image
    for refImgPath in referenceImages:
        # print(f"Processing reference image: {refImgPath}")

        # Load and preprocess the reference image
        refImg = cv2.imread(refImgPath)
        refImgGray = cv2.cvtColor(refImg, cv2.COLOR_BGR2GRAY)
        ref_kp, ref_des = featureDetect.detectAndCompute(refImgGray, None)

        # Iterate over sampled frames
        for frameIdx in sampleFrames:
            vid.set(cv2.CAP_PROP_POS_FRAMES, frameIdx)
            ret, frame = vid.read()
            if not ret:
                continue

            # Convert frame to grayscale
            frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_kp, frame_des = featureDetect.detectAndCompute(frameGray, None)

            # Find matches
            if frame_des is not None and ref_des is not None:
                ref_matchPts, frame_matchPts = findMatches(ref_kp, ref_des, frame_kp, frame_des)
                if ref_matchPts is not None:
                    matchCounts[refImgPath] += len(ref_matchPts)
        # print(f"Match count: {matchCounts[refImgPath]}")
    # Release the video
    vid.release()


    bestReferenceImage = select_best_match(matchCounts)
    bestReferenceImageName = '/'.join(bestReferenceImage.split('/')[-2:])
    print(f"Best reference image selected: {bestReferenceImageName} with {matchCounts[bestReferenceImage]} matches.")
    
    return bestReferenceImage


def select_best_match(matchCounts):
    # Get all matches and find the top two
    sorted_matches = sorted(matchCounts.items(), key=lambda x: x[1], reverse=True)
    
    if len(sorted_matches) < 2:
        return sorted_matches[0] if sorted_matches else (None, 0)
    
    best_ref, best_count = sorted_matches[0]
    second_ref, second_count = sorted_matches[1]
    
    # Check if the top two are SURVAGE and VOUET with close counts
    is_survage = "Survage" in best_ref
    is_vouet = "VOUET" in second_ref
    
    # If SURVAGE is first and VOUET is second and counts are within 80%
    if (is_survage and is_vouet) and (best_count * 0.6 <= second_count):
        # print(f"Overriding SURVAGE ({best_count}) with VOUET ({second_count}) due to close match")
        return second_ref
    
    # Normal case - return best match
    return best_ref

############################################################################################################



if __name__ == '__main__':
    if len(sys.argv) == 6:
        path_gazeData = sys.argv[1]
        path_worldCameraVid = sys.argv[2]
        path_referenceImage = sys.argv[3]
        output_directory = sys.argv[4]
        ordered_tablo_path = sys.argv[5]

        print("Arguments received:")
        print(f"1. Gaze Data: {path_gazeData}")
        print(f"2. World Camera Video: {path_worldCameraVid}")
        print(f"3. Reference Image: {path_referenceImage}")
        print(f"4. Output Directory: {output_directory}")
        print(f"5. Ordered Tablo Path: {ordered_tablo_path}")

        

        # Select the best reference image if order is not held
        if ordered_tablo_path == 'False':
            # Initialize SIFT feature detector
            featureDetect = cv2.SIFT_create()
            bestReferenceImage = selectBestReferenceImage(path_worldCameraVid, path_referenceImage, featureDetect)
        else:
            bestReferenceImage = ordered_tablo_path

        main(path_gazeData, path_worldCameraVid, bestReferenceImage, output_directory, ordered_tablo_path, ref2world_on=False, n_frames=None, print_every=1000)
    else:
        print("Error with input args in MapGaze.py")
        sys.exit(1)
