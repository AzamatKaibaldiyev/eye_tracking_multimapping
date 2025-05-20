""" Map gaze data from world camera coordinate system to reference image

This script automates mapping of gaze data from a world camera coordinate
system to a fixed reference image. Mobile eye-trackers often record gaze data
with respect to an outward facing world camera approximating the
participant's point-of-view. As a result, the gaze data is expressed in
an egocentric coordinate system which moves along with the participant's head.

Typical eye-tracking research, on the other hand, seeks to analyze gaze
behavior on a particular stimulus over time. In order to use mobile
eye-trackers in this context, one must first map the recorded gaze points from
the world camera coordinate system to the fixed coordinate system of the
target stimulus. This requires 1) identifying the target stimulus in every
frame of the world camera recording, 2) finding a linear transform that will
map between the appearance of the stimulus on the world camera frame and a 2D
reference version of the same stimulus, and 3) using that transform to project
the recorded gaze points to the 2D reference stimulus.

With the help of computers vision tools, this script automates this process
and yeilds output data files that facilitate subsequent analysis, specifically:
    - world_gaze.m4v:           world video w/ gaze points overlaid
    - ref_gaze.m4v:             video of ref image w/ gaze points overlaid
    - ref2world_mapping.m4v     video of reference image projected back into
                                world video
    - gazeData_mapped.tsv:      gazeData mapped to both coordinate systems, the
                                world and reference image

"""

# python 2/3 compatibility
from __future__ import division
from __future__ import print_function

import os
import sys
from os.path import join
import logging
import shutil
import time
import argparse
import time

import numpy as np
import pandas as pd
import cv2

OPENCV3 = (cv2.__version__.split('.')[0] == '3')


def findMatches(img1_kp, img1_des, img2_kp, img2_des):
    """ Find the matches between the descriptors for two images

    Parameters
    ----------
    img1_kp, img2_kp : list
        list of identified keypoints for each image; returned from
        detectAndCompute method on the cv2 featureDetect class.
    img1_des, img2_des : np.ndarray
        descriptors for each image; returned from detectAndCompute method on
        the cv2 featureDetect class.

    Returns
    -------
    img1_pts, img2_pts : list or None
        list of matched keypoints on each image

    """
    # Match settings
    min_good_matches = 4
    num_matches = 2
    FLANN_INDEX_KDTREE = 0
    distance_ratio = 0.5                # 0-1; lower values more conservative
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=10)        # lower = faster, less accurate
    matcher = cv2.FlannBasedMatcher(index_params, search_params)

    # find all matches
    matches = matcher.knnMatch(img1_des, img2_des, k=num_matches)

    # filter out cases where the 2 matches are too close to each other
    goodMatches = []
    for m, n in matches:
        if m.distance < distance_ratio * n.distance:
            goodMatches.append(m)

    if len(goodMatches) > min_good_matches:
        img1_pts = np.float32([img1_kp[i.queryIdx].pt for i in goodMatches])
        img2_pts = np.float32([img2_kp[i.trainIdx].pt for i in goodMatches])

        return img1_pts, img2_pts

    else:
        return None, None


def mapCoords2D(coords, transform2D):
    """ Map the supplied coords to a new coordinate system using the supplied
    transformation matrix

    Parameters
    ----------
    coords : tuple
        (x,y) coordinates
    transform2D : np.ndarray
        2D transformation matrix; produce by cv2.findHomography

    Returns
    -------
    float, float
        mapped coordinates after applying transform2D

    """

    coords = np.array(coords).reshape(-1, 1, 2)
    mappedCoords = cv2.perspectiveTransform(coords, transform2D)
    mappedCoords = np.round(mappedCoords.ravel())

    return mappedCoords[0], mappedCoords[1]


def projectImage2D(origFrame, transform2D, newImage):
    """ Project newImage into the origFrame

    Warp newImage according to the supplied transformation matrix, then
    project (insert) into the original frame.

    Parameters
    ----------
    origFrame : np.ndarray
        The original image you want to insert the newImage into
    transform2D : np.ndarray
        2D transformation matrix; produce by cv2.findHomography
    newImage : np.ndarray
        The image you would like to warp and project into the origFrame


    Returns
    -------
    newFrame : np.ndarray
        New frame (same dimensions as origFrame) with the warped and projected
        newImage written into it

    """
    # warp the new image to the video frame
    warpedImage = cv2.warpPerspective(newImage,
                                      transform2D,
                                      origFrame.T.shape[1:])

    # mask and subtract new image from video frame
    warpedImage_bw = cv2.cvtColor(warpedImage, cv2.COLOR_BGR2GRAY)
    if warpedImage.shape[2] == 4:
        alpha = warpedImage[:, :, 3]
        alpha[alpha == 255] = 1       # create mask of non-transparent pixels
        warpedImage_bw = cv2.multiply(warpedImage_bw, alpha)

    ret, mask = cv2.threshold(warpedImage_bw, 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    origFrame_bg = cv2.bitwise_and(origFrame, origFrame, mask=mask_inv)

    # mask the warped new image, and add to the masked background frame
    warpedImage_fg = cv2.bitwise_and(warpedImage[:, :, :3],
                                     warpedImage[:, :, :3],
                                     mask=mask)
    newFrame = cv2.add(origFrame_bg, warpedImage_fg)

    # return the warped new frame
    return newFrame


def processRecording(gazeData=None, worldCameraVid=None, referenceImage=None, outputDir=None, nFrames=None, vidOut_ref_on=False, Ref2world_on=False, num_matches_thresh=10):
    """ Map the gaze across all frames of mobile eye-tracking session

    This method will iterate over every frame of the supplied video recording.
    On each frame, it will look for the matches with the specified
    referenceImage, create a linear transformation matrix, and map the gaze
    data from the world camera coordinate system to the reference image
    coordinate system.

    This parent method will take care of setting up all of the inputs, and at
    the end, writing all of the output files

    Parameters
    ----------
    gazeData : string
        Path to the gazeData file. This file expected to be a .csv/.tsv file
        with columns for:
            timestamp - timestamp (ms) corresponding to each sample
            frame_idx - index (0-based) of the worldCameraVid frame
                        corresponding to each sample
            confidence - confidence of the validity of each sample (0-1)
            norm_pos_x - normalized x position of gaze location (0-1).
                         Normalized with respect to width of worldCameraVid
            norm_pos_y - normalized y position of gaze location (0-1).
                         Normalized with respect to height of worldCameraVid
    worldCameraVid : string
        Path to the video recording from the world camera (.mp4)
    referenceImage : string
        Path to the 2D reference image
    outputDir : string
        Path to output directory where data will be saved
    nFrames : int, optional
        If specified, will only process given number of frames (default of
        None means it will process ALL frames in the video). Useful for testing
        on abbreviated number of frames

    Output files
    ------------
    world_gaze.m4v : video
        world video with original gaze points overlaid
    ref_gaze.m4v : video
         ref image with mapped gaze points overlaid
    ref2world_mapping.m4v : video
        world video with reference image projected and inserted into it.
    gazeData_mapped.tsv :  data file
        gazeData represented in both coordinate systems, the world and
        reference image

    """
    # Create output directory
    if not os.path.isdir(outputDir):
        os.makedirs(outputDir)

    # Set up Logging
    fileLogger = logging.FileHandler(join(outputDir, 'mapGazeLog.log'), mode='w')
    fileLogger.setLevel(logging.DEBUG)
    fileLogFormat = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', '%m-%d %H:%M:%S')
    fileLogger.setFormatter(fileLogFormat)
    consoleLogger = logging.StreamHandler(sys.stdout)
    consoleLogger.setLevel(logging.INFO)
    consoleLogFormat = logging.Formatter('%(message)s')
    consoleLogger.setFormatter(consoleLogFormat)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.addHandler(fileLogger)
    logger.addHandler(consoleLogger)

    # Log Inputs
    # logger.info('Gaze Data File: {}'.format(gazeData))
    # logger.info('World Camera Video: {}'.format(worldCameraVid))
    # logger.info('Reference Image: {}'.format(referenceImage))
    # logger.info('Output Directory: {}'.format(outputDir))

    # Copy the reference stim into the output dir
    shutil.copy(referenceImage, outputDir)

    # Load gaze data
    gazeWorld_df = pd.read_table(gazeData, sep=',', header=0)

    # Load fixations data
    directory = os.path.dirname(gazeData)
    fixationData = os.path.join(directory, 'fixations.csv')
    fixations_df = pd.read_csv(fixationData)  # path to fixations.csv
    fixations_df['start_frame_index'] = fixations_df['start_frame_index'].astype(int)
    fixations_df['end_frame_index'] = fixations_df['end_frame_index'].astype(int)

    # Normalize the world_index column to start from 0
    min_frame_index = min(
    gazeWorld_df['world_index'].min(),
    fixations_df['start_frame_index'].min())
    
    # Normalize both gaze and fixation data to start from 0
    gazeWorld_df['world_index'] = gazeWorld_df['world_index'] - min_frame_index
    fixations_df['start_frame_index'] = fixations_df['start_frame_index'] - min_frame_index
    fixations_df['end_frame_index'] = fixations_df['end_frame_index'] - min_frame_index


    # Load the reference image
    refImg = cv2.imread(join(outputDir, referenceImage.split('/')[-1]))
    refImgColor = refImg.copy()      # store a color copy of the image
    refImg = cv2.cvtColor(refImg, cv2.COLOR_BGR2GRAY)  # convert the orig to bw

    ### Prep the video data #######################################
    # Load the video, get parameters
    vid = cv2.VideoCapture(worldCameraVid)
    total_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    if not OPENCV3:
        totalFrames = vid.get(cv2.CAP_PROP_FRAME_COUNT)
        vidSize = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                   int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        fps = vid.get(cv2.CAP_PROP_FPS)
        vidCodec = cv2.VideoWriter_fourcc(*'mp4v')
        featureDetect = cv2.SIFT_create()
    else:
        totalFrames = vid.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)
        vidSize = (int(vid.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)), int(vid.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)))
        fps = vid.get(cv2.cv.CV_CAP_PROP_FPS)
        vidCodec = cv2.cv.CV_FOURCC(*'mp4v')
        featureDetect = cv2.SIFT()

    # World camera output video
    # vidOut_world_fname = join(outputDir, 'world_gaze.mp4')
    # vidOut_world = cv2.VideoWriter()
    # vidOut_world.open(vidOut_world_fname, vidCodec, fps, vidSize, True)

    # Reference image output video
    if vidOut_ref_on:
        vidOut_ref_fname = join(outputDir, 'ref_gaze.mp4')
        vidOut_ref = cv2.VideoWriter()
        vidOut_ref.open(vidOut_ref_fname,
                        vidCodec,
                        fps,
                        (refImg.shape[1], refImg.shape[0]),
                        True)

    # Ref2world mapping output video (useful for debugging)
    if Ref2world_on:
        vidOut_ref2world_fname = join(outputDir, 'ref2world_mapping.mp4')
        vidOut_ref2world = cv2.VideoWriter()
        vidOut_ref2world.open(vidOut_ref2world_fname, vidCodec, fps, vidSize, True)

    ### Find keypoints, descriptors for the reference image
    refImg_kp, refImg_des = featureDetect.detectAndCompute(refImg, None)
    # logger.info('Reference Image: found {} keypoints'.format(len(refImg_kp)))

    ### Loop over video frames ###############################################
    if nFrames and nFrames < totalFrames:
        framesToUse = np.arange(0, nFrames, 1)
    else:
        framesToUse = np.arange(0, totalFrames, 1)
    frameProcessing_startTime = time.time()
    frameCounter = 0

    while vid.isOpened():
        # read the next frame of the video
        ret, frame = vid.read()

        # check if it's a valid frame
        if (ret is True) and (frameCounter in framesToUse):

            # make copy of the reference image for later use
            ref_frame = refImgColor.copy()

            # process this frame
            processedFrame = processFrame(frame,
                                          frameCounter,
                                          refImg_kp,
                                          refImg_des,
                                          featureDetect,
                                          total_frames, num_matches_thresh)

            # if good match between reference image and this frame
            if processedFrame['foundGoodMatch']:
                # print('found good match on frame {}'.format(frameCounter))

                # grab the gaze data (world coords) for this frame
                thisFrame_gazeData_world = gazeWorld_df.loc[gazeWorld_df['world_index'] == frameCounter]
                # print(frameCounter)
                # print('thisFrame_gazeData_world: {}'.format(thisFrame_gazeData_world.shape[0]))
                # project the reference image back into the video as a way to check for good mapping
                if Ref2world_on:
                    ref2world_frame = projectImage2D(processedFrame['origFrame'], processedFrame['ref2world'], refImgColor)

                # loop over all gaze data for this frame, translate to different coordinate systems
                for i, gazeRow in thisFrame_gazeData_world.iterrows():

                    ts = gazeRow['gaze_timestamp']
                    conf = gazeRow['confidence']

                    # translate normalized gaze data to world pixel coords
                    world_gazeX = gazeRow['norm_pos_x'] * processedFrame['frame_gray'].shape[1]
                    world_gazeY = (1-gazeRow['norm_pos_y']) * processedFrame['frame_gray'].shape[0]


                    # covert from world to reference image pixel coordinates
                    ref_gazeX, ref_gazeY = mapCoords2D((world_gazeX, world_gazeY), processedFrame['world2ref'])

                    # create dict for this row
                    thisRow_df = pd.DataFrame({'gaze_ts': ts,
                                               'worldFrame': frameCounter,
                                               'confidence': conf,
                                               'world_gazeX': world_gazeX,
                                               'world_gazeY': world_gazeY,
                                               'ref_gazeX': ref_gazeX,
                                               'ref_gazeY': ref_gazeY},
                                               index=[i])

                    # append row to gazeMapped_df output
                    if 'gazeMapped_df' in locals():
                        # print('before locals')
                        gazeMapped_df = pd.concat([gazeMapped_df, thisRow_df])
                        # print('LOCALS')
                    else:
                        # print('before not')
                        gazeMapped_df = thisRow_df
                        # print('NOT LOCALS')

                    # ### Draw gaze circles on frames
                    # if i == thisFrame_gazeData_world.index.max():
                    #     dotColor = [96, 52, 234]            # pinkish/red
                    #     dotSize = 12
                    # else:
                    #     dotColor = [168, 231, 86]            # minty green
                    #     dotSize = 8

                    dotColor = [0, 255, 0]            #green
                    dotSize = 6
                    
                    if vidOut_ref_on:
                        # world frame
                        cv2.circle(frame,
                                (int(world_gazeX), int(world_gazeY)),
                                dotSize,
                                dotColor,
                                -1)
                        
                        
                        # ref frame
                        dotSize = 12
                        cv2.circle(ref_frame,
                                (int(ref_gazeX), int(ref_gazeY)),
                                dotSize,
                                dotColor,
                                -1)
                    


                # Process fixations that span this frame
                thisFrame_fixations = fixations_df[
                    (fixations_df['start_frame_index'] <= frameCounter) &
                    (fixations_df['end_frame_index'] >= frameCounter)
                ]

                for j, fixRow in thisFrame_fixations.iterrows():
                    fix_ts = fixRow['start_timestamp']
                    fix_conf = fixRow['confidence']
                    fix_id = fixRow['id']
                    fix_dur = fixRow['duration']

                    # Get normalized fixation position
                    fixX_world = fixRow['norm_pos_x'] * processedFrame['frame_gray'].shape[1]
                    fixY_world = (1 - fixRow['norm_pos_y']) * processedFrame['frame_gray'].shape[0]

                    # Map to reference image
                    fixX_ref, fixY_ref = mapCoords2D((fixX_world, fixY_world), processedFrame['world2ref'])

                    fixRow_df = pd.DataFrame({
                        'fixation_id': fix_id,
                        'start_ts': fix_ts,
                        'duration': fix_dur,
                        'confidence': fix_conf,
                        'worldFrame': frameCounter,
                        'world_fixX': fixX_world,
                        'world_fixY': fixY_world,
                        'ref_fixX': fixX_ref,
                        'ref_fixY': fixY_ref
                    }, index=[j])

                    if 'fixationsMapped_df' in locals():
                        fixationsMapped_df = pd.concat([fixationsMapped_df, fixRow_df])
                    else:
                        fixationsMapped_df = fixRow_df

                    # Optional: draw fixation points
                    if vidOut_ref_on:
                        cv2.circle(frame,
                                (int(fixX_world), int(fixY_world)),
                                10, [255, 0, 0], -1)  # blue

                        cv2.circle(ref_frame,
                                (int(fixX_ref), int(fixY_ref)),
                                16, [255, 0, 0], -1)  # blue

            else:
                # if not a good match, use the original frame for the ref2world
                if Ref2world_on:
                    ref2world_frame = processedFrame['origFrame']

            # write outputs to video
            # vidOut_world.write(frame)
            if vidOut_ref_on:
                vidOut_ref.write(ref_frame)
            if Ref2world_on:
                vidOut_ref2world.write(ref2world_frame)

        # increment frame counter
        frameCounter += 1
        if frameCounter > np.max(framesToUse):
            # release all videos
            vid.release()
            # vidOut_world.release()
            if vidOut_ref_on:
                vidOut_ref.release()
            if Ref2world_on:
                vidOut_ref2world.release()

            # write out gaze data
            try:
                colOrder = ['worldFrame', 'gaze_ts', 'confidence',
                            'world_gazeX', 'world_gazeY',
                            'ref_gazeX', 'ref_gazeY']
                gazeMapped_df[colOrder].to_csv(join(outputDir, 'gazeData_mapped.tsv'),
                                               sep='\t',
                                               index=False,
                                               float_format='%.3f')
            except Exception as e:
                logger.info(e)
                logger.info('cound not write gazeData_mapped to csv')
                pass

            try:
                fixColOrder = ['fixation_id', 'start_ts', 'duration', 'confidence',
                            'worldFrame', 'world_fixX', 'world_fixY',
                            'ref_fixX', 'ref_fixY']
                fixationsMapped_df[fixColOrder].to_csv(join(outputDir, 'fixations_mapped.tsv'),
                                                    sep='\t',
                                                    index=False,
                                                    float_format='%.3f')
            except Exception as e:
                print("Could not write fixation output:", e)

    endTime = time.time()
    frameProcessing_time = endTime - frameProcessing_startTime
    logger.info('Total time: %s seconds' % frameProcessing_time)
    logger.info('Avg time/frame: %s seconds' % (frameProcessing_time / framesToUse.shape[0]))


def processFrame(frame, frameIdx, ref_kp, ref_des, featureDetect, total_frames, num_matches_thresh=10):
    """ Process single frame from the world camera to determine mapping to
    ref image

    Parameters
    ---------
    frame : np.ndarray
        frame from world camera video
    frameIdx : int
        frame index (0-based)
    ref_kp : list
        identified keypoints on the reference image
    ref_des : np.ndarray
        descriptors for the reference image keypoints
    featureDetect : object
        instance of cv2 SIFT class

    Returns
    -------
    fr : dict
        dictionary with entries storing all of the relevant output for this
        particular frame

    """
    logger = logging.getLogger()

    fr = {}        # create dict to store info for this frame

    # create copy of original frame
    origFrame = frame.copy()
    fr['origFrame'] = origFrame         # store

    # convert to grayscale
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    fr['frame_gray'] = frame_gray

    # try to match the frame and the reference image
    try:
        frame_kp, frame_des = featureDetect.detectAndCompute(frame_gray, None)
        if frameIdx%100==0:
            logger.info(f'Frame {frameIdx}/{total_frames}')
            #logger.info('found {} features on frame {}'.format(len(frame_kp), frameIdx))

        if len(frame_kp) < 2:
            ref_matchPts = None
        else:
            ref_matchPts, frame_matchPts = findMatches(ref_kp,
                                                       ref_des,
                                                       frame_kp,
                                                       frame_des)

        # check if matches were found
        try:
            numMatches = ref_matchPts.shape[0]

            # if sufficient number of matches....
            if numMatches > num_matches_thresh:
                #logger.info('found {} matches on frame {}'.        format(numMatches, frameIdx))
                sufficientMatches = True
            else:
                logger.info('Insufficient matches ({}} matches) on frame {}'.format(numMatches, frameIdx))
                sufficientMatches = False

        except:
            # print('no matches found on frame {}'.format(frameIdx))
            sufficientMatches = False
            pass

        fr['foundGoodMatch'] = sufficientMatches

        # figure out homographies between coordinate systems
        if sufficientMatches:
            ref2world_transform, mask = cv2.findHomography(ref_matchPts.reshape(-1, 1, 2),
                                                           frame_matchPts.reshape(-1, 1, 2),
                                                           cv2.RANSAC,
                                                           5.0)
            world2ref_transform = cv2.invert(ref2world_transform)

            fr['ref2world'] = ref2world_transform
            fr['world2ref'] = world2ref_transform[1]

    except:
        fr['foundGoodMatch'] = False

    # return the processed frame
    return fr

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




if __name__ == '__main__':

    if len(sys.argv)==6:

        path_gazeData = sys.argv[1]
        path_worldCameraVid = sys.argv[2]
        path_referenceImage = sys.argv[3]
        output_directory = sys.argv[4]
        ordered_tablo_path = sys.argv[5]

        # Initialize SIFT feature detector
        featureDetect = cv2.SIFT_create()

        # Select the best reference image if order is not held
        if ordered_tablo_path == 'False':
            bestReferenceImage = selectBestReferenceImage(path_worldCameraVid, path_referenceImage, featureDetect)
        else:
            bestReferenceImage = ordered_tablo_path
        
        if 'Survage' in bestReferenceImage:
            num_matches_thresh = 40
            Ref2world_on = True
        else:
            num_matches_thresh = 10
            Ref2world_on = False

        processRecording(gazeData=path_gazeData,
                                 worldCameraVid=path_worldCameraVid,
                                 referenceImage=bestReferenceImage,
                                 outputDir=output_directory,
                                 nFrames = None,
                                 vidOut_ref_on=False,
                                 Ref2world_on=Ref2world_on,
                                 num_matches_thresh = num_matches_thresh)
        
    else:
        print("Problem with arguments")
