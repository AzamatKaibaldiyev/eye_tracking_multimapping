import multiprocessing
import subprocess
import os
import pandas as pd
import argparse
import time
import sys
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")


def run_sync_audios_and_export_timestamps(script_path, general_recording_folder, video_audio_path, microphone_audio_path, micro_type, output_path):
    """Function to execute the script with the provided arguments."""
    start_time = time.time()
    command = ['python', script_path] + [general_recording_folder, video_audio_path, microphone_audio_path, micro_type, output_path]
    subprocess.run(command)
    end_time = time.time()
    print(f">>>>>>>>>>>>>>>>>>Concatenating and converting mjpegs to mp4 took: {end_time - start_time:.2f} seconds<<<<<<<<<<<<<<<<<<<<")

def run_mapGaze(folder_name, script_path, path_gazeData, path_worldCameraVid, path_referenceImage, output_directory, ordered_tablo_path):
    """Function to execute the script with the provided arguments."""
    start_time = time.time()
    command = ['python', script_path] + [path_gazeData, path_worldCameraVid, path_referenceImage, output_directory, ordered_tablo_path]
    subprocess.run(command)
    end_time = time.time()
    print(f">>>>>>>>>>>>>>>>>>Gaze mapping for {folder_name} took: {end_time - start_time:.2f} seconds<<<<<<<<<<<<<<<<<<<<")

def get_phase(input_arg):
    if input_arg == "A":
        return "phase1A"
    elif input_arg == "B":
        return "phase1B"
    elif input_arg == "2":
        return "phase2"
    else:
        return None  # or you could raise an exception for invalid input

def generate_processed_name(path_str, phase):
    # Extract the number (e.g., '158') from the path
    number = Path(path_str).parent.parent.name
    
    # Format based on the second argument
    if phase in ('A', 'B'):
        return f"processed_abc{number}"
    elif phase == '2':
        return f"processed_2abc{number}"
    else:
        raise ValueError("Second argument must be 'A', 'B', or '2'")
    
def get_image_path(folder_path, phase, index):
    """
    Returns the image path based on phase and index.
    
    Args:
        folder_path (str): Path to the folder containing images.
        phase (str): "phase1A", "phase1B", or "phase2".
        index (int): Index (0-5 for phase1A/B, 0-7 for phase2).
    
    Returns:
        str: Full path to the image.
    """
    # Define the letter order for each phase
    phase_orders = {
        "A": ["V", "B", "P", "L", "S", "H"],
        "B": ["M", "B", "V", "d", "D", "S"],
        "2": ["S", "L", "D", "P", "C", "M", "V", "F"],
    }

    # Check if phase is valid
    if phase not in phase_orders:
        raise ValueError(f"Invalid phase: {phase}. Must be 'phase1A', 'phase1B', or 'phase2'.")

    # Check if index is within valid range
    max_index = len(phase_orders[phase]) - 1
    if index < 0 or index > max_index:
        raise ValueError(f"Invalid index for {phase}. Must be 0-{max_index}.")

    # Get the expected starting letter
    target_letter = phase_orders[phase][index]

    # Scan the folder for matching images
    for file in os.listdir(folder_path):
        if file.startswith(target_letter) or file.startswith(target_letter.lower()):
            return str(Path(folder_path) / file)

    # If no match found
    raise FileNotFoundError(f"No image starting with '{target_letter}' found in {folder_path}.")





if __name__ == '__main__':

    # SET PATH AND SELECT PHASE    
    # Add more combinations as needed:
    # input_folder_group = [['/home/kaibald231/ABC/26Apr/BIN14:15/129/20250426151921955/exports',
    # '2',
    # True]]

    input_folder_group = [['/home/kaibald231/zProcessing_ABC/phase1/018/20241001093226680/exports',
    'B',
    True]]

    
    tableau_order = True # False if tablo order is not held in the video, #check word document for correct order







    for input_default, phase_group, tableau_order in input_folder_group:
        ####################################################################################################################################3
        
        # Select scripts to run
        run_sync_audio_export_timestamps = False
        run_gaze_map = True
        script_folder_path = os.path.dirname(os.path.abspath(__file__))
        references_folder_path = os.path.join(os.path.dirname(script_folder_path), 'reference_images')
        reference_default = os.path.join(references_folder_path, get_phase(phase_group))
        
        # Parse arguments
        parser = argparse.ArgumentParser()
        parser.add_argument('--inputDir', help = 'Input directory of pupil recordings and data', default = input_default)
        parser.add_argument('--referenceDir', help = 'Path to folder with the reference images', default  = reference_default)
        parser.add_argument('--outputRoot', help = 'Path to where output data is saved to')
        parser.add_argument('--scripts_folder',help = 'Path to scripts folder',  default = script_folder_path)
        args = parser.parse_args()

        # Check if input directory is provided
        if args.inputDir is not None:
            # Check if the directory is valid
            if not os.path.isdir(args.inputDir):
                print('Invalid input dir: {}'.format(args.inputDir))
                sys.exit()
            else:
                print(f'User provided input : {args.inputDir}')
        else:
            print('Please provide input directory path')
            sys.exit()

        # Check if reference images directory is provided
        if args.referenceDir is not None:
            # Check if the directory is valid
            if not os.path.isdir(args.referenceDir):
            # if not os.path.isfile(args.referenceDir):
                print('Invalid reference dir: {}'.format(args.referenceDir))
                sys.exit()
            else:
                print(f'User provided input : {args.referenceDir}')
        else:
            print('Please provide reference images directory path')
            sys.exit()

        # General folder containing raw recording data
        general_recording_folder = os.path.dirname(args.inputDir)
        parent_folder = os.path.dirname(general_recording_folder)
        bin_folder = os.path.dirname(parent_folder)
        print(general_recording_folder)
        if not os.path.isdir(general_recording_folder):
            print('Pupil video folder not found')
            sys.exit()

        total_start_time = time.time()

        ##############################################################################################################
        # Sync video and microphone audios, and provide timestamps for each painting
        output_folder_name  = generate_processed_name(input_default, phase_group)
        output_directory = os.path.join(args.inputDir, output_folder_name)
        os.makedirs(output_directory, exist_ok=True)

        script_path = os.path.join(args.scripts_folder, 'audio_processing_export_timestamps.py')
        audio_files = [os.path.join(general_recording_folder, f) for f in os.listdir(general_recording_folder) if f.startswith('audio') and f.endswith('.mp4')]
        if not audio_files:
            print("No pupil audio files found!")
            video_audio_path = None  # or handle this case appropriately
        else:
            video_audio_path = audio_files[0]

        if run_sync_audio_export_timestamps:
            # Always process pupil audio first
            pupil_script_args_audio_sync = [script_path, general_recording_folder, video_audio_path, video_audio_path, 'pupil', output_directory]
            print('^^^Processing pupil audio')
            pupil_process = multiprocessing.Process(target=run_sync_audios_and_export_timestamps, args=pupil_script_args_audio_sync)    
            pupil_process.start()
            pupil_process.join()

            # Handle mediation and participant audio based on phase_group
            if phase_group in ('A', 'B'):
                # Case 1: Only mediation audio (in parent_folder), no participant audio
                med_microphone_audio_path = [os.path.join(parent_folder, file) for file in os.listdir(parent_folder) 
                                        if file.endswith('.WAV')]
                if med_microphone_audio_path:
                    med_microphone_audio_path = med_microphone_audio_path[0]
                    med_script_args_audio_sync = [script_path, general_recording_folder, video_audio_path, 
                                                med_microphone_audio_path, 'med', output_directory]
                    print('^^^Processing mediation audio (phase A/B)')
                    med_process = multiprocessing.Process(target=run_sync_audios_and_export_timestamps, 
                                                    args=med_script_args_audio_sync)
                    med_process.start()
                    med_process.join()
                else:
                    print("No mediation microphone audio files found in parent_folder!")

            elif phase_group == '2':
                # Case 2: Both mediation (in bin_folder) and participant (in parent_folder) audio might exist
                # Check participant audio first
                participant_microphone_audio_path = [os.path.join(parent_folder, file) for file in os.listdir(parent_folder) 
                                                if file.endswith('.WAV')]
                if participant_microphone_audio_path:
                    participant_microphone_audio_path = participant_microphone_audio_path[0]
                    par_script_args_audio_sync = [script_path, general_recording_folder, video_audio_path, 
                                                participant_microphone_audio_path, 'partic', output_directory]
                    print('^^^Processing participant audio (phase 2)')
                    par_process = multiprocessing.Process(target=run_sync_audios_and_export_timestamps, 
                                                    args=par_script_args_audio_sync)
                    par_process.start()
                    par_process.join()
                
                # Then check mediation audio
                med_microphone_audio_path = [os.path.join(bin_folder, file) for file in os.listdir(bin_folder) 
                                        if file.endswith('.WAV')]
                if med_microphone_audio_path:
                    med_microphone_audio_path = med_microphone_audio_path[0]
                    med_script_args_audio_sync = [script_path, general_recording_folder, video_audio_path, 
                                                med_microphone_audio_path, 'med', output_directory]
                    print('^^^Processing mediation audio (phase 2)')
                    med_process = multiprocessing.Process(target=run_sync_audios_and_export_timestamps, 
                                                    args=med_script_args_audio_sync)
                    med_process.start()
                    med_process.join()
                else:
                    print("No mediation microphone audio files found in bin_folder!")
            else:
                raise ValueError(f"Unknown phase_group: {phase_group}")  # Stops with traceback


        ###############################################################################################################
        ### GAZE MAPPING
        # Paths for files
        folder_path = input_default
        script_path = os.path.join(args.scripts_folder, 'mapGaze.py')
        script_args_mapgaze = []

        for index,folder_name in enumerate(sorted(os.listdir(folder_path))):
            subfolder_path = os.path.join(folder_path, folder_name) 
            
            if os.path.isdir(subfolder_path) and folder_name != output_folder_name and len(folder_name) == 3:
                print('Subfolder: ', folder_name)
                # List all files in the folder
                all_files = os.listdir(subfolder_path)
                # Get paths of files
                gazeData_file = [file for file in all_files if file.endswith('gaze_positions.csv')]
                path_gazeData = os.path.join(subfolder_path, gazeData_file[0])
                worldCameraVid_file = [file for file in all_files if file.endswith('.mp4')]
                path_worldCameraVid = os.path.join(subfolder_path, worldCameraVid_file[0])

                path_referenceImage =reference_default
                if tableau_order:
                    print('Tableau order is held in the video++++++')
                    ordered_tablo_path = get_image_path(reference_default, phase_group, index)
                else:
                    print('Tableau order is NOT HELD-------')
                    ordered_tablo_path = 'False'

                # Set output directory
                output_directory = os.path.join(folder_path, output_folder_name, folder_name)
                os.makedirs(output_directory, exist_ok=True)

                #Packing all the arguments
                script_args_mapgaze.append((folder_name, (folder_name, script_path, path_gazeData, path_worldCameraVid, path_referenceImage,output_directory, ordered_tablo_path)))

                # Input error checking
                badInputs = []
                for arg in [path_gazeData, path_worldCameraVid, path_referenceImage]:
                    if not os.path.exists(arg):
                        badInputs.append(arg)
                if len(badInputs) > 0:
                    raise ValueError('{} does not exist! Check your input file path'.format(badInputs))
                    #sys.exit()


        from multiprocessing import Pool
        # Increase FFmpeg read attempts
        os.environ["OPENCV_FFMPEG_READ_ATTEMPTS"] = "30000"    

        def process_folder(args):
            folder_name, folder_args = args
            print(f'Processing {folder_name}')
            run_mapGaze(*folder_args)

        if run_gaze_map:
            pool = Pool(processes=3)  # Don't use 'with' statement
            try:
                pool.map(process_folder, script_args_mapgaze)
            except KeyboardInterrupt:
                print("\nCaught KeyboardInterrupt, terminating workers...")
                pool.terminate()  # Force terminate all processes
                pool.join()       # Wait for processes to actually terminate
            except Exception as e:
                print(f"\nError occurred: {e}, terminating workers...")
                pool.terminate()
                pool.join()
            else:
                # Normal completion
                pool.close()
                pool.join()

    ###############################################################################################################
        
        total_end_time = time.time()
        print("FINAL execution time : {} seconds", total_end_time-total_start_time)

print("|||||||||||||| FINISHED ALL |||||||||||||||")
