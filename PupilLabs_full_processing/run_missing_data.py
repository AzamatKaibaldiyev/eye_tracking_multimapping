import subprocess
import multiprocessing
from pathlib import Path
import os
import glob

# Automatically resolve the path to mapGaze.py
SCRIPT_PATH = Path(__file__).parent / "mapGaze_new22.py"

def run_mapgaze_job(args):
    gaze_path, world_vid, ref_img, output_dir, ordered_tablo = args

    command = [
        "python",
        str(SCRIPT_PATH),  # Use absolute path for robustness
        gaze_path,
        world_vid,
        ref_img,
        output_dir,
        ref_img
    ]

    try:
        subprocess.run(command, check=True)
        print(f"✅ Finished: {output_dir}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed: {output_dir} — {e}")

def create_jobs_from_missing_folders(missing_folders):
    """
    Create a jobs list from missing folder paths.
    
    Args:
        missing_folders (list): List of folder paths that are missing required files
        
    Returns:
        list: List of tuples in the format (gaze_csv, world_mp4, ref_jpg, output_path, "False")
    """
    jobs = []
    
    for missing_folder in missing_folders:
        try:
            missing_path = Path(missing_folder)
            
            # Extract the folder name (e.g., "006" from the path)
            folder_name = missing_path.name
            
            # Navigate to the exports folder
            # From: /home/kaibald231/ABC/26Apr/BIN13:15/014/20250426142058230/exports/processed_2abc014/006
            # To:   /home/kaibald231/ABC/26Apr/BIN13:15/014/20250426142058230/exports/006
            
            # Go up to exports folder and then to the folder with same name
            exports_path = missing_path.parent.parent / folder_name
            
            # Path 1: gaze_positions.csv in exports folder
            gaze_csv = exports_path / "gaze_positions.csv"
            
            # Path 2: Find .mp4 file in exports folder (usually world.mp4)
            mp4_files = list(exports_path.glob("*.mp4"))
            if mp4_files:
                world_mp4 = mp4_files[0]  # Take the first .mp4 file found
            else:
                # If no .mp4 found, use expected path
                world_mp4 = exports_path / "world.mp4"
            
            # Path 3: Find .jpg file in the missing folder (original path)
            jpg_files = list(missing_path.glob("*.jpg"))
            if jpg_files:
                ref_jpg = jpg_files[0]  # Take the first .jpg file found
            else:
                # If no .jpg found, use a generic name
                ref_jpg = missing_path / "reference.jpg"
            
            # Path 4: Output path (the missing folder itself)
            output_path = missing_folder
            
            # Path 5: Always "False"
            flag = "True"
            
            # Create the job tuple
            job_tuple = (
                str(gaze_csv),
                str(world_mp4),
                str(ref_jpg),
                str(output_path),
                flag
            )
            
            jobs.append(job_tuple)
            
        except Exception as e:
            print(f"Error processing folder {missing_folder}: {e}")
            continue
    
    return jobs

def print_jobs_list(jobs):
    """
    Print the jobs list in a readable format.
    """
    print("jobs = [")
    for i, job in enumerate(jobs):
        print("    (")
        print(f'        "{job[0]}",')
        print(f'        "{job[1]}",')
        print(f'        "{job[2]}",')
        print(f'        "{job[3]}",')
        print(f'        "{job[4]}"')
        if i < len(jobs) - 1:
            print("    ),")
        else:
            print("    )")
    print("]")


if __name__ == "__main__":
    # 🧠 Define your argument sets (EDIT these)
    jobs = [
    (
        "/home/kaibald231/ABC/27jan/BIN9:15/110/20250127100924976/exports/007/gaze_positions.csv",
        "/home/kaibald231/ABC/27jan/BIN9:15/110/20250127100924976/exports/007/world.mp4",
        "/home/kaibald231/ABC/27jan/BIN9:15/110/20250127100924976/exports/processed_2abc110/007/FLORIS Frans, Portrait de dame âgée, parfois dit la Femme du fauconnier Inv.47.jpg",
        "/home/kaibald231/ABC/27jan/BIN9:15/110/20250127100924976/exports/processed_2abc110/007",
        "True"
    ),

]

    os.environ["OPENCV_FFMPEG_READ_ATTEMPTS"] = "30000" 

    from multiprocessing import Pool
    pool = Pool(processes=3)  
    try:
        pool.map(run_mapgaze_job, jobs)
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
