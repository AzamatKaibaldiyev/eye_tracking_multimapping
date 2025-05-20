import os
import sys
from pydub import AudioSegment
import json
import audalign as ad
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import numpy as np
import subprocess
import shutil
from pathlib import Path




# Aligning audios
def sync_audios(pupil_audio, microph_audio, destination_path):
    correlation_rec = ad.CorrelationRecognizer()
    cor_spec_rec = ad.CorrelationSpectrogramRecognizer()

    results = ad.align_files(
        pupil_audio,
        microph_audio,
        destination_path = destination_path,
        recognizer=correlation_rec
    )

    # results can then be sent to fine_align
    fine_results = ad.fine_align(
        results,
        recognizer=cor_spec_rec,
    )
    pupil_audio_name = pupil_audio.split('/')[-1]
    offset_pupil = fine_results[pupil_audio_name]
    if offset_pupil==0:
        micr_audio_name = microph_audio.split('/')[-1]
        offset_pupil = -fine_results[micr_audio_name]

    return offset_pupil 


# Get audio length
def get_audio_duration(audio_file):
    command = [
        "ffprobe",
        "-i", audio_file,
        "-show_entries", "format=duration",
        "-v", "quiet",
        "-of", "csv=p=0"
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    duration = float(result.stdout.strip())
    return duration


# Adjusting microphone audio to pupil world video
def adjust_audio_offset(input_audio_path, output_audio_path, offset_beginning, offset_end):
    # Load the input audio file
    audio = AudioSegment.from_file(input_audio_path)
    
    # Calculate the offset_beginning in milliseconds
    offset_ms = int(offset_beginning * 1000)  # Convert offset_beginning from seconds to milliseconds
    
    if offset_ms > 0:
        # If offset_beginning is positive, trim the audio from the beginning
        adjusted_audio = audio[offset_ms:]
        print(f"Trimming the beginning by {offset_ms/1000} seconds")
    else:
        # If offset_beginning is negative, add silence duration at the beginning
        offset_ms = abs(offset_ms)
        silence_duration = AudioSegment.silent(duration=offset_ms)
        adjusted_audio = silence_duration + audio
        print(f"Adding duration to the beginning by {offset_ms/1000} seconds")

    # Calculate the offset_end in milliseconds
    offset_ms = int(offset_end * 1000)  # Convert offset_beginning from seconds to milliseconds
    
    if offset_ms > 0:
        # If offset_end is positive, trim the audio from the end
        adjusted_audio = adjusted_audio[:len(adjusted_audio)-offset_ms]
        print(f"Trimming the end by {offset_ms/1000} seconds")
    else:
        # If offset_end is negative, add silence duration at the end
        offset_ms = abs(offset_ms)
        silence_duration = AudioSegment.silent(duration=offset_ms)
        adjusted_audio = adjusted_audio + silence_duration
        print(f"Adding duration to the end by {offset_ms/1000} seconds")
    
    # Export the adjusted audio to the output file
    adjusted_audio.export(output_audio_path, format="mp3")


# Cutting audios
def cut_audio(input_audio_path, timestamps, output_path):
    # Load the audio file
    audio = AudioSegment.from_mp3(input_audio_path)

    # Iterate over timestamps
    start_time, end_time  = timestamps
    # Extract audio segment
    segment = audio[start_time : end_time]  
    
    # Export audio segment
    segment.export(output_path, format="mp3")




def process_folders_for_transcription(input_root, language="french", microph_type = ''):
    """Process all folders to transcribe audio and save results in same folders"""
    # Initialize device
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    
    print(f"Initializing Whisper model on {device}...")
    
    # Model config with fallback for FlashAttention2
    model_kwargs = {
        "torch_dtype": torch_dtype,
        "low_cpu_mem_usage": True,
        "use_safetensors": True
    }
    
    # # Only try FlashAttention2 if CUDA is available
    # if torch.cuda.is_available():
    #     try:
    #         model_kwargs["attn_implementation"] = "flash_attention_2"
    #         print("Attempting to use FlashAttention2 for better performance...")
    #     except ImportError:
    #         print("FlashAttention2 not available, using default attention")
    
    # Load model
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        "openai/whisper-large-v3",
        **model_kwargs
    ).to(device)
    
    processor = AutoProcessor.from_pretrained("openai/whisper-large-v3")
    
    # Initialize pipeline
    asr_pipeline = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
        max_new_tokens=256
    )

    # Process each folder
    for folder_name in sorted(os.listdir(input_root)):
        if folder_name != 'audio_syncs':
            folder_path = os.path.join(input_root, folder_name)
            
            if not os.path.isdir(folder_path):
                continue
                
            print(f"\nProcessing {folder_name}...")
            
            # Find first audio file in folder
            audio_files = [f for f in sorted(os.listdir(folder_path))
                        if f.lower().endswith(('.mp3', '.wav', '.flac', '.m4a'))]
            
            if not audio_files:
                print(f"No supported audio files found in {folder_name}")
                continue
                
            audio_path = os.path.join(folder_path, audio_files[0])
            
            try:
                # Generate both word-level and segment-level transcriptions
                word_result = asr_pipeline(
                    audio_path,
                    return_timestamps="word",
                    generate_kwargs={"language": language, "task": "transcribe"}
                )
                
                segment_result = asr_pipeline(
                    audio_path,
                    return_timestamps=True,
                    generate_kwargs={"language": language, "task": "transcribe"}
                )
                
                # Save results with UTF-8 encoding
                with open(os.path.join(folder_path, microph_type + "_word_transcriptions.json"), 
                        'w', encoding='utf-8') as f:
                    json.dump(word_result, f, ensure_ascii=False, indent=2)
                
                with open(os.path.join(folder_path, microph_type + "_segment_transcriptions.json"), 
                        'w', encoding='utf-8') as f:
                    json.dump(segment_result, f, ensure_ascii=False, indent=2)
                
                print(f"Successfully processed {folder_name}")
                
            except Exception as e:
                print(f"Failed to process {folder_name}: {str(e)}")



# Get offsets between pupil Video recording and audio recording
def get_final_offsets_audio_video(general_recording_folder):
    files_in_folder = os.listdir(general_recording_folder)
    world_npy_files = sorted([os.path.join(general_recording_folder, file) for file in files_in_folder if file.startswith('world') and file.endswith('timestamps.npy')])
    audios_npy_files = sorted([os.path.join(general_recording_folder, file) for file in files_in_folder if file.startswith('audio') and file.endswith('timestamps.npy')])
    audio_beg = np.load(audios_npy_files[0])[0]
    audio_end = np.load(audios_npy_files[0])[-1]
    if len(world_npy_files)>1:
        np_list = []
        for path in world_npy_files:
            np_list.append(np.load(path))
        np_list = [np_list[-1]] + np_list[:-1] 
        video_beg = np_list[0][0]
        video_end = np_list[-1][-1]
        offset_beg = video_beg -audio_beg
        offset_end = audio_end - video_end
    else:
        video_beg = np.load(world_npy_files[0])[0]
        video_end = np.load(world_npy_files[0])[-1]
        offset_beg = video_beg - audio_beg 
        offset_end = audio_end - video_end

    return offset_beg, offset_end



#######################################################################
def get_relative_timestamps(folder_path, audio_timestamps_dir):
    """
    Get relative timestamps by subtracting the reference start time from world timestamps.
    
    Args:
        folder_path (str): Path to the participant folder
        audio_timestamps_dir (str): Directory containing audio_00010000_timestamps.npy files
        
    Returns:
        tuple: (start_ms, end_ms) in milliseconds or None if files not found
    """
    try:
        # Load world timestamps (in seconds)
        world_ts_path = os.path.join(folder_path, 'world_timestamps.npy')
        world_timestamps = np.load(world_ts_path)
        
        # load audio timestamp file
        audio_ref_path = os.path.join(audio_timestamps_dir, 'audio_00010000_timestamps.npy')
        
        # Load audio reference timestamp (in seconds)
        audio_ref_time = np.load(audio_ref_path)[0]  # Assuming it's a single value
        
        # Calculate relative timestamps
        relative_timestamps = world_timestamps - audio_ref_time
        
        # Convert to milliseconds and return first and last
        start_ms = int(relative_timestamps[0] * 1000)
        end_ms = int(relative_timestamps[-1] * 1000)
        
        return (start_ms, end_ms)
        
    except FileNotFoundError as e:
        print(f"Timestamp files not found for {folder_path}: {str(e)}")
        return None
    except Exception as e:
        print(f"Error processing timestamps in {folder_path}: {str(e)}")
        return None

def cut_audio(audio_path, timestamps, output_path):
    """
    Cut an audio file based on start and end timestamps (in milliseconds).
    """
    try:
        audio = AudioSegment.from_file(audio_path)
        start_ms, end_ms = timestamps
        trimmed_audio = audio[start_ms:end_ms]
        trimmed_audio.export(output_path, format="mp3")
    except Exception as e:
        print(f"Error cutting audio: {str(e)}")

def process_folders(input_root, audio_timestamps_dir, original_audio_path, output_root, microph_type):
    """
    Process all folders in the input directory, calculate relative timestamps,
    and cut the original audio accordingly.
    """
    # Create output root if it doesn't exist
    os.makedirs(output_root, exist_ok=True)
    
    # Process each folder
    for folder_name in os.listdir(input_root):
        folder_path = os.path.join(input_root, folder_name)
        if os.path.isdir(folder_path) and len(folder_name) == 3 and folder_name.isdigit():
    
            if not os.path.isdir(folder_path):
                continue
                
            print(f"\nProcessing folder: {folder_name}")
            
            # Get relative timestamps in milliseconds
            timestamps = get_relative_timestamps(folder_path, audio_timestamps_dir)
            if not timestamps:
                continue
                
            print(f"Calculated timestamps: start={timestamps[0]} ms, end={timestamps[1]} ms")
            
            # Create output folder
            output_folder = os.path.join(output_root, folder_name)
            os.makedirs(output_folder, exist_ok=True)
            
            # Cut and save audio
            output_path = os.path.join(output_folder, microph_type + '_trimmed_audio.mp3')
            cut_audio(original_audio_path, timestamps, output_path)
            print(f"Saved trimmed audio to {output_path}")


def sync_and_adjust_audio(video_audio: str, 
                         microph_audio: str, 
                         general_recording_folder: str, 
                         output_folder: str,
                         microph_type: str) -> str:
    """
    Synchronizes and adjusts audio between pupil and microphone recordings.
    
    Args:
        video_audio: Path to the pupil recording audio file
        microph_audio: Path to the microphone audio file
        general_recording_folder: Path to the main recording folder
        output_folder: Path where results should be saved
        
    Returns:
        Path to the final adjusted audio file
    """
    print("\nSTARTING AUDIO SYNC ##############################")
    
    # Create output directory
    destination_path = os.path.join(output_folder, 'audio_syncs')
    audio_name = microph_type +'_synced_audio_adjusted.mp3'
    os.makedirs(destination_path, exist_ok=True)
    adjusted_audio_path = os.path.join(destination_path, microph_type +'_synced_audio_adjusted.mp3')
    
    # Sync the audios
    offset_pupil = sync_audios(video_audio, microph_audio, destination_path)
    print('Audio aligning is complete')
    
    # Calculate offsets
    length_pupil_audio = get_audio_duration(video_audio)
    length_microphone_audio = get_audio_duration(microph_audio)
    offset_end = length_microphone_audio - (length_pupil_audio + offset_pupil)
    
    # Adjust audio with offsets
    adjust_audio_offset(microph_audio, adjusted_audio_path, offset_pupil, offset_end)
    offset_beg, offset_end = get_final_offsets_audio_video(general_recording_folder)
    adjust_audio_offset(adjusted_audio_path, adjusted_audio_path, offset_beg, offset_end)
    print('Audio adjusting is complete')
    
    # Print summary
    print(f"""
    Audio Synchronization Summary:
    - Microphone audio: {microph_audio} ({length_microphone_audio:.2f}s)
    - Pupil audio: {video_audio} ({length_pupil_audio:.2f}s)
    - Beginning offset: {offset_pupil:.2f}s
    - End offset: {offset_end:.2f}s
    - Final duration: {get_audio_duration(adjusted_audio_path):.2f}s
    """)
    
    # Clean up temporary files
    file_to_keep = audio_name
    for filename in os.listdir(destination_path):
        file_path = os.path.join(destination_path, filename)
        if os.path.isfile(file_path) and filename != file_to_keep:
            os.remove(file_path)
            print(f"Cleaned up: {filename}")
    
    return adjusted_audio_path



if __name__ == "__main__":
    if len(sys.argv)==6:
        general_recording_folder = sys.argv[1]
        video_audio = sys.argv[2]
        microph_audio = sys.argv[3]
        microph_type = sys.argv[4]
        output_folder = sys.argv[5]
        skip_sync = False


        ### AUDIO SYNC
        if microph_audio != video_audio:
            adjusted_audio_path = sync_and_adjust_audio(
                video_audio=video_audio,
                microph_audio=microph_audio,
                general_recording_folder=general_recording_folder,
                output_folder=output_folder,
                microph_type = microph_type)
            print(f"Final adjusted audio saved to: {adjusted_audio_path}")
        else:
            print("Skipping synchronization, using original audio")
            adjusted_audio_path = video_audio
            skip_sync = True


        ### AUDIO CUT corresponding to each extract
        INPUT_FOLDER =  os.path.join(general_recording_folder, 'exports') # Contains participant folders with world_timestamps.npy
        print("Starting audio processing...")
        process_folders(INPUT_FOLDER, general_recording_folder, adjusted_audio_path, output_folder, microph_type = microph_type)
        print("\nAudio cut completed!")
        

        # Delete the folder with adjusted audio
        if not skip_sync:
            folder_to_delete = os.path.dirname(adjusted_audio_path)
            try:
                if Path(folder_to_delete).name == 'audio_syncs':
                    shutil.rmtree(folder_to_delete)
                    print(f"Deleted audio folder: {folder_to_delete}")
                else:
                    print(f"!!!!!!!Folder to delete is not 'audio_syncs': {folder_to_delete}")
            except Exception as e:
                print(f"Failed to delete audio folder {folder_to_delete}: {e}")


        # AUDIO TRANSCRIPTIONS of corresponding audios
        INPUT_FOLDER = output_folder
        process_folders_for_transcription(INPUT_FOLDER, language="french", microph_type = microph_type)
        print("\nAll transcriptions for folders are processed!")

    else:
        print("Problem with arguments for audio processing script")











