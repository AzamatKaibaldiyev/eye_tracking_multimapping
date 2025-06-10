## Install dependecies

Create a new Python virtual environment and install the required libraries from the `requirements.txt` file. Use the following commands:
```bash
python -m venv script_env
source script_env/bin/activate  # On Windows, use `env\Scripts\activate`
pip install -r requirements.txt
```
This ensures all dependencies are installed in an isolated environment.


## How to Run the Code

1. Open the `runner_pl_full_processing.py` file
2. In Check_steps_notebook.ipynb (Extracting paths and group numbers) put folder_path and run it to get arguments automatically, copy past them into input_folder_group variable in  `runner_pl_full_processing.py` file 
3. Update the `tableau_order` variable only if the order of paintings has changed. Refer to the provided Word document for the correct order.
4. Run the code.
5. Check for missing data after processing by running Check_steps_notebook.ipynb (Checking if there are no missing files after processing)
6. If found, copy paste the output from (If there are missing data code part) into run_missing_data.py and run it


### Functionality
The code performs the following tasks:
- Synchronizes audio files (if present).
- Projects the eye-tracking data from the `world_view` into the corresponding painting.
- Outputs the following files for each painting folder:
    - `fixations.tsv`: Contains fixation data.
    - `gazeData_mapped.tsv`: Contains mapped gaze data.
    - Corresponding audio files (telephone and/or microphone).
    - Word and sentence-level transcriptions of the audio files.
    - An image of the corresponding painting.

Ensure all required inputs are correctly set before running the code.