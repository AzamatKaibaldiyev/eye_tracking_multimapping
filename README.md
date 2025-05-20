## How to Run the Code

1. Open the `runner_pl_full_processing.py` file, and set the path of the export directory of the eye-tracking folder into the `default_input` variable.
2. Specify the `phase_group` variable with one of the following values: `'A'`, `'B'`, or `'2'`.
3. Update the `tableau_order` variable only if the order of paintings has changed. Refer to the provided Word document for the correct order.
4. Run the code.

Create a new Python virtual environment and install the required libraries from the `requirements.txt` file. Use the following commands:
```bash
python -m venv script_env
source script_env/bin/activate  # On Windows, use `env\Scripts\activate`
pip install -r requirements.txt
```
This ensures all dependencies are installed in an isolated environment.


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