## How to Run the Code

1. Set the path of the export directory of the eye-tracking folder into the `default_input` variable.
2. Specify the `phase_group` variable with one of the following values: `'A'`, `'B'`, or `'2'`.
3. Update the `tableau_order` variable only if the order of paintings has changed. Refer to the provided Word document for the correct order.
4. Run the code.

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