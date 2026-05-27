# Storyboard Generation

An agent for generating anime-style visual storyboards from a natural language prompt.

Users can enter a story idea, generate storyboard frames, view the generated results, and check previous generation history through a browser-based interface.

---

## GitHub Repository

```text
https://github.com/Zhedong-Lin/Visual_Story_Generation
```

---

## Demo Video

Add the demo video link here:

```text
https://drive.google.com/file/d/1EBKKVUzVGaJlIF72s4a368dG4nHvmmlU/view?usp=drive_link
```

---

## What This App Does

This project provides a web interface called **AniBoard**.

With AniBoard, users can:

- Enter a natural language story prompt.
- Generate an anime-style visual storyboard.
- View the generated frames in the browser.
- Check previous generation records in the history page.
- Run the system locally or on a remote server.

The system is designed for visual story generation. It takes a user prompt, processes the request, plans the generation steps, and produces storyboard images.

---

## Requirements

Before running the project, make sure you have:

- Python 3.10
- Conda or Miniconda
- Git
- A terminal or command line environment

If you run the project on a remote server through VS Code SSH, you also need to forward the web server port to your local machine.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/Zhedong-Lin/Visual_Story_Generation.git
cd Visual_Story_Generation
```

Create and activate a Conda environment:

```bash
conda create -n storyboard-generation python=3.10 -y
conda activate storyboard-generation
```

Install the project:

```bash
pip install -e .
```

---

## Run the Web App

Start the web application with:

```bash
python app.py --host 127.0.0.1 --port 8000
```

After the server starts, open the following address in your browser:

```text
http://127.0.0.1:8000
```

If the project is running correctly, the AniBoard homepage will appear.

---


## How to Use the Web Interface

### 1. Open the Homepage

After opening the local web page, you will see the AniBoard homepage.

Click the start button or navigate to the workspace page to begin creating a storyboard.

### 2. Enter a Story Prompt

Type a natural language story prompt into the input box.

Example:

```text
Lulu is sleeping on the bed. She suddenly wakes up from a nightmare. Then she cries in fear.
```

You can describe:

- The main character
- The scene
- The action
- The mood
- The story events
- The expected number of frames

### 3. Generate the Storyboard

Click the generate button.

The system will process the prompt and generate storyboard frames.

### 4. View the Output

After generation finishes, the generated storyboard images will be displayed on the results page.

Each image corresponds to one part of the input story.

### 5. Check Generation History

The history page shows previous generation records, including:

- Run ID
- Prompt
- Number of frames
- Generation time

These history records are created during runtime and are not included in the GitHub repository.

---

## Quick Demo without API Keys

The project also provides a dry-run script for quick testing:

```bash
python scripts/run_demo_dry.py
```

Dry-run mode:

- Does not require API keys.
- Uses mock LLM and mock image providers.
- Generates sample output files.
- Saves intermediate results under the `runs/` directory.

This mode is useful for checking whether the system can run correctly on your machine.



---



## Output Files

Generated results are saved under:

```bash
runs/<run_id>/
```

A run folder may include:

- Generated storyboard images
- Intermediate JSON files
- Planning results
- Execution logs
- Graph visualization files

The `runs/` directory is ignored by Git because it contains runtime outputs.

---


## Notes

- Runtime outputs such as generated images and history records are not included in the GitHub repository.
- API keys should be stored locally in a `.env` file and should not be uploaded to GitHub.
- If local image generation models are used, make sure the machine has enough GPU memory.
- Some Python warnings may appear when starting the server. They can usually be ignored if the web page opens normally.

---

## Project Summary

This project is an intelligent visual storyboard generation system. It receives a user story prompt, plans the generation process, executes the required steps, and produces storyboard images through a web interface.
