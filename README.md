# A11yShape

## Changes made in this fork
This redux makes several modifications to the original code.

### Backend
- Refactored main program to store API keys and model definitions in the environment file instead of hardcoded authorization headers
- Added support for an OpenAI-compatible API base URL override
- Updated `requirements.txt` to use an updated version of Flask to avoid dependency on a deprecated version of jinja2
- Updated mode detection logic

### Interface
- Responsive CSS used instead of per-div styling
- Added a manual switcher for describe/modify modes, with a default automatic mode
- Added image carousel/tile view image viewer allowing access to all rendered views of the model
- Added light/dark mode toggle

### Accessibility
- Added automatic prerequisite installation scripts for windows and macOS
- Increased verbosity of errors displayed in the errors panel

### AI pipeline
- Removed hardcoded OpenSCAD code responses to certain prompts bypassing the LLM generation pipeline
- Added deduplication for rendered images from different views using sha256 hashes before uploading to the LLM to save tokens
- Updated default models to GPT-5 line for improved performance
- Changed prompting for describe/modify modes for improved adherence and output
- Expanded API responses and logging

## Setup
1. Clone the repository to a local folder.
2. Run `setup_windows.bat` (windows) or `setup_macos.sh` (macOS) to install Python and OpenSCAD (if not already installed and accessible in PATH), and the project dependencies via pip.
3. Rename the `.env.example` file to `.env` and change the value of `OPENAI_API_KEY` to your OpenAI API key. (For third-party model providers, change `OPENAI_BASE_URL` to the model provider's OpenAI-compatible base URL, and use your API token as the `OPENAI_API_KEY`).
4. Open `app.py` (or run `python app.py` in a terminal opened to the project directory). a new tab should open to http://localhost:3000/ in your default browser.
