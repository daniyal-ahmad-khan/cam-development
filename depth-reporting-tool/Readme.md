# OAK-D Depth Reporting Tool

This application uses an OAK-D camera mounted on the front of a vehicle to detect objects, measure their distances from the vehicle, and report the results through real-time graphical visualizations and CSV outputs.

## Prerequisites

Before you begin, make sure you have:
- Python 3.8 or higher installed.
- An OAK-D camera installed and configured on your vehicle.

## Installation

### Clone the Repository

Start by cloning the repository to your local machine:

```bash
git clone https://example.com/depth-reporting-tool.git
cd depth-reporting-tool
```

### Create and Activate a Python Virtual Environment
It is recommended to use a Python virtual environment to manage dependencies:

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows
venv\Scripts\activate
# On Unix or MacOS
source venv/bin/activate
```

### Install Required Packages
Install all necessary dependencies:

```bash
pip install -r requirements.txt
```
### Usage
To start the application, follow these steps:

1. Run the Main Script:

    * Launch the tool by running the main script:
    
    ```bash
    python main.py
    ```
2. View Outputs

    * The detections and calculated distances are recorded in `detection_data/detections.csv`.
    * Graphical visualizations are saved in `detection_data/detections.html`. Open this file in any web browser to view the results.