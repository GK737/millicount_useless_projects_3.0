<img width="898" height="234" alt="Screenshot 2026-09-04 080715" src="https://github.com/user-attachments/assets/8862d297-2385-4103-8f54-a4e6f11615d9" /><img width="1280" height="640" alt="git (1)" src="https://github.com/user-attachments/assets/8920b256-2ba8-4988-b824-5351134eb4bd" />



# [MILLICODE]🪱 


## Basic Details
### Team Name: [Latverians]


### Team Members
- Team Lead: Gopikrishna R S - Muthoot Institute Of Technology and Science
- Member 2: Ebin Jose - Muthoot Institute of Technology and Science
  

### Project Description
 Millicount is a simple image-processing project designed to count the body segments (rings) of a millipede from an image or live feed.

### The Problem (that doesn't exist)
Millipedes curl on itself to count their legs

### The Solution (that nobody asked for)
1) The project uses basic computer vision techniques to identify and count the segments.
2) Provide the image to a chatbot connected via api keys who does the hard work.
   
## Technical Details
### Technologies/Components Used
For Software:
Languages
- Python (3.10+)

Frameworks
- Streamlit — quick web app/frontend for demos and UI (pip: streamlit)
- Ultralytics YOLO / YOLO-OBB — deep-learning model & training/inference toolkit (pip: ultralytics or custom YOLO-OBB package)

Libraries
- OpenCV (cv2) — classical computer vision operations, image I/O (pip: opencv-python / opencv-contrib-python)
- NumPy — numerical arrays and vectorized ops (pip: numpy)
- Pandas — tabular data handling (pip: pandas)
- Matplotlib — plotting and visualization (pip: matplotlib)
- PIL / Pillow — image I/O and lightweight image transforms (pip: pillow)
- SciPy — signal-processing tools used in classical CV pipeline (pip: scipy)

Tools / Services
- providers.py — abstraction layer for vision-capable AI APIs (can be configured to use cloud providers or custom endpoints)
- Git / GitHub — version control and hosting

### Implementation

For Software:

1. Clone & create venv
   - git clone https://github.com/GK737/millicount_useless_projects_3.0.git
   - cd millicount_useless_projects_3.0
   - python -m venv .venv
   - source .venv/bin/activate    # Windows: .venv\Scripts\activate

2. Install dependencies
   - pip install --upgrade pip
   - pip install -r requirements.txt
   - (If you don't have a requirements.txt yet, see the example below.)

3. Environment configuration
   - Copy .env.example -> .env and set any keys needed by providers.py (API keys, endpoints).
   - Example entries:
     - PROVIDER_API_KEY=your_api_key_here
     - STREAMLIT_SERVER_PORT=8501

4. Run the Streamlit app (development)
   - streamlit run app_v3.py inside Vision_Based_Model or api_based_app.py In API_Based_Model
   - Or if the main app is under src/: streamlit run src/main.py
   - Open http://localhost:8501

5. Deep-learning pipeline (Ultralytics YOLO)
   - To run inference using Ultralytics:
     - python -c "from ultralytics import YOLO; model = YOLO('yolov8n.pt'); results = model('path/to/image.jpg')"
   - For YOLO-OBB usage, follow the specific training/inference instructions for the OBB fork/package you use (model weights, custom dataset YAML, augmentation).

6. Classical CV pipeline
   - Core flow:
     - Load image via OpenCV / Pillow
     - Preprocess (resize, normalize, denoise using SciPy filters or OpenCV)
     - Feature extraction / edge detection / morphological ops (OpenCV + SciPy)
     - Postprocess detections (NMS, bounding box refinement; use SciPy signal tools for filtering noisy signals)
     - Visualize with Matplotlib or Streamlit image display

7. Provider abstraction (providers.py)
   - providers.py exposes a common interface to call vision APIs (send image bytes or URLs)
   - Implement provider classes for each API (e.g., GoogleVisionProvider, AzureVisionProvider, CustomProvider)
   - Configure active provider via .env or config.json; code should fall back to local inference if no API keys present.




### Project Documentation
For Software:

# Screenshots
<img width="947" height="434" alt="Screenshot 2026-09-04 043154" src="https://github.com/user-attachments/assets/2fc31511-2861-4836-807c-a12cacb80ff3" />
*Vision Based Model Actively counting the number of segments of a millipede*

<img width="959" height="539" alt="Screenshot 2026-09-04 020649" src="https://github.com/user-attachments/assets/6883847e-5f5d-4559-a2ee-b40e16b5c5f0" />
*Api based model predicting the number of segments from an uploaded image*

<img width="898" height="234" alt="Screenshot 2026-09-04 080715" src="https://github.com/user-attachments/assets/b9491c9f-7e48-40f9-a38d-7e3776456c22" />
*Model Correctly Identifies non millipede objects*

# Diagrams
![Workflow](Add your workflow/architecture diagram here)
*Add caption explaining your workflow*


### Project Demo
# Video
[Add your demo video link here]
*Explain what the video demonstrates*



## Team Contributions
- Ebin Jose: Documentation/Prompting
- Gopikrishna R S: Coding/Debugging 


---
Made with ❤️ at TinkerHub Useless Projects 

![Static Badge](https://img.shields.io/badge/TinkerHub-24?color=%23000000&link=https%3A%2F%2Fwww.tinkerhub.org%2F)
![Static Badge](https://img.shields.io/badge/UselessProjects--26-26?link=https%3A%2F%2Ftinkerhub.org%2Fevents%2F1M8ORET9A1%2Fuseless-projects-3.0)



