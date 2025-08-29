# AMSTRA: Autonomous/Manual Spatial Tagging & Reconnaissance Application

<p align="center">
  <img src="resources/gifs/recordings.gif" alt="AMSTRA Demo">
  <br>
  <em>Demonstration of the AMSTRA framework in action.</em>
</p>

## Description

AMSTRA is a designed framework for autonomous and manually controlled drone navigation with real-time object detection and spatial tagging modules. It leverages YoloV8 and SORT-based manager to identify, and track objects of interest. A separate triangulation module uses monocular camera stream to attempt 3D spatial tagging of detected objects. The system is designed for real-time operation, featuring a server-client architecture, a control dashboard, and simulation capabilities.

The project uses Webots as the main simulator due to easy prototyping and access to pre-installed Mavic 2 pro drone model. The simulator can be downloaded [here](https://cyberbotics.com/#download).

## Project Structure

```
/
├── modules/             # Core Python modules for the system.
├── resources/           # Images and GIFs for documentation.
├── webots/              # Webots simulation environment.
├── .gitignore           # Git ignore file.
├── README.md            # This README file.
└── requirements.txt     # Python dependencies.
```

## Server System

![Server Design Diagram](resources/images/Server%20Design%20Diagram.png)

The AMSTRA server architecture offloads heavy computation from the Mavic 2 Pro to a ground station. Frames are transmitted via **User Datagram Protocol (UDP)** and processed in real time using a dual-buffer system. A **motion gate** compares new and old frames to decide whether to trigger object detection or rely on tracking. Detection results and navigation data are output separately as JSON for downstream modules.

### Processing Flow

- **Frame Ingestion** → Receive frames from drone via UDP.
- **Buffering** → Store frames in short-term (motion gate) and long-term (overflow) buffers.
- **Motion Gate Check**
  - Compare newest frame (Queue) with oldest frame (short-term).
  - Evaluate using **SSIM**, **absolute difference**, or **histogram difference**.
- **Frame Handling**
  - If sufficient change → send to **YOLO detection**.
  - If not → forward to **SORT tracker** for predicted tracks.
- **Tracking Update** → Use YOLO results to update SORT and refine tracks.
- **Output**
  - Write **navigation data** (K, R, t) to JSON.
  - Write **tracking data** (detections + tracks) to separate JSON.

### Server-Client Dashboard

To make the server simpler to use and invariant to user's experience, `server.py` incorporates simple `rich`-based dashboard for easy viewing of incoming streams.

![Server Dashboard](resources/images/Ground%20Station%20Dashboard.png)

In addition, the autonomous and manual webot controller scripts, `mavic2pro_geotag_auto.py` and `mavic2pro_geotag.py` respectively, have simple print based UI or control hints. Rich-based dashboard could not be incorporated since it could not be displayed via Webots simulator's std output.

<p align="center">
  <img src="resources/images/Drone Controller Dashboard.png" alt="AMSTRA Controller Dashboard">
  <br>
  <em>AMSTRA Drone Controller Dashboard (Autonomous Mode).</em>
</p>

<p align="center">
  <img src="resources/images/Drone Controller Dashboard Manual.png" alt="AMSTRA Manual Controller">
  <br>
  <em>AMSTRA Drone Controller Dashboard (Manual Mode).</em>
</p>

## SORT-based Tracking

AMSTRA employs a **SORT-based tracking manager** built on top of an extended **Kalman Filter** framework. This tracker maintains consistent identities for objects across frames, even when detections are missing.

- **Kalman Filter Model**  
  Tracks are represented with state vectors containing bounding box center, scale, and aspect ratio, along with velocity terms. The filter predicts object motion frame-to-frame and updates states when new detections are available.

- **Track Management**

  - **Association:** Uses **Hungarian matching** with IoU as the cost metric to associate detections with existing tracks.
  - **Unmatched Tracks:** If a detection is missing, tracks are propagated forward using only Kalman predictions.
  - **New Tracks:** New detections above a confidence threshold spawn new tracks if they are not too close to existing ones.
  - **Aging:** Tracks expire if not updated after a configurable number of frames (`max_age`).

- **Robustness Features**
  - Confidence thresholding avoids spurious tracks from low-confidence detections.
  - IoU-based filtering ensures overlapping detections do not spawn duplicate tracks.
  - The filter dynamically adjusts its covariance and measurement noise to stabilize predictions during object scale or aspect ratio changes.

## Triangulation Module

The triangulation subsystem attempts to recover **3D spatial positions** of tracked objects using only the monocular camera feed and drone pose (K, R, t). It consists of two modes:

1. **Frame-to-Frame Triangulation (F2F)**

   - Extracts ORB features within detection bounding boxes.
   - Matches features between two buffered frames and computes 3D points using **triangulatePoints**.
   - Associates reconstructed points with bounding boxes by checking if feature projections fall within their regions.

2. **Bundle Adjustment Triangulation (BA)**
   - Maintains a longer buffer of frames and feature tracks across multiple viewpoints.
   - Initializes 3D landmarks by triangulating between frame pairs with sufficient parallax.
   - Refines landmarks with **bundle adjustment (least-squares reprojection minimization)**.
   - Applies **cheirality** (points in front of both cameras) and **parallax filters** (minimum angular separation) to reject unstable points.

- **Outputs**
  - Produces a JSON log of 3D points associated with tracked bounding boxes.

## Limitations

While AMSTRA benefits from real-time processing via separation of computing responsibilities, the inherent limitations prevent AMSTRA from performing more robustly in challenging circumstances. Specifically, Mavic 2 Pro drones feature **monocular camera streams**, which limits triangulation accuracy and SORT-based tracking management.

- If the camera or drone rotates the view too abruptly, SORT attempts to predict how bounding boxes (bboxes) should move to compensate.
- Without reliable **depth estimation**, triangulation predictions often drift or break down.
- Monocular input cannot easily resolve scale or distance in real time, leading to unrealistic adjustments by the SORT manager and triangulation results with oscillating magnitudes.

### Key Challenges

- [ ] **Monocular Triangulation** → Depth cannot be directly inferred from a single stream, reducing localization accuracy.
- [ ] **Rapid Motion** → Abrupt rotations or shifts cause tracker drift due to lack of 3D scene understanding.
- [ ] **SORT Limitations** → Assumes relatively smooth motion; struggles under perspective changes without depth cues.

## Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/CodeKnight314/AMSTRA.git
    ```
2.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1.  Start the server:
    ```bash
    python modules/server.py
    ```
2.  Launch the simulation environment in Webots.
3.  Open the dashboard to monitor and control the system.
