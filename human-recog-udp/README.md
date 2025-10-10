# Human Recognition and UDP Communication Project

This project implements a human detection system using the YOLO model and facilitates communication via UDP. It consists of multiple components that work together to detect humans in video frames, remove backgrounds, and send processed images to a Unity application.

## Project Structure

```
human-recog-udp
├── src
│   ├── humanRecog.py        # Contains the HumanDetector class for human detection and background removal.
│   ├── posetest_UDP.py      # Handles UDP communication for sending and receiving data.
│   └── run_both.py          # Runs both humanRecog.py and posetest_UDP.py simultaneously.
├── scripts
│   └── start_both.sh        # Shell script to start both Python scripts concurrently.
├── requirements.txt          # Lists the required Python dependencies.
├── .gitignore                # Specifies files and directories to be ignored by Git.
└── README.md                 # Documentation for the project.
```

## Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd human-recog-udp
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

To run the human detection and UDP communication scripts simultaneously, you can use the provided shell script:

```bash
./scripts/start_both.sh
```

Alternatively, you can run the scripts individually:

1. Start the human detection script:
   ```bash
   python src/humanRecog.py
   ```

2. Start the UDP communication script:
   ```bash
   python src/posetest_UDP.py
   ```

## Dependencies

This project requires the following Python packages:

- OpenCV
- NumPy
- ultralytics (for YOLO model)
- torch (for PyTorch)

Make sure to install these packages using the `requirements.txt` file.

## Notes

- Ensure that your camera is connected and accessible for the human detection script to function properly.
- Modify the IP address and port in `humanRecog.py` if you need to send data to a different Unity instance.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.