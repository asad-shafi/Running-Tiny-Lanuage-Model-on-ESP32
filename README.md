# Running-Tiny-Lanuage-Model-on-ESP32
This project demonstrates the deployment of a Tiny Language Model for Natural Language understanding on an ESP32-S3, compiled using the SynapEdge, proving that sophisticated, responsive AI can run on low-cost, resource-constrained devices entirely offline, enabling a new generation of private, low-latency, and power-efficient applications at the edge.

## Model
I have built a custom Language model and trained it on "noanabeshima/TinyStoriesV2". This was an easily available dataset, so I picked it up for demonstration. I am looking for real-world applications and data sets, such as IoT, Sensor readings, and alerts. If you have such a dataset, please share.

## ✨ Key Features
 - 🛜 Fully Offline: No cloud connection required. All inference happens locally on the ESP32.
 - 🔒 Privacy-First: User data never leaves the device.
 - ⚡ Ultra-Low Power: Designed to run on battery-powered microcontrollers for extended periods.
 - 💰 Cost-Effective: Leverages affordable, ubiquitous ESP32 hardware.
 - 🛠️ Hardware Agnostic: The SynapEdge toolchain converts ONNX models to portable C code, making deployment on other microcontrollers straightforward.

## 🚀 How It Works
The SynapEdge compiler takes a pre-trained Tiny Language Model in the ONNX format and generates highly efficient, pure ANSI C code. This generated code is then compiled and flashed onto the ESP32-S3 or any other microcontroller.
- Model Preparation: A TLM is trained or sourced and exported to ONNX.
- Compilation: The SynapEdge compiler converts the ONNX model into optimized C code.
- Deployment: The generated code is integrated into an ESP-IDF project and flashed to the device.
- Inference: The ESP32 runs the model to perform tasks like voice command recognition or intent classification.

## 📋 Hardware Requirements
- An ESP32-S3 development board (e.g., ESP32-S3-DevKitC-1)
- USB cable for programming and power
- (Optional) Microphone module for voice input

🧩 Software Requirements
- SynapEdge Compiler
- Arduino IDE

<a href="https://www.hackster.io/asadshafi5/run-tiny-language-model-on-esp32-8b5dd8"> Visit this Project</a>

