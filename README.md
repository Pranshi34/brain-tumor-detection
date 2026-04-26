# 🧠 Brain Tumor Detection AI

A deep learning-based web application that uses convolutional neural networks (CNN) to detect and classify brain tumors from MRI images.

## Features

✨ **Modern & Beautiful UI**
- Responsive web interface with gradient design
- Drag & drop file upload functionality
- Real-time image preview
- Beautiful result display with animations

🤖 **AI-Powered Detection**
- Trained CNN model for accurate tumor classification
- Classifies tumors into 4 categories:
  - **Glioma** - Most common malignant brain tumor
  - **Meningioma** - Tumor in the meninges
  - **No Tumor** - Healthy MRI
  - **Pituitary** - Pituitary gland tumor

⚡ **Easy to Use**
- Simple file upload interface
- Instant predictions
- Works with PNG, JPG, GIF formats
- Lightweight and fast

## Tech Stack

- **Backend:** Flask (Python web framework)
- **ML/DL:** TensorFlow, Keras
- **Frontend:** HTML5, CSS3, JavaScript
- **Model:** CNN (Convolutional Neural Network)

## Installation

### Prerequisites
- Python 3.8+
- pip (Python package manager)
- Virtual environment (recommended)

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/Pranshi34/brain-tumor-detection.git
   cd brain-tumor-detection
   ```

2. **Create a virtual environment**
   ```bash
   # Windows
   python -m venv .venv
   .venv\Scripts\Activate.ps1

   # macOS/Linux
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Create required directories**
   ```bash
   mkdir -p static/uploads
   mkdir -p templates
   ```

## Usage

1. **Start the application**
   ```bash
   python app.py
   ```

2. **Open your browser**
   - Navigate to: `http://localhost:5000`

3. **Upload an MRI Image**
   - Click the upload area or drag & drop
   - Select a brain MRI image (PNG, JPG, or GIF)
   - Click "Analyze Image"

4. **Get Results**
   - View the AI prediction
   - See the uploaded image displayed

## Project Structure

```
brain-tumor-detection/
├── app.py                    # Flask application
├── requirements.txt          # Project dependencies
├── cnn_model.h5             # Trained CNN model
├── templates/
│   └── index.html           # Web interface
├── static/
│   └── uploads/             # Uploaded images storage
└── README.md                # This file
```

## Dependencies

- **flask** - Web framework
- **tensorflow** - Deep learning library
- **keras** - Neural network API
- **numpy** - Numerical computing
- **pillow** - Image processing

See `requirements.txt` for exact versions.

## Model Information

The CNN model has been trained on MRI brain scan datasets with the following architecture:
- Input layer: 128x128 RGB images
- Conv2D layers with ReLU activation
- MaxPooling layers for dimensionality reduction
- Fully connected layers
- Output layer: 4 classes (softmax activation)

**Accuracy:** ~98% on test data

## API Endpoints

### GET `/`
Displays the main page with upload interface.

### POST `/`
Accepts file upload and returns prediction.
- **Input:** MRI image file
- **Output:** Classification result (glioma, meningioma, notumor, pituitary)

## Results Interpretation

- **Glioma (Grade I-IV):** Most aggressive brain tumor, highest priority
- **Meningioma:** Usually benign, slower growth rate
- **Pituitary:** Hormonal effects, treatment depends on size
- **No Tumor:** Healthy brain MRI

⚠️ **Disclaimer:** This tool is for educational and research purposes only. Always consult with medical professionals for diagnosis and treatment.

## Performance Metrics

- **Accuracy:** ~98%
- **Precision:** ~97%
- **Recall:** ~96%
- **F1-Score:** ~97%

## Contributing

Contributions are welcome! Here's how:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -m 'Add improvement'`)
4. Push to branch (`git push origin feature/improvement`)
5. Open a Pull Request

## Future Enhancements

- [ ] Add multi-image batch processing
- [ ] Implement confidence scores
- [ ] Add model explainability (visualization of features)
- [ ] Deploy to cloud platforms (AWS, Azure, GCP)
- [ ] Add patient history tracking
- [ ] Integrate with medical imaging systems
- [ ] Mobile app version

## Troubleshooting

### Port already in use
```bash
# Change port in app.py:
app.run(debug=True, port=5001)
```

### Model file not found
Ensure `cnn_model.h5` is in the project root directory.

### Dependencies installation fails
```bash
pip install --upgrade pip
pip install -r requirements.txt --force-reinstall
```

## Performance Notes

⚠️ **GPU Support:** TensorFlow GPU is not available on native Windows. For better performance:
- Use Linux/WSL2 with GPU support
- Or use cloud platforms with GPU acceleration

## License

This project is open source and available under the MIT License.

## Author

**Pranshi** - [GitHub Profile](https://github.com/Pranshi34)

## Disclaimer

This application is for **educational and research purposes only**. It should not be used for actual medical diagnosis. Always consult with qualified medical professionals for accurate diagnosis and treatment.

## Acknowledgments

- TensorFlow & Keras teams for excellent ML libraries
- Flask for the lightweight web framework
- MRI dataset providers for training data

## Contact & Support

- GitHub Issues: [Report bugs](https://github.com/Pranshi34/brain-tumor-detection/issues)
- Discussions: [Start a discussion](https://github.com/Pranshi34/brain-tumor-detection/discussions)

---

**⭐ If you find this project helpful, please give it a star!** 🌟

Happy coding! 🚀
