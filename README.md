# Few Shot Verification

A deep learning project implementing a Siamese Neural Network to verify image similarity using few-shot learning techniques. It is capable of verifying whether two images belong to the same category with minimal training data per class.

## Project Structure

- `backend/`: Contains the PyTorch machine learning models, training scripts, loss functions, and API logic.
- `frontend/`: The React-based frontend application for uploading images and viewing verification results.
- `datasets/`: The directory where local datasets can be placed for testing or training.

## Usage

### Training (Kaggle Recommended)
This model is trained on the Labeled Faces in the Wild (LFW) dataset. To train on Kaggle:
1. Zip your local project folder and upload it to Kaggle as a Dataset.
2. Create a new Kaggle Notebook and attach your uploaded dataset.
3. Run the following script in a notebook cell to fetch LFW, load the code, and train for 20 epochs:

   ```python
   import os
   import shutil
   import kagglehub

   lfw_path = kagglehub.dataset_download("jessicali9530/lfw-dataset")
   images_dir = os.path.join(lfw_path, "lfw-deepfunneled", "lfw-deepfunneled")

   backend_path = None
   for root, dirs, files in os.walk('/kaggle/input'):
       if 'run.py' in files:
           backend_path = root
           break

   shutil.copytree(backend_path, '/kaggle/working/backend', dirs_exist_ok=True)
   os.chdir('/kaggle/working/backend')

   !python run.py --data_dir {images_dir} --epochs 20 --batch_size 32
   ```

4. Once completed, download `siamese_model.pth` from the Kaggle output.

### Local Inference (Manual)
1. Place your trained `siamese_model.pth` into the `backend/` directory.
2. Navigate to `backend/` and start the server:
   ```bash
   uvicorn app:app --reload
   ```
3. Navigate to `frontend/` and start the UI:
   ```bash
   npm run dev
   ```

### Running with Docker
The easiest way to run the entire application locally is using Docker Compose.
1. Place your trained `siamese_model.pth` into the `backend/` directory.
2. Make sure Docker Desktop is running on your machine.
3. Run the following command from the root project directory:
   ```bash
   docker-compose up --build -d
   ```
4. The frontend UI will be available at `http://localhost:5173` and the API at `http://localhost:8000`.

## Technologies
- **PyTorch**: Deep learning framework
- **React**: Web Interface
- **FastAPI**: Backend API
