"""
Face Recognition API - FastAPI Backend
Supports enrollment and recognition of existing users
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import cv2
import numpy as np
from scipy.spatial.distance import cosine
import base64
from insightface.app import FaceAnalysis
import os
from datetime import datetime
import json

# =========================
# APP INITIALIZATION
# =========================
app = FastAPI(
    title="Face Recognition API",
    description="Face enrollment and recognition system",
    version="1.0.0"
)

# CORS - Allow frontend to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your Netlify domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# CONFIGURATION
# =========================
MIN_VALID_FRAMES = 3
DUPLICATE_THRESHOLD = 0.35  # cosine distance
RECOGNITION_THRESHOLD = 0.40  # cosine distance

# =========================
# FACE MODEL INITIALIZATION
# =========================
print("Loading face recognition model...")
face_app = FaceAnalysis(name="buffalo_l")
ctx_id = 0 if os.environ.get('GPU_ENABLED', 'false').lower() == 'true' else -1
face_app.prepare(ctx_id=ctx_id, det_size=(640, 640))
print(f"Model loaded successfully (GPU: {ctx_id >= 0})")

# =========================
# IN-MEMORY STORAGE
# Replace with your PostgreSQL database
# =========================
enrolled_faces = {}

# Load from file if exists (for persistence)
STORAGE_FILE = "enrolled_faces.json"

def load_enrolled_faces():
    """Load enrolled faces from file"""
    global enrolled_faces
    if os.path.exists(STORAGE_FILE):
        try:
            with open(STORAGE_FILE, 'r') as f:
                data = json.load(f)
                # Convert embeddings back to numpy arrays
                for name, face_data in data.items():
                    face_data['embedding'] = np.array(face_data['embedding'])
                    enrolled_faces[name] = face_data
                print(f"Loaded {len(enrolled_faces)} enrolled faces")
        except Exception as e:
            print(f"Error loading enrolled faces: {e}")

def save_enrolled_faces():
    """Save enrolled faces to file"""
    try:
        # Convert numpy arrays to lists for JSON serialization
        data = {}
        for name, face_data in enrolled_faces.items():
            data[name] = {
                **face_data,
                'embedding': face_data['embedding'].tolist()
            }
        with open(STORAGE_FILE, 'w') as f:
            json.dump(data, f)
    except Exception as e:
        print(f"Error saving enrolled faces: {e}")

# Load on startup
load_enrolled_faces()

# =========================
# REQUEST/RESPONSE MODELS
# =========================

class EnrollRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    images: List[str] = Field(..., min_items=1, max_items=20)

class RecognizeRequest(BaseModel):
    image: str = Field(..., description="Base64 encoded image")

class EnrollResponse(BaseModel):
    success: bool
    message: str
    name: Optional[str] = None
    images_used: Optional[int] = None
    enrolled_count: int
    best_image: Optional[str] = None

class RecognitionResponse(BaseModel):
    recognized: bool
    name: Optional[str] = None
    confidence: Optional[float] = None
    message: str
    enrolled_image: Optional[str] = None

class EnrolledFace(BaseModel):
    name: str
    image: str
    images_used: int
    enrolled_at: str

# =========================
# UTILITY FUNCTIONS
# =========================

def base64_to_image(base64_string: str) -> Optional[np.ndarray]:
    """Convert base64 string to OpenCV image"""
    try:
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        
        img_data = base64.b64decode(base64_string)
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print(f"Error decoding image: {e}")
        return None

def image_to_base64(img: np.ndarray) -> str:
    """Convert OpenCV image to base64 string"""
    _, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return base64.b64encode(buffer).decode('utf-8')

def normalize(vec: np.ndarray) -> np.ndarray:
    """Normalize embedding vector"""
    vec = np.array(vec, dtype=np.float32)
    norm = np.linalg.norm(vec)
    return vec if norm == 0 else vec / norm

def validate_face(face, frame) -> tuple:
    """
    Validate face quality
    Returns: (is_valid, error_message)
    """
    try:
        x1, y1, x2, y2 = map(int, face.bbox)
        h, w = frame.shape[:2]
        
        # Bounds check
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        if x2 <= x1 or y2 <= y1:
            return False, "Invalid face bounding box"
        
        face_w = x2 - x1
        face_h = y2 - y1
        
        # Size check
        min_dimension = min(w, h)
        min_face_size = max(60, min_dimension * 0.10)
        
        if face_w < min_face_size or face_h < min_face_size:
            return False, "Face too small"
        
        # Blur check
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return False, "Invalid face crop"
        
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        blur = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        if blur < 50:
            return False, "Face too blurry"
        
        return True, None
        
    except Exception as e:
        return False, str(e)

# =========================
# API ENDPOINTS
# =========================

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "Face Recognition API",
        "version": "1.0.0",
        "status": "running",
        "enrolled_count": len(enrolled_faces)
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model": "buffalo_l",
        "enrolled_count": len(enrolled_faces),
        "gpu_enabled": ctx_id >= 0
    }

@app.post("/api/enroll", response_model=EnrollResponse)
async def enroll_face(request: EnrollRequest):
    """
    Enroll a new face
    
    - Processes multiple images
    - Validates face quality
    - Checks for duplicates
    - Stores face embedding
    """
    try:
        name = request.name.strip()
        images = request.images
        
        if not name:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Name is required"
            )
        
        # Check if name already exists
        if name in enrolled_faces:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Name '{name}' is already enrolled"
            )
        
        embeddings = []
        best_image = None
        best_quality = 0
        validation_errors = []
        
        # Process each image
        for idx, img_b64 in enumerate(images):
            img = base64_to_image(img_b64)
            
            if img is None:
                validation_errors.append(f"Image {idx}: Invalid format")
                continue
            
            # Detect faces
            faces = face_app.get(img)
            
            if len(faces) == 0:
                validation_errors.append(f"Image {idx}: No face detected")
                continue
            
            if len(faces) > 1:
                validation_errors.append(f"Image {idx}: Multiple faces detected")
                continue
            
            face = faces[0]
            
            # Validate face quality
            is_valid, error_msg = validate_face(face, img)
            if not is_valid:
                validation_errors.append(f"Image {idx}: {error_msg}")
                continue
            
            # Extract embedding
            embedding = normalize(face.embedding)
            embeddings.append(embedding)
            
            # Track best quality image
            quality = face.det_score
            if quality > best_quality:
                best_quality = quality
                best_image = img_b64
        
        # Check if we have enough valid images
        if len(embeddings) < MIN_VALID_FRAMES:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Not enough valid face images. Got {len(embeddings)}, need {MIN_VALID_FRAMES}. Errors: {validation_errors}"
            )
        
        # Calculate average embedding
        avg_embedding = normalize(np.mean(embeddings, axis=0))
        
        # Check for duplicates against existing enrollments
        for enrolled_name, enrolled_data in enrolled_faces.items():
            stored_embedding = enrolled_data['embedding']
            distance = cosine(avg_embedding, stored_embedding)
            
            if distance < DUPLICATE_THRESHOLD:
                similarity = (1 - distance) * 100
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail=f"Face already enrolled as '{enrolled_name}' ({similarity:.1f}% match)"
                )
        
        # Save enrollment
        enrolled_faces[name] = {
            'embedding': avg_embedding,
            'image': best_image,
            'images_used': len(embeddings),
            'enrolled_at': datetime.utcnow().isoformat()
        }
        
        # Persist to file
        save_enrolled_faces()
        
        return EnrollResponse(
            success=True,
            message=f"Successfully enrolled {name}",
            name=name,
            images_used=len(embeddings),
            enrolled_count=len(enrolled_faces),
            best_image=best_image
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal error: {str(e)}"
        )

@app.post("/api/recognize", response_model=RecognitionResponse)
async def recognize_face(request: RecognizeRequest):
    """
    Recognize a face against enrolled database
    
    - Detects face in image
    - Matches against enrolled faces
    - Returns best match with confidence
    """
    try:
        img = base64_to_image(request.image)
        
        if img is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid image format"
            )
        
        # Detect faces
        faces = face_app.get(img)
        
        if len(faces) == 0:
            return RecognitionResponse(
                recognized=False,
                message="No face detected"
            )
        
        if len(faces) > 1:
            return RecognitionResponse(
                recognized=False,
                message="Multiple faces detected - show one face only"
            )
        
        face = faces[0]
        
        # Validate face
        is_valid, error_msg = validate_face(face, img)
        if not is_valid:
            return RecognitionResponse(
                recognized=False,
                message=f"Face validation failed: {error_msg}"
            )
        
        # Extract embedding
        embedding = normalize(face.embedding)
        
        # Find best match
        best_match_name = None
        best_distance = float('inf')
        
        for name, enrolled_data in enrolled_faces.items():
            stored_embedding = enrolled_data['embedding']
            distance = cosine(embedding, stored_embedding)
            
            if distance < best_distance:
                best_distance = distance
                best_match_name = name
        
        # Check if match is good enough
        if best_distance < RECOGNITION_THRESHOLD:
            confidence = (1 - best_distance) * 100
            enrolled_image = enrolled_faces[best_match_name]['image']
            
            return RecognitionResponse(
                recognized=True,
                name=best_match_name,
                confidence=round(confidence, 2),
                message=f"Recognized as {best_match_name}",
                enrolled_image=enrolled_image
            )
        else:
            confidence = (1 - best_distance) * 100 if best_match_name else 0
            return RecognitionResponse(
                recognized=False,
                name=None,
                confidence=round(confidence, 2),
                message=f"Unknown person (best match: {best_match_name} at {confidence:.1f}%)"
            )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal error: {str(e)}"
        )

@app.get("/api/enrolled", response_model=List[EnrolledFace])
async def get_enrolled_faces():
    """Get all enrolled faces"""
    return [
        EnrolledFace(
            name=name,
            image=data['image'],
            images_used=data['images_used'],
            enrolled_at=data['enrolled_at']
        )
        for name, data in enrolled_faces.items()
    ]

@app.delete("/api/enrolled/{name}")
async def delete_enrolled_face(name: str):
    """Delete an enrolled face"""
    if name not in enrolled_faces:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Face '{name}' not found"
        )
    
    del enrolled_faces[name]
    save_enrolled_faces()
    
    return {
        "success": True,
        "message": f"Deleted {name}",
        "enrolled_count": len(enrolled_faces)
    }

@app.delete("/api/enrolled")
async def delete_all_enrolled():
    """Delete all enrolled faces"""
    count = len(enrolled_faces)
    enrolled_faces.clear()
    save_enrolled_faces()
    
    return {
        "success": True,
        "message": f"Deleted {count} enrolled faces",
        "enrolled_count": 0
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
