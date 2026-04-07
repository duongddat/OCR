<template>
  <div class="app-layout">
    <header class="app-header glass">
      <div class="logo">
        <span class="material-icons-round text-primary">document_scanner</span>
        <h1>ScanAI</h1>
      </div>
    </header>

    <main class="main-content">
      <!-- 1. Scanner Camera View -->
      <section v-if="appState === 'scanning'" class="view-panel">
        <div class="camera-wrapper">
          <video ref="videoElement" autoplay playsinline></video>
          
          <div class="scanner-overlay">
            <div class="frame-corner top-left"></div>
            <div class="frame-corner top-right"></div>
            <div class="frame-corner bottom-left"></div>
            <div class="frame-corner bottom-right"></div>
            <div class="instruction-text">Đặt tài liệu vào khung và chụp</div>
          </div>
        </div>

        <div class="camera-controls glass">
          <button @click="triggerFileUpload" class="control-btn" aria-label="Chọn ảnh">
            <span class="material-icons-round">photo_library</span>
            <input type="file" ref="fileInput" @change="handleFileUpload" accept="image/*" class="hidden">
          </button>
          
          <button @click="capturePhoto" class="capture-button" aria-label="Chụp ảnh">
            <div class="capture-inner"></div>
          </button>
          
          <button @click="switchCamera()" class="control-btn" aria-label="Đổi Camera">
            <span class="material-icons-round">cameraswitch</span>
          </button>
        </div>
      </section>

      <!-- 2. Loading / Processing View -->
      <section v-if="appState === 'loading'" class="view-panel center-content">
        <div class="glass-card loading-card">
          <div class="ai-scanner-animation">
            <div class="document-icon">
              <span class="material-icons-round">description</span>
            </div>
            <div class="scanning-laser"></div>
          </div>
          <h2>Đang phân tích bằng AI...</h2>
          <p class="text-secondary">Vui lòng đợi giây lát để hệ thống trích xuất văn bản.</p>
        </div>
      </section>

      <!-- 3. Results View -->
      <section v-if="appState === 'results'" class="view-panel result-layout">
        <div class="result-container">
          <div class="result-columns">
            <div class="preview-box glass-card">
              <img :src="capturedPhoto" alt="Tài liệu đã quét">
              
              <!-- Bounding Boxes overlay -->
              <div class="bbox-container">
                <div 
                  v-for="(detail, i) in ocrDetails" 
                  :key="i"
                  class="bbox"
                  :style="getBBoxStyle(detail.box)"
                  :title="`Text: ${detail.text} | Confidence: ${(detail.confidence*100).toFixed(1)}%`"
                ></div>
              </div>
            </div>
            
            <div class="text-box glass-card">
              <div class="text-header">
                <h3>Văn bản được nhận dạng</h3>
                <button @click="copyText" class="icon-button text-primary" aria-label="Sao chép">
                  <span class="material-icons-round">content_copy</span>
                </button>
              </div>
              <textarea readonly v-model="extractedText" placeholder="Không tìm thấy văn bản nào..."></textarea>
            </div>
          </div>

          <div class="result-actions">
            <button @click="resetScanner" class="secondary-button">
              <span class="material-icons-round">replay</span>
              Quét lại
            </button>
            <button @click="downloadJSON" class="primary-button">
              <span class="material-icons-round">download</span>
              Lưu JSON
            </button>
          </div>
        </div>
      </section>
    </main>
    
    <!-- Hidden Canvas for capturing video frames -->
    <canvas ref="canvasElement" class="hidden"></canvas>
    
    <!-- Toast notification -->
    <div :class="['toast', { 'toast-visible': toastVisible }]">
      <span class="material-icons-round">check_circle</span>
      <span>{{ toastMessage }}</span>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, onBeforeUnmount } from 'vue';

// State
const appState = ref('scanning'); // scanning, loading, results
const stream = ref(null);
const facingMode = ref('environment'); // environment (back) or user (front)
const capturedPhoto = ref(null); // Data URL of the picture
const extractedText = ref('');
const ocrDetails = ref([]);

// UI / Elements refs
const videoElement = ref(null);
const canvasElement = ref(null);
const fileInput = ref(null);

// Original image dimensions to scale Bboxes
const imageActualSize = ref({ width: 0, height: 0 });

// Toast
const toastVisible = ref(false);
const toastMessage = ref('');

const showToast = (msg) => {
  toastMessage.value = msg;
  toastVisible.value = true;
  setTimeout(() => {
    toastVisible.value = false;
  }, 3000);
};

// Start Camera
const startCamera = async () => {
  try {
    if (stream.value) {
      stream.value.getTracks().forEach(track => track.stop());
    }
    
    const constraints = {
      video: { 
        facingMode: facingMode.value,
        width: { ideal: 1920 },
        height: { ideal: 1080 }
      }
    };
    
    stream.value = await navigator.mediaDevices.getUserMedia(constraints);
    if (videoElement.value) {
      videoElement.value.srcObject = stream.value;
    }
  } catch (error) {
    console.error("Camera access error:", error);
    showToast("Không thể kết nối Camera. " + error.message);
  }
};

const switchCamera = () => {
  facingMode.value = facingMode.value === 'environment' ? 'user' : 'environment';
  startCamera();
};

const triggerFileUpload = () => {
  if (fileInput.value) {
    fileInput.value.click();
  }
};

const processImageFile = async (file) => {
  // Preview
  const reader = new FileReader();
  reader.onload = (e) => {
    capturedPhoto.value = e.target.result;
    
    // Load image to get original dimensions for bbox scaling
    const img = new Image();
    img.onload = () => {
      imageActualSize.value = { width: img.width, height: img.height };
    };
    img.src = e.target.result;
  };
  reader.readAsDataURL(file);

  // Stop camera
  if (stream.value) {
    stream.value.getTracks().forEach(track => track.stop());
  }

  appState.value = 'loading';
  
  // Create FormData
  const formData = new FormData();
  formData.append('image', file);

  // Send request to FastAPI
  try {
    // Lấy URL của Backend từ file .env (VD: http://localhost:8000)
    // Nếu deploy mà không cấu hình, nó sẽ tự động chạy thư mục tương đối (relative path)
    const baseUrl = import.meta.env.VITE_API_URL || '';
    const endpoint = `${baseUrl}/api/extract-text`;
    const response = await fetch(endpoint, {
      method: 'POST',
      body: formData
    });
    
    const data = await response.json();
    if (data.status === 'success') {
      console.log(data);
      extractedText.value = data.text;
      ocrDetails.value = data.details || [];
      // Dùng kích thước gốc từ server để scale bounding box chính xác
      if (data.imageSize) {
        imageActualSize.value = { width: data.imageSize.width, height: data.imageSize.height };
      }
    } else {
      extractedText.value = "Lỗi server: " + (data.message || "Unknown error");
    }
  } catch (err) {
    console.error(err);
    extractedText.value = `Failed to connection: ${err.message}`;
  } finally {
    appState.value = 'results';
  }
};

const capturePhoto = () => {
  if (!videoElement.value || !canvasElement.value) return;
  
  const video = videoElement.value;
  const canvas = canvasElement.value;
  
  // Match canvas dimensions to video actual size
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  imageActualSize.value = { width: video.videoWidth, height: video.videoHeight };
  
  const ctx = canvas.getContext('2d');
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  
  // Stop camera
  if (stream.value) {
    stream.value.getTracks().forEach(track => track.stop());
  }
  
  // Get Data URL for preview
  capturedPhoto.value = canvas.toDataURL('image/jpeg', 0.9);
  
  // Convert blob to file and process
  canvas.toBlob((blob) => {
    if (blob) {
      const file = new File([blob], "capture.jpg", { type: "image/jpeg" });
      processImageFile(file);
    }
  }, 'image/jpeg', 0.9);
};

const handleFileUpload = (e) => {
  const file = e.target.files[0];
  if (file) {
    processImageFile(file);
  }
};

const resetScanner = () => {
  capturedPhoto.value = null;
  extractedText.value = '';
  ocrDetails.value = [];
  appState.value = 'scanning';
  startCamera();
};

const copyText = () => {
  navigator.clipboard.writeText(extractedText.value)
    .then(() => showToast('Đã sao chép văn bản!'))
    .catch(() => showToast('Sao chép thất bại!'));
};

const downloadJSON = () => {
  const data = JSON.stringify(ocrDetails.value, null, 2);
  const blob = new Blob([data], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'ocr-results.json';
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
};

// Calculate CSS geometry for bounding boxes based on backend coordinates
const getBBoxStyle = (box) => {
  if (!box || box.length < 4 || !imageActualSize.value.width) return {};
  
  // PaddleOCR box format: [[x1, y1], [x2, y2], [x3, y3], [x4, y4]] (top-left, top-right, bottom-right, bottom-left)
  const xs = box.map(p => p[0]);
  const ys = box.map(p => p[1]);
  
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);

  // Convert to percentages relative to original image size
  const leftPct = (minX / imageActualSize.value.width) * 100;
  const topPct = (minY / imageActualSize.value.height) * 100;
  const widthPct = ((maxX - minX) / imageActualSize.value.width) * 100;
  const heightPct = ((maxY - minY) / imageActualSize.value.height) * 100;

  return {
    left: `${leftPct}%`,
    top: `${topPct}%`,
    width: `${widthPct}%`,
    height: `${heightPct}%`
  };
};

onMounted(() => {
  startCamera();
});

onBeforeUnmount(() => {
  if (stream.value) {
    stream.value.getTracks().forEach(track => track.stop());
  }
});
</script>

<style scoped>
/* App Layout */
.app-layout {
  height: 100%;
  display: flex;
  flex-direction: column;
}

.app-header {
  min-height: 70px;
  display: flex;
  justify-content: center;
  align-items: center;
  padding: 0 24px;
  position: relative;
  z-index: 10;
  box-shadow: 0 4px 30px rgba(0, 0, 0, 0.2);
}

.logo {
  display: flex;
  align-items: center;
  gap: 12px;
  animation: pulseLogo 3s infinite alternate;
}

@keyframes pulseLogo {
  0% { filter: drop-shadow(0 0 8px rgba(99, 102, 241, 0.3)); }
  100% { filter: drop-shadow(0 0 16px rgba(6, 182, 212, 0.6)); }
}

.logo span {
  font-size: 32px;
}

.logo h1 {
  font-size: 1.8rem;
  font-weight: 800;
  letter-spacing: -0.5px;
  background: linear-gradient(to right, #ffffff, var(--accent), var(--primary));
  background-size: 200% auto;
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  animation: shineText 5s linear infinite;
}

@keyframes shineText {
  to { background-position: 200% center; }
}

.main-content {
  flex: 1;
  position: relative;
  overflow: hidden;
}

/* Views */
.view-panel {
  position: absolute;
  top: 0; left: 0; right: 0; bottom: 0;
  display: flex;
  flex-direction: column;
  animation: slideUp 0.6s cubic-bezier(0.2, 0.8, 0.2, 1) forwards;
}

.center-content {
  justify-content: center;
  align-items: center;
  padding: 24px;
}

/* 1. Camera View */
.camera-wrapper {
  position: relative;
  flex: 1;
  background: #000;
  overflow: hidden;
  border-radius: 0 0 24px 24px;
  box-shadow: inset 0 0 40px rgba(0,0,0,0.8);
}

video {
  width: 100%;
  height: 100%;
  object-fit: cover;
  opacity: 0.9;
  transition: opacity 0.3s ease;
}

.scanner-overlay {
  position: absolute;
  top: 15%; left: 8%; right: 8%; bottom: 15%;
  border: 1px dashed rgba(255, 255, 255, 0.3);
  pointer-events: none;
  border-radius: 16px;
  box-shadow: inset 0 0 50px rgba(0,0,0,0.2);
}

.frame-corner {
  position: absolute;
  width: 40px; height: 40px;
  border: 4px solid var(--accent);
  border-radius: 6px;
  filter: drop-shadow(0 0 8px var(--accent));
}

.top-left { top: -2px; left: -2px; border-right: none; border-bottom: none; }
.top-right { top: -2px; right: -2px; border-left: none; border-bottom: none; }
.bottom-left { bottom: -2px; left: -2px; border-right: none; border-top: none; }
.bottom-right { bottom: -2px; right: -2px; border-left: none; border-top: none; }

.instruction-text {
  position: absolute;
  bottom: -45px;
  left: 0; width: 100%;
  text-align: center;
  color: #fff;
  font-weight: 500;
  letter-spacing: 0.5px;
  font-size: 0.95rem;
  text-shadow: 0 2px 8px rgba(0,0,0,0.9);
  background: rgba(0,0,0,0.4);
  backdrop-filter: blur(4px);
  padding: 8px 0;
  border-radius: 20px;
  width: max-content;
  margin: 0 auto;
  transform: translateX(10%); /* Adjust centering hack */
  left: 50%;
  transform: translateX(-50%);
}

.camera-controls {
  height: 110px;
  display: flex;
  justify-content: space-evenly;
  align-items: center;
  padding-bottom: env(safe-area-inset-bottom);
  background: linear-gradient(to top, rgba(0,0,0,0.8) 0%, transparent 100%);
  position: absolute;
  bottom: 0;
  width: 100%;
  border-top: none;
}

.control-btn {
  width: 56px; height: 56px;
  border-radius: 50%;
  background: rgba(255, 255, 255, 0.1);
  backdrop-filter: blur(12px);
  color: #fff;
  display: flex; justify-content: center; align-items: center;
  box-shadow: 0 4px 15px rgba(0,0,0,0.3);
}

.capture-button {
  width: 80px; height: 80px;
  border-radius: 50%;
  border: 4px solid rgba(255,255,255,0.8);
  padding: 4px;
  display: flex; justify-content: center; align-items: center;
  box-shadow: 0 0 20px rgba(0, 0, 0, 0.5);
  transition: transform 0.2s cubic-bezier(0.175, 0.885, 0.32, 1.275);
}

.capture-button:hover {
  transform: scale(1.05);
  border-color: #fff;
}

.capture-inner {
  width: 100%; height: 100%;
  border-radius: 50%;
  background: var(--primary);
  box-shadow: inset 0 0 10px rgba(255,255,255,0.5);
}

/* 2. Loading View */
.loading-card {
  display: flex;
  flex-direction: column;
  align-items: center;
  text-align: center;
  gap: 20px;
  max-width: 420px;
  width: 90%;
  position: relative;
  overflow: hidden;
}

.loading-card::before {
  content: ''; position: absolute; top: -50%; left: -50%; width: 200%; height: 200%;
  background: conic-gradient(transparent, var(--accent), transparent 30%);
  animation: spin 4s linear infinite;
  z-index: -1;
  opacity: 0.15;
}

@keyframes spin { 100% { transform: rotate(360deg); } }

.ai-scanner-animation {
  position: relative;
  width: 110px; height: 110px;
  margin: 0 auto 16px;
  display: flex; justify-content: center; align-items: center;
  background: rgba(99, 102, 241, 0.1);
  border-radius: 24px;
  border: 1px solid rgba(99, 102, 241, 0.2);
}

.document-icon span {
  font-size: 70px;
  color: var(--accent);
  opacity: 0.9;
  filter: drop-shadow(0 0 10px var(--accent));
}

.scanning-laser {
  position: absolute;
  left: -10px; right: -10px;
  height: 2px;
  background: #fff;
  box-shadow: 0 0 15px 3px var(--accent);
  animation: scanLaser 2s infinite ease-in-out;
  border-radius: 50%;
}

.loading-card h2 {
  font-size: 1.5rem;
  font-weight: 700;
  background: linear-gradient(to right, #fff, var(--text-secondary));
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}

/* 3. Results View */
.result-layout {
  padding: 24px;
  padding-bottom: calc(24px + env(safe-area-inset-bottom));
  overflow-y: auto;
}

.result-container {
  display: flex;
  flex-direction: column;
  gap: 24px;
  max-width: 1280px;
  margin: 0 auto;
  width: 100%;
}

.result-columns {
  display: flex;
  flex-direction: column;
  gap: 24px;
  width: 100%;
}

@media (min-width: 900px) {
  .result-columns {
    flex-direction: row;
    align-items: stretch;
  }
}

.preview-box {
  flex: 1;
  min-width: 0;
  width: 100%;
  display: flex;
  justify-content: center;
  align-items: center;
  overflow: hidden;
  padding: 16px;
  position: relative;
  background: rgba(0,0,0,0.3);
  border-radius: 24px;
  min-height: 350px;
}

.preview-box img {
  width: 100%;
  height: 100%;
  max-height: 65vh;
  object-fit: contain;
  border-radius: 8px;
  filter: drop-shadow(0 10px 20px rgba(0,0,0,0.5));
}

.bbox-container {
  position: absolute;
  top: 16px; left: 16px; right: 16px; bottom: 16px;
  pointer-events: none;
}

.bbox {
  position: absolute;
  border: 1.5px solid var(--accent);
  background: rgba(6, 182, 212, 0.15);
  box-shadow: 0 0 8px rgba(6, 182, 212, 0.6);
  border-radius: 3px;
  transition: all 0.2s;
}

.bbox:hover {
  background: rgba(6, 182, 212, 0.3);
  border-width: 2px;
}

.text-box {
  flex: 1.2;
  min-width: 0;
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.text-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding-bottom: 12px;
  border-bottom: 1px solid var(--surface-border);
}

.text-header h3 {
  font-size: 1.3rem;
  font-weight: 700;
  margin: 0;
  color: var(--text-primary);
  letter-spacing: -0.3px;
}

textarea {
  flex: 1;
  width: 100%;
  min-height: 350px;
  background: rgba(0, 0, 0, 0.2);
  border: 1px solid var(--surface-border);
  border-radius: 16px;
  padding: 24px;
  color: #fff;
  font-family: 'Inter', system-ui, -apple-system, sans-serif;
  font-size: 1.1rem;
  line-height: 1.8;
  resize: vertical;
  box-shadow: inset 0 4px 20px rgba(0,0,0,0.3);
  transition: border-color 0.3s, box-shadow 0.3s;
}

textarea:focus {
  outline: none;
  border-color: var(--primary);
  box-shadow: inset 0 4px 20px rgba(0,0,0,0.3), 0 0 0 3px rgba(99, 102, 241, 0.2);
}

.result-actions {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 20px;
}

/* Toast */
.toast {
  position: fixed;
  bottom: 40px;
  left: 50%;
  transform: translateX(-50%) translateY(150px);
  background: rgba(20, 20, 25, 0.85);
  backdrop-filter: blur(16px);
  -webkit-backdrop-filter: blur(16px);
  border: 1px solid rgba(255, 255, 255, 0.1);
  padding: 14px 28px;
  border-radius: 50px;
  display: flex;
  align-items: center;
  gap: 12px;
  color: #fff;
  font-weight: 500;
  box-shadow: 0 10px 40px rgba(0,0,0,0.5), 0 0 0 1px rgba(255,255,255,0.05) inset;
  transition: transform 0.5s cubic-bezier(0.175, 0.885, 0.32, 1.275);
  z-index: 100;
}

.toast-visible {
  transform: translateX(-50%) translateY(0);
}

.toast .material-icons-round {
  color: var(--success);
  font-size: 24px;
  filter: drop-shadow(0 0 5px var(--success));
}
</style>
