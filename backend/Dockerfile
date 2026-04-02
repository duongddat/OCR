# Sử dụng phiên bản Python 3.10 chính thức siêu nhẹ
FROM python:3.10-slim

# Biến môi trường chống lỗi buffer và cache
ENV PYTHONUNBUFFERED=1

# Thiết lập thư mục làm việc trong Container
WORKDIR /app

# Cài đặt các thư viện hệ thống cần thiết cho OpenCV và OCR (Cực kỳ quan trọng trên Linux/VPS)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy file cài đặt rễ vào Container
COPY requirements.txt .

# Cài đặt toàn bộ thư viện Python (dùng CPU cho rẻ và nhẹ máy)
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir paddlepaddle
RUN pip install --no-cache-dir fastapi uvicorn python-multipart paddleocr pydantic vietocr einops gdown

# Copy toàn bộ code Backend vào
COPY . .

# Mở cổng chạy Server
EXPOSE 8000

# Lệnh quyết định cách chạy dự án (Uvicorn - Chuyên cho môi trường thực tế Production)
CMD ["uvicorn", "main_vietocr:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
