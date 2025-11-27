import streamlit as st
import cv2
import torch
import torch.nn as nn
import numpy as np
import tempfile
import os
from datetime import datetime
import time
from PIL import Image

# =============================================================================
# CSRNet MODEL
# =============================================================================

class CSRNet(nn.Module):
    def __init__(self):
        super(CSRNet, self).__init__()
        
        self.frontend = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.backend = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1)
        )
        
    def forward(self, x):
        x = self.frontend(x)
        x = self.backend(x)
        return x


# =============================================================================
# HYBRID DETECTOR
# =============================================================================

class HybridCrowdDetector:
    def __init__(self, csrnet_path=None, use_yolo=True, device='cuda', fusion_method='max'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.use_yolo = use_yolo
        self.fusion_method = fusion_method
        
        # Load YOLO
        if use_yolo:
            try:
                from ultralytics import YOLO
                self.yolo = YOLO('yolov8n.pt')
            except ImportError:
                st.error("⚠️ ultralytics not installed. Install with: pip install ultralytics")
                self.use_yolo = False
        
        # Load CSRNet (optional)
        self.use_csrnet = csrnet_path is not None and os.path.exists(csrnet_path)
        if self.use_csrnet:
            self.csrnet = CSRNet().to(self.device)
            checkpoint = torch.load(csrnet_path, map_location=self.device)
            self.csrnet.load_state_dict(checkpoint['model_state_dict'])
            self.csrnet.eval()
    
    def detect_people_yolo(self, frame):
        results = self.yolo(frame, verbose=False)
        boxes = []
        confidences = []
        
        for result in results:
            for box in result.boxes:
                if int(box.cls[0]) == 0:  # Person class
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0])
                    boxes.append([int(x1), int(y1), int(x2), int(y2)])
                    confidences.append(conf)
        
        return boxes, confidences
    
    def predict_csrnet_count(self, frame):
        if not self.use_csrnet:
            return 0, None
        
        with torch.no_grad():
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_normalized = frame_rgb.astype(np.float32) / 255.0
            frame_tensor = torch.from_numpy(frame_normalized).permute(2, 0, 1).unsqueeze(0).float()
            frame_tensor = frame_tensor.to(self.device)
            
            output = self.csrnet(frame_tensor)
            count = output.sum().item()
            density_map = output.squeeze().cpu().numpy()
            
        return count, density_map
    
    def fuse_counts(self, yolo_count, csrnet_count):
        if self.fusion_method == 'max':
            return max(yolo_count, csrnet_count)
        elif self.fusion_method == 'average':
            return int((yolo_count + csrnet_count) / 2)
        elif self.fusion_method == 'yolo':
            return yolo_count
        elif self.fusion_method == 'csrnet':
            return csrnet_count
        return max(yolo_count, csrnet_count)
    
    def process_frame(self, frame, confidence_threshold=0.3):
        # YOLO detection
        if self.use_yolo:
            boxes, confidences = self.detect_people_yolo(frame)
            filtered = [(b, c) for b, c in zip(boxes, confidences) if c >= confidence_threshold]
            boxes = [b for b, c in filtered]
            confidences = [c for b, c in filtered]
            yolo_count = len(boxes)
        else:
            boxes, confidences = [], []
            yolo_count = 0
        
        # CSRNet estimation
        csrnet_count, density_map = self.predict_csrnet_count(frame)
        
        # Fuse counts
        final_count = self.fuse_counts(yolo_count, int(csrnet_count))
        
        return boxes, confidences, final_count, yolo_count, int(csrnet_count)
    
    def draw_detections(self, frame, boxes, confidences):
        vis_frame = frame.copy()
        
        for box, conf in zip(boxes, confidences):
            x1, y1, x2, y2 = box
            
            # Color based on confidence
            if conf > 0.7:
                color = (0, 255, 0)
            elif conf > 0.5:
                color = (0, 255, 255)
            else:
                color = (0, 165, 255)
            
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
            
            label = f"{conf:.2f}"
            cv2.putText(vis_frame, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return vis_frame


# =============================================================================
# PAGE NAVIGATION
# =============================================================================

def show_home_page():
    """Display the futuristic home/landing page with mode selection"""
    
    # Add particle animation CSS
    st.markdown("""
    <style>
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes glowPulse {
        0%, 100% {
            box-shadow: 0 0 20px rgba(122, 92, 255, 0.5), 0 0 40px rgba(0, 212, 255, 0.3);
        }
        50% {
            box-shadow: 0 0 30px rgba(122, 92, 255, 0.8), 0 0 60px rgba(0, 212, 255, 0.5);
        }
    }
    
    @keyframes float {
        0%, 100% {
            transform: translateY(0px);
        }
        50% {
            transform: translateY(-10px);
        }
    }
    
    .hero-container {
        animation: fadeInUp 0.8s ease-out;
        background: rgba(26, 28, 32, 0.6);
        backdrop-filter: blur(20px);
        border-radius: 24px;
        border: 1px solid rgba(122, 92, 255, 0.3);
        padding: 60px 40px;
        margin: 60px auto;
        max-width: 900px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4), 
                    0 0 80px rgba(122, 92, 255, 0.15),
                    inset 0 1px 0 rgba(255, 255, 255, 0.05);
    }
    
    .neon-title {
        font-size: 4.5rem;
        font-weight: 900;
        background: linear-gradient(135deg, #7A5CFF 0%, #00D4FF 50%, #9B59B6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 0 80px rgba(122, 92, 255, 0.5);
        margin-bottom: 20px;
        letter-spacing: -2px;
        animation: glowPulse 3s ease-in-out infinite;
    }
    
    .subtitle {
        font-size: 1.4rem;
        color: #B0B3B8;
        margin-bottom: 40px;
        font-weight: 300;
        letter-spacing: 0.5px;
    }
    
    .mode-card {
        background: linear-gradient(145deg, rgba(26, 28, 32, 0.8) 0%, rgba(38, 40, 44, 0.6) 100%);
        border-radius: 20px;
        padding: 40px 30px;
        text-align: center;
        border: 1px solid rgba(122, 92, 255, 0.2);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        cursor: pointer;
        position: relative;
        overflow: hidden;
        animation: fadeInUp 0.8s ease-out;
    }
    
    .mode-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(135deg, rgba(122, 92, 255, 0.1) 0%, rgba(0, 212, 255, 0.1) 100%);
        opacity: 0;
        transition: opacity 0.4s ease;
    }
    
    .mode-card:hover {
        transform: translateY(-12px);
        border-color: rgba(122, 92, 255, 0.8);
        box-shadow: 0 20px 60px rgba(122, 92, 255, 0.4),
                    0 0 80px rgba(0, 212, 255, 0.3),
                    inset 0 1px 0 rgba(255, 255, 255, 0.1);
    }
    
    .mode-card:hover::before {
        opacity: 1;
    }
    
    .mode-icon {
        font-size: 4rem;
        margin-bottom: 20px;
        filter: drop-shadow(0 0 20px rgba(122, 92, 255, 0.6));
        animation: float 3s ease-in-out infinite;
    }
    
    .mode-title {
        font-size: 1.8rem;
        font-weight: 700;
        color: #FFFFFF;
        margin-bottom: 12px;
        text-shadow: 0 0 20px rgba(122, 92, 255, 0.5);
    }
    
    .mode-desc {
        font-size: 1rem;
        color: #8E9196;
        line-height: 1.6;
    }
    
    .feature-item {
        display: flex;
        align-items: center;
        margin: 20px 0;
        padding: 15px;
        background: rgba(26, 28, 32, 0.4);
        border-radius: 12px;
        border-left: 3px solid #7A5CFF;
        transition: all 0.3s ease;
    }
    
    .feature-item:hover {
        background: rgba(38, 40, 44, 0.6);
        border-left-color: #00D4FF;
        transform: translateX(5px);
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Hero Section with glassmorphism
    st.markdown("""
    <div class='hero-container'>
        <div style='text-align: right;'>
            <h1 class='neon-title'>Crowd Detection System</h1>
            <p class='subtitle'>AI-powered real-time monitoring and crowd analysis</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Add spacing
    st.markdown("<div style='height: 40px;'></div>", unsafe_allow_html=True)
    
    # Detection Mode Cards
    st.markdown("<h2 style='text-align: center; color: #FFFFFF; font-size: 2.5rem; margin-bottom: 60px; text-shadow: 0 0 30px rgba(122, 92, 255, 0.6);'>Choose Your Detection Mode</h2>", unsafe_allow_html=True)
    
    col_a, col_b, col_c = st.columns(3, gap="large")
    
    with col_a:
        st.markdown("""
        <div class='mode-card'>
            <div class='mode-icon'>🖼️</div>
            <h3 class='mode-title'>Image Detection</h3>
            <p class='mode-desc'>Upload and analyze static images with advanced AI detection algorithms</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)
        if st.button("📸 Analyze Image", use_container_width=True, type="primary", key="btn_image"):
            st.session_state.page = "image"
            st.rerun()
    
    with col_b:
        st.markdown("""
        <div class='mode-card'>
            <div class='mode-icon'>📹</div>
            <h3 class='mode-title'>Video Analysis</h3>
            <p class='mode-desc'>Process pre-recorded videos with frame-by-frame crowd tracking</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)
        if st.button("🎬 Process Video", use_container_width=True, type="primary", key="btn_video"):
            st.session_state.page = "video"
            st.rerun()
    
    with col_c:
        st.markdown("""
        <div class='mode-card'>
            <div class='mode-icon'>🎥</div>
            <h3 class='mode-title'>Live Webcam</h3>
            <p class='mode-desc'>Real-time detection and monitoring directly from your camera feed</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)
        if st.button("📡 Live Stream", use_container_width=True, type="primary", key="btn_webcam"):
            st.session_state.page = "webcam"
            st.rerun()
    
    # Features Section
    st.markdown("<div style='height: 80px;'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div style='max-width: 1200px; margin: 0 auto;'>
        <h3 style='text-align: center; color: #FFFFFF; font-size: 2.2rem; margin-bottom: 50px; text-shadow: 0 0 30px rgba(0, 212, 255, 0.6);'>
            ✨ Advanced Features
        </h3>
    </div>
    """, unsafe_allow_html=True)
    
    feat_col1, feat_col2 = st.columns(2, gap="large")
    
    with feat_col1:
        st.markdown("""
        <div class='feature-item'>
            <span style='font-size: 2rem; margin-right: 15px;'>🎯</span>
            <div>
                <strong style='color: #7A5CFF; font-size: 1.1rem;'>Accurate Detection</strong>
                <p style='color: #8E9196; margin: 5px 0 0 0;'>YOLO-powered person detection with high precision</p>
            </div>
        </div>
        <div class='feature-item'>
            <span style='font-size: 2rem; margin-right: 15px;'>🚨</span>
            <div>
                <strong style='color: #00D4FF; font-size: 1.1rem;'>Smart Alerts</strong>
                <p style='color: #8E9196; margin: 5px 0 0 0;'>Configurable thresholds and intelligent alarm system</p>
            </div>
        </div>
        <div class='feature-item'>
            <span style='font-size: 2rem; margin-right: 15px;'>📊</span>
            <div>
                <strong style='color: #9B59B6; font-size: 1.1rem;'>Real-time Analytics</strong>
                <p style='color: #8E9196; margin: 5px 0 0 0;'>Live count tracking with visual feedback</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with feat_col2:
        st.markdown("""
        <div class='feature-item'>
            <span style='font-size: 2rem; margin-right: 15px;'>🎨</span>
            <div>
                <strong style='color: #7A5CFF; font-size: 1.1rem;'>Visual Feedback</strong>
                <p style='color: #8E9196; margin: 5px 0 0 0;'>Bounding boxes with confidence scores overlay</p>
            </div>
        </div>
        <div class='feature-item'>
            <span style='font-size: 2rem; margin-right: 15px;'>📈</span>
            <div>
                <strong style='color: #00D4FF; font-size: 1.1rem;'>Detailed Statistics</strong>
                <p style='color: #8E9196; margin: 5px 0 0 0;'>Comprehensive counting history and trends</p>
            </div>
        </div>
        <div class='feature-item'>
            <span style='font-size: 2rem; margin-right: 15px;'>💾</span>
            <div>
                <strong style='color: #9B59B6; font-size: 1.1rem;'>Export Capability</strong>
                <p style='color: #8E9196; margin: 5px 0 0 0;'>Download alarm logs and analysis reports</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<div style='height: 60px;'></div>", unsafe_allow_html=True)


def show_back_button():
    """Display a back button to return to home page"""
    if st.button("⬅️ Back to Home", type="secondary"):
        st.session_state.page = "home"
        if 'webcam_active' in st.session_state:
            st.session_state.webcam_active = False
        st.rerun()


def show_image_page():
    """Image detection page"""
    show_back_button()
    
    st.title("🖼️ Image Crowd Detection")
    st.markdown("Upload an image to detect and count people")
    
    st.sidebar.header("⚙️ Detection Settings")
    confidence_threshold = st.sidebar.slider("Detection Confidence", 0.1, 0.9, 0.3, 0.05)
    fusion_method = st.sidebar.selectbox("Count Fusion Method", ['max', 'average', 'yolo'], 0)
    
    st.sidebar.header("🚨 Alarm Settings")
    enable_alarm = st.sidebar.checkbox("Enable Alarm System", True)
    alarm_threshold = st.sidebar.number_input("Alarm Threshold", 1, 1000, 10, 1)
    warning_threshold = int(alarm_threshold * 0.8)
    
    uploaded_file = st.file_uploader("Choose an image file", type=['jpg', 'jpeg', 'png', 'bmp'])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image_np = np.array(image)
        
        if len(image_np.shape) == 3 and image_np.shape[2] == 3:
            frame = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        else:
            frame = image_np
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.image(image, caption="Uploaded Image", use_container_width=True)
        
        with col2:
            st.markdown("### 📋 Image Info")
            st.write(f"**Size:** {image.size[0]} x {image.size[1]} px")
            st.write(f"**Format:** {image.format}")
            st.write(f"**Mode:** {image.mode}")
        
        if st.button("🚀 Start Detection", type="primary", use_container_width=True):
            with st.spinner("🔄 Analyzing image..."):
                detector = HybridCrowdDetector(
                    csrnet_path=None,
                    use_yolo=True,
                    device='cuda' if torch.cuda.is_available() else 'cpu',
                    fusion_method=fusion_method
                )
                
                boxes, confidences, final_count, yolo_count, csrnet_count = detector.process_frame(frame, confidence_threshold)
                vis_frame = detector.draw_detections(frame, boxes, confidences)
                height, width = vis_frame.shape[:2]
                
                overlay_text = f"Count: {final_count}"
                cv2.putText(vis_frame, overlay_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
                
                if enable_alarm:
                    if final_count >= alarm_threshold:
                        status = "🚨 ALARM"
                        status_color = "red"
                        cv2.rectangle(vis_frame, (0, 0), (width-1, height-1), (0, 0, 255), 10)
                    elif final_count >= warning_threshold:
                        status = "⚠️ WARNING"
                        status_color = "orange"
                    else:
                        status = "✅ NORMAL"
                        status_color = "green"
                else:
                    status = "ℹ️ DETECTED"
                    status_color = "blue"
                
                vis_frame_rgb = cv2.cvtColor(vis_frame, cv2.COLOR_BGR2RGB)
                
                st.markdown("---")
                st.markdown(f"### Status: :{status_color}[{status}]")
                
                col1, col2 = st.columns([3, 2])
                
                with col1:
                    st.image(vis_frame_rgb, caption="Detection Results", use_container_width=True)
                
                with col2:
                    st.markdown("### 📊 Detection Results")
                    st.metric("👥 Total Count", final_count)
                    st.metric("🎯 YOLO Count", yolo_count)
                    st.metric("🗺️ CSRNet Count", csrnet_count)
                    st.metric("📦 Detections", len(boxes))
                    
                    if boxes:
                        st.success(f"✅ Found {len(boxes)} people")
                    else:
                        st.info("ℹ️ No people detected")


def show_video_page():
    """Video detection page"""
    show_back_button()
    
    st.title("📹 Video Crowd Detection")
    st.markdown("Upload a video file to analyze crowd density over time")
    
    st.sidebar.header("⚙️ Detection Settings")
    confidence_threshold = st.sidebar.slider("Detection Confidence", 0.1, 0.9, 0.3, 0.05)
    fusion_method = st.sidebar.selectbox("Count Fusion Method", ['max', 'average', 'yolo'], 0)
    
    st.sidebar.header("🎬 Processing Settings")
    process_every_n_frames = st.sidebar.slider("Process Every N Frames", 1, 10, 2)
    max_frames = st.sidebar.number_input("Max Frames to Process", 10, 1000, 300)
    
    st.sidebar.header("🚨 Alarm Settings")
    enable_alarm = st.sidebar.checkbox("Enable Alarm System", True)
    alarm_threshold = st.sidebar.number_input("Alarm Threshold", 1, 1000, 10, 1)
    warning_threshold = int(alarm_threshold * 0.8)
    
    uploaded_file = st.file_uploader("Choose a video file", type=['mp4', 'avi', 'mov', 'mkv'])
    
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_file.read())
        video_path = tfile.name
        
        st.success(f"✅ Video uploaded: **{uploaded_file.name}**")
        
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps if fps > 0 else 0
        cap.release()
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("📐 Resolution", f"{width}x{height}")
        col2.metric("🎬 FPS", fps)
        col3.metric("🎞️ Frames", total_frames)
        col4.metric("⏱️ Duration", f"{duration:.1f}s")
        
        if st.button("🚀 Start Processing", type="primary", use_container_width=True):
            process_video(video_path, confidence_threshold, fusion_method, enable_alarm, 
                         alarm_threshold, warning_threshold, process_every_n_frames, max_frames)
        
        try:
            os.unlink(video_path)
        except:
            pass


def show_webcam_page():
    """Webcam detection page"""
    show_back_button()
    
    st.title("🎥 Live Webcam Detection")
    st.markdown("Real-time crowd detection from your webcam")
    
    st.sidebar.header("⚙️ Detection Settings")
    confidence_threshold = st.sidebar.slider("Detection Confidence", 0.1, 0.9, 0.3, 0.05)
    fusion_method = st.sidebar.selectbox("Count Fusion Method", ['max', 'average', 'yolo'], 0)
    
    st.sidebar.header("🚨 Alarm Settings")
    enable_alarm = st.sidebar.checkbox("Enable Alarm System", True)
    alarm_threshold = st.sidebar.number_input("Alarm Threshold", 1, 1000, 10, 1)
    warning_threshold = int(alarm_threshold * 0.8)
    st.sidebar.info(f"⚠️ Warning at: {warning_threshold} people")
    
    if 'webcam_active' not in st.session_state:
        st.session_state.webcam_active = False
    if 'webcam_stats' not in st.session_state:
        st.session_state.webcam_stats = {
            'count_history': [], 'alarm_log': [], 'max_count': 0, 
            'total_frames': 0, 'start_time': None
        }
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("▶️ Start Webcam", type="primary", disabled=st.session_state.webcam_active, use_container_width=True):
            st.session_state.webcam_active = True
            st.session_state.webcam_stats = {
                'count_history': [], 'alarm_log': [], 'max_count': 0, 
                'total_frames': 0, 'start_time': time.time()
            }
            st.rerun()
    
    with col2:
        if st.button("⏹️ Stop Webcam", disabled=not st.session_state.webcam_active, use_container_width=True):
            st.session_state.webcam_active = False
            st.rerun()
    
    with col3:
        if st.button("🔄 Reset Stats", disabled=st.session_state.webcam_active, use_container_width=True):
            st.session_state.webcam_stats = {
                'count_history': [], 'alarm_log': [], 'max_count': 0, 
                'total_frames': 0, 'start_time': None
            }
            st.rerun()
    
    if st.session_state.webcam_active:
        st.info("🔴 **Webcam is ACTIVE** - Processing live feed...")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            frame_placeholder = st.empty()
        
        with col2:
            metrics_placeholder = st.empty()
            status_placeholder = st.empty()
            chart_placeholder = st.empty()
        
        stats_placeholder = st.empty()
        
        with st.spinner("🔄 Loading models..."):
            detector = HybridCrowdDetector(
                csrnet_path=None, use_yolo=True,
                device='cuda' if torch.cuda.is_available() else 'cpu',
                fusion_method=fusion_method
            )
        
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            st.error("❌ Could not access webcam")
            st.session_state.webcam_active = False
            return
        
        frame_count = 0
        fps_history = []
        
        try:
            while st.session_state.webcam_active:
                frame_start_time = time.time()
                
                ret, frame = cap.read()
                if not ret:
                    st.error("❌ Failed to read from webcam")
                    break
                
                frame_count += 1
                st.session_state.webcam_stats['total_frames'] = frame_count
                
                if frame_count % 2 == 0:
                    boxes, confidences, final_count, yolo_count, csrnet_count = detector.process_frame(
                        frame, confidence_threshold
                    )
                    
                    st.session_state.webcam_stats['count_history'].append(final_count)
                    if final_count > st.session_state.webcam_stats['max_count']:
                        st.session_state.webcam_stats['max_count'] = final_count
                    
                    vis_frame = detector.draw_detections(frame, boxes, confidences)
                    height, width = vis_frame.shape[:2]
                    
                    overlay_text = f"Count: {final_count}"
                    cv2.putText(vis_frame, overlay_text, (20, 50),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
                    
                    if enable_alarm:
                        if final_count >= alarm_threshold:
                            status = "🚨 ALARM"
                            status_color = "red"
                            
                            st.session_state.webcam_stats['alarm_log'].append({
                                'frame': frame_count,
                                'count': final_count,
                                'timestamp': datetime.now().strftime("%H:%M:%S")
                            })
                            
                            cv2.rectangle(vis_frame, (0, 0), (width-1, height-1), (0, 0, 255), 10)
                        elif final_count >= warning_threshold:
                            status = "⚠️ WARNING"
                            status_color = "orange"
                        else:
                            status = "✅ NORMAL"
                            status_color = "green"
                    else:
                        status = "ℹ️ MONITORING"
                        status_color = "blue"
                    
                    frame_time = time.time() - frame_start_time
                    current_fps = 1.0 / frame_time if frame_time > 0 else 0
                    fps_history.append(current_fps)
                    if len(fps_history) > 30:
                        fps_history.pop(0)
                    avg_fps = np.mean(fps_history) if fps_history else 0
                    
                    vis_frame_rgb = cv2.cvtColor(vis_frame, cv2.COLOR_BGR2RGB)
                    frame_placeholder.image(vis_frame_rgb, channels="RGB", use_container_width=True)
                    
                    with metrics_placeholder.container():
                        m1, m2, m3 = st.columns(3)
                        m1.metric("👥 Total", final_count)
                        m2.metric("🎯 YOLO", yolo_count)
                        m3.metric("🗺️ CSRNet", csrnet_count)
                    
                    status_placeholder.markdown(f"### Status: :{status_color}[{status}]")
                    
                    if len(st.session_state.webcam_stats['count_history']) > 1:
                        chart_placeholder.line_chart(
                            st.session_state.webcam_stats['count_history'][-50:],
                            height=200
                        )
                    
                    elapsed_time = time.time() - st.session_state.webcam_stats['start_time']
                    avg_count = np.mean(st.session_state.webcam_stats['count_history']) if st.session_state.webcam_stats['count_history'] else 0
                    
                    with stats_placeholder.container():
                        st.markdown("### 📊 Live Statistics")
                        stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
                        stat_col1.metric("⏱️ Duration", f"{elapsed_time:.0f}s")
                        stat_col2.metric("📈 Max Count", st.session_state.webcam_stats['max_count'])
                        stat_col3.metric("📊 Avg Count", f"{avg_count:.1f}")
                        stat_col4.metric("🎬 FPS", f"{avg_fps:.1f}")
                        
                        if st.session_state.webcam_stats['alarm_log']:
                            st.warning(f"🚨 **Alarm Triggers:** {len(st.session_state.webcam_stats['alarm_log'])}")
                
                time.sleep(0.03)
                
        finally:
            cap.release()
            st.session_state.webcam_active = False
    
    elif st.session_state.webcam_stats['count_history']:
        st.markdown("---")
        st.header("📈 Session Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Frames", st.session_state.webcam_stats['total_frames'])
        col2.metric("Max Count", st.session_state.webcam_stats['max_count'])
        
        avg_count = np.mean(st.session_state.webcam_stats['count_history'])
        col3.metric("Avg Count", f"{avg_count:.1f}")
        col4.metric("Alarm Events", len(st.session_state.webcam_stats['alarm_log']))
        
        if st.session_state.webcam_stats['alarm_log']:
            st.subheader("🚨 Alarm Events")
            st.dataframe(st.session_state.webcam_stats['alarm_log'], use_container_width=True)
            
            alarm_text = "WEBCAM ALARM LOG\n" + "="*50 + "\n\n"
            for event in st.session_state.webcam_stats['alarm_log']:
                alarm_text += f"[{event['timestamp']}] Frame {event['frame']}: {event['count']} people\n"
            
            st.download_button(
                label="📥 Download Alarm Log",
                data=alarm_text,
                file_name=f"webcam_alarm_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )
        
        if st.session_state.webcam_stats['count_history']:
            st.subheader("📊 Count History")
            st.line_chart(st.session_state.webcam_stats['count_history'])
            
            st.subheader("📈 Count Distribution")
            st.bar_chart(np.bincount(st.session_state.webcam_stats['count_history']))


def process_video(video_path, confidence_threshold, fusion_method, 
                 enable_alarm, alarm_threshold, warning_threshold,
                 process_every_n_frames, max_frames):
    
    with st.spinner("🔄 Loading detection models..."):
        detector = HybridCrowdDetector(
            csrnet_path=None, use_yolo=True,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            fusion_method=fusion_method
        )
    
    st.success("✅ Models loaded successfully!")
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        st.error("❌ Error: Could not open video file")
        return
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    frames_to_process = min(total_frames, max_frames)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        frame_placeholder = st.empty()
    
    with col2:
        metrics_placeholder = st.empty()
        status_placeholder = st.empty()
        chart_placeholder = st.empty()
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    frame_count = 0
    processed_count = 0
    alarm_log = []
    count_history = []
    
    start_time = time.time()
    
    while frame_count < frames_to_process:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        if frame_count % process_every_n_frames != 0:
            continue
        
        processed_count += 1
        
        boxes, confidences, final_count, yolo_count, csrnet_count = detector.process_frame(
            frame, confidence_threshold
        )
        
        vis_frame = detector.draw_detections(frame, boxes, confidences)
        
        if enable_alarm:
            if final_count >= alarm_threshold:
                status = "🚨 ALARM"
                status_color = "red"
                
                alarm_log.append({
                    'frame': frame_count,
                    'count': final_count,
                    'timestamp': datetime.now().strftime("%H:%M:%S")
                })
                
                cv2.rectangle(vis_frame, (0, 0), (width-1, height-1), (0, 0, 255), 10)
                
            elif final_count >= warning_threshold:
                status = "⚠️ WARNING"
                status_color = "orange"
            else:
                status = "✅ NORMAL"
                status_color = "green"
        else:
            status = "ℹ️ MONITORING"
            status_color = "blue"
        
        count_history.append(final_count)
        
        overlay_text = f"Count: {final_count}"
        cv2.putText(vis_frame, overlay_text, (20, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
        
        vis_frame_rgb = cv2.cvtColor(vis_frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(vis_frame_rgb, channels="RGB", use_container_width=True)
        
        with metrics_placeholder.container():
            m1, m2, m3 = st.columns(3)
            m1.metric("👥 Total Count", final_count)
            m2.metric("🎯 YOLO", yolo_count)
            m3.metric("🗺️ CSRNet", csrnet_count)
        
        status_placeholder.markdown(f"### Status: :{status_color}[{status}]")
        
        if len(count_history) > 1:
            chart_placeholder.line_chart(count_history, height=200)
        
        progress = frame_count / frames_to_process
        progress_bar.progress(progress)
        
        elapsed = time.time() - start_time
        processing_fps = processed_count / elapsed if elapsed > 0 else 0
        status_text.text(f"Frame {frame_count}/{frames_to_process} | Processing: {processing_fps:.1f} FPS")
    
    cap.release()
    
    st.success("✅ Processing Complete!")
    
    st.markdown("---")
    st.header("📈 Summary Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Frames Processed", processed_count)
    col2.metric("Max Count", max(count_history) if count_history else 0)
    col3.metric("Avg Count", f"{np.mean(count_history):.1f}" if count_history else 0)
    col4.metric("Alarm Triggers", len(alarm_log))
    
    if alarm_log:
        st.header("🚨 Alarm Events")
        st.dataframe(alarm_log, use_container_width=True)
        
        alarm_text = "VIDEO ALARM LOG\n" + "="*50 + "\n\n"
        for event in alarm_log:
            alarm_text += f"[{event['timestamp']}] Frame {event['frame']}: {event['count']} people\n"
        
        st.download_button(
            label="📥 Download Alarm Log",
            data=alarm_text,
            file_name=f"alarm_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
            use_container_width=True
        )
    
    if count_history:
        st.header("📊 Count Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Count Over Time")
            st.line_chart(count_history)
        
        with col2:
            st.subheader("📊 Count Distribution")
            st.bar_chart(np.bincount(count_history))


# =============================================================================
# MAIN APP
# =============================================================================

def main():
    st.set_page_config(
        page_title="Crowd Detection System",
        page_icon="👥",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS with futuristic dark theme
    st.markdown("""
    <style>
    :root {
        --bg-dark: #0D0D0F;
        --bg-mid: #1A1C20;
        --bg-light: #26282C;
        --neon-violet: #7A5CFF;
        --neon-cyan: #00D4FF;
        --neon-purple: #9B59B6;
    }
    
    .stApp {
        background: linear-gradient(135deg, #0D0D0F 0%, #1A1C20 50%, #26282C 100%);
        color: #E0E0E0 !important;
    }
    
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0D0D0F 0%, #1A1C20 50%, #26282C 100%);
        border-right: 1px solid rgba(122, 92, 255, 0.2);
    }
    
    h1, h2, h3 {
        color: #7A5CFF !important;
        text-shadow: 0 0 20px rgba(122, 92, 255, 0.6);
    }
    
    [data-testid="stMetricValue"] {
        color: #00D4FF !important;
        font-weight: bold;
        text-shadow: 0 0 10px rgba(0, 212, 255, 0.4);
    }
    
    [data-testid="stMetricLabel"] {
        color: #B0B3B8 !important;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #7A5CFF 0%, #00D4FF 100%);
        color: #0D0D0F !important;
        border: none;
        font-weight: 700;
        box-shadow: 0 4px 20px rgba(122, 92, 255, 0.4), 0 0 40px rgba(0, 212, 255, 0.2);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        border-radius: 12px;
        padding: 12px 24px;
        font-size: 1rem;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #00D4FF 0%, #9B59B6 100%);
        box-shadow: 0 8px 30px rgba(0, 212, 255, 0.6), 0 0 60px rgba(122, 92, 255, 0.4);
        transform: translateY(-3px);
    }
    
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #7A5CFF 0%, #00D4FF 100%);
        color: #0D0D0F !important;
    }
    
    .stButton > button[kind="secondary"] {
        background: linear-gradient(135deg, #26282C 0%, #1A1C20 100%);
        color: #00D4FF !important;
        border: 2px solid #7A5CFF;
    }
    
    .stAlert {
        background-color: rgba(26, 28, 32, 0.8);
        border-left: 4px solid #7A5CFF;
        color: #B0B3B8;
        backdrop-filter: blur(10px);
    }
    
    [data-baseweb="notification"][kind="success"] {
        background-color: rgba(0, 212, 255, 0.15);
        border-left: 4px solid #00D4FF;
        color: #00D4FF;
    }
    
    [data-baseweb="notification"][kind="warning"] {
        background-color: rgba(155, 89, 182, 0.15);
        border-left: 4px solid #9B59B6;
        color: #9B59B6;
    }
    
    [data-baseweb="notification"][kind="error"] {
        background-color: rgba(255, 107, 107, 0.15);
        border-left: 4px solid #FF6B6B;
        color: #FF6B6B;
    }
    
    [data-testid="stFileUploader"] {
        background-color: rgba(26, 28, 32, 0.6);
        border: 2px dashed #7A5CFF;
        border-radius: 16px;
        backdrop-filter: blur(10px);
    }
    
    [data-testid="stFileUploader"] label {
        color: #00D4FF !important;
    }
    
    .stProgress > div > div {
        background: linear-gradient(90deg, #7A5CFF 0%, #00D4FF 50%, #9B59B6 100%);
    }
    
    .stSlider > div > div > div {
        background: linear-gradient(90deg, #7A5CFF 0%, #00D4FF 100%);
    }
    
    .stSlider label {
        color: #B0B3B8 !important;
    }
    
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input {
        background-color: rgba(26, 28, 32, 0.8);
        border: 1px solid rgba(122, 92, 255, 0.3);
        color: #E0E0E0 !important;
        border-radius: 8px;
    }
    
    .stTextInput label,
    .stNumberInput label {
        color: #B0B3B8 !important;
    }
    
    .stSelectbox > div > div {
        background-color: rgba(26, 28, 32, 0.8);
        border: 1px solid rgba(122, 92, 255, 0.3);
        color: #E0E0E0 !important;
        border-radius: 8px;
    }
    
    .stSelectbox label {
        color: #B0B3B8 !important;
    }
    
    .element-container {
        color: #B0B3B8;
    }
    
    [data-testid="stDataFrame"] {
        background-color: rgba(26, 28, 32, 0.7);
        border: 1px solid rgba(122, 92, 255, 0.3);
        border-radius: 12px;
    }
    
    .streamlit-expanderHeader {
        background-color: rgba(26, 28, 32, 0.6);
        border: 1px solid rgba(122, 92, 255, 0.3);
        color: #00D4FF !important;
        border-radius: 8px;
    }
    
    a {
        color: #00D4FF !important;
    }
    
    a:hover {
        color: #7A5CFF !important;
    }
    
    .stCheckbox {
        color: #B0B3B8 !important;
    }
    
    .stCheckbox label {
        color: #B0B3B8 !important;
    }
    
    .stRadio label {
        color: #B0B3B8 !important;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        background-color: rgba(26, 28, 32, 0.5);
    }
    
    .stTabs [data-baseweb="tab"] {
        color: #B0B3B8;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: rgba(122, 92, 255, 0.3) !important;
        color: #00D4FF !important;
        border-bottom: 2px solid #7A5CFF;
    }
    
    .stSpinner > div {
        border-top-color: #7A5CFF !important;
    }
    
    p, span, div {
        color: #B0B3B8;
    }
    
    .stMarkdown {
        color: #B0B3B8;
    }
    
    .stMarkdown, .stText {
        background-color: transparent !important;
    }
    
    [data-testid="stHeader"] {
        background-color: transparent;
    }
    
    [data-testid="stSidebar"] * {
        color: #B0B3B8;
    }
    
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #7A5CFF !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    if 'page' not in st.session_state:
        st.session_state.page = "home"
    
    if st.session_state.page == "home":
        show_home_page()
    elif st.session_state.page == "image":
        show_image_page()
    elif st.session_state.page == "video":
        show_video_page()
    elif st.session_state.page == "webcam":
        show_webcam_page()


if __name__ == '__main__':
    main()