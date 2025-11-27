import cv2
import torch
import torch.nn as nn
import numpy as np
import time

# =============================================================================
# CSRNet MODEL (For crowd counting verification)
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
# HYBRID DETECTOR (YOLO + CSRNet)
# =============================================================================

class HybridCrowdDetector:
    """Combines YOLO for bounding boxes + CSRNet for density estimation"""
    
    def __init__(self, csrnet_path=None, use_yolo=True, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.use_yolo = use_yolo
        
        # Load YOLO for person detection
        if use_yolo:
            try:
                from ultralytics import YOLO
                print("Loading YOLOv8 for person detection...")
                self.yolo = YOLO('yolov8m.pt')  # Nano version (fastest)
                print("✓ YOLOv8 loaded successfully")
            except ImportError:
                print("⚠️  ultralytics not installed. Install with: pip install ultralytics")
                print("   Falling back to CSRNet only mode")
                self.use_yolo = False
        
        # Load CSRNet for density estimation (optional)
        self.use_csrnet = csrnet_path is not None
        if self.use_csrnet:
            print(f"Loading CSRNet from {csrnet_path}...")
            self.csrnet = CSRNet().to(self.device)
            checkpoint = torch.load(csrnet_path, map_location=self.device)
            self.csrnet.load_state_dict(checkpoint['model_state_dict'])
            self.csrnet.eval()
            print("✓ CSRNet loaded successfully")
        
        self.fps_history = []
        
    def detect_people_yolo(self, frame):
        """Detect people using YOLO and return bounding boxes"""
        results = self.yolo(frame, verbose=False)
        
        boxes = []
        confidences = []
        
        for result in results:
            for box in result.boxes:
                # Class 0 is 'person' in COCO dataset
                if int(box.cls[0]) == 0:
                    # Get box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0])
                    
                    boxes.append([int(x1), int(y1), int(x2), int(y2)])
                    confidences.append(conf)
        
        return boxes, confidences
    
    def predict_csrnet_count(self, frame):
        """Get count from CSRNet density map"""
        if not self.use_csrnet:
            return 0, None
        
        with torch.no_grad():
            # Preprocess
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_normalized = frame_rgb.astype(np.float32) / 255.0
            frame_tensor = torch.from_numpy(frame_normalized).permute(2, 0, 1).unsqueeze(0).float()
            frame_tensor = frame_tensor.to(self.device)
            
            # Predict
            output = self.csrnet(frame_tensor)
            count = output.sum().item()
            density_map = output.squeeze().cpu().numpy()
            
        return count, density_map
    
    def create_visualization(self, frame, boxes, confidences, csrnet_count, density_map, fps):
        """Create visualization with bounding boxes and info"""
        vis_frame = frame.copy()
        h, w = vis_frame.shape[:2]
        
        # Draw bounding boxes
        for box, conf in zip(boxes, confidences):
            x1, y1, x2, y2 = box
            
            # Color based on confidence
            if conf > 0.7:
                color = (0, 255, 0)  # Green for high confidence
            elif conf > 0.5:
                color = (0, 255, 255)  # Yellow for medium
            else:
                color = (0, 165, 255)  # Orange for low
            
            # Draw box
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw confidence
            label = f"{conf:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(vis_frame, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            cv2.putText(vis_frame, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Create info panel
        panel_height = 160
        panel = np.zeros((panel_height, w, 3), dtype=np.uint8)
        
        y_offset = 35
        
        # Title
        cv2.putText(panel, "Real-time Person Detection", (15, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        y_offset += 40
        
        # YOLO count (detected boxes)
        yolo_text = f"Detected: {len(boxes)} people"
        cv2.putText(panel, yolo_text, (15, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        y_offset += 40
        
        # CSRNet count (if available)
        if self.use_csrnet:
            csrnet_text = f"CSRNet Estimate: {int(csrnet_count)} people"
            cv2.putText(panel, csrnet_text, (15, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 200, 255), 2)
            y_offset += 35
        
        # FPS
        fps_text = f"FPS: {fps:.1f}"
        cv2.putText(panel, fps_text, (w - 150, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Controls
        cv2.putText(panel, "'q' quit | 's' screenshot | 'd' toggle density", (15, panel_height - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Combine panel with frame
        vis_frame = np.vstack([panel, vis_frame])
        
        return vis_frame
    
    def run(self, camera_id=0, confidence_threshold=0.3, show_density=False):
        """Run real-time detection"""
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            print(f"❌ Error: Could not open camera/video {camera_id}")
            return
        
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"\n{'='*60}")
        print(f"📹 Camera: {frame_width}x{frame_height}")
        print(f"🎯 Detection Mode: {'YOLO + CSRNet' if (self.use_yolo and self.use_csrnet) else 'YOLO only' if self.use_yolo else 'CSRNet only'}")
        print(f"{'='*60}\n")
        print("🚀 Starting detection... Press 'q' to quit\n")
        
        frame_count = 0
        start_time = time.time()
        screenshot_count = 0
        show_density_overlay = show_density
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_start = time.time()
                
                # Detect people with YOLO
                if self.use_yolo:
                    boxes, confidences = self.detect_people_yolo(frame)
                    # Filter by confidence
                    filtered = [(b, c) for b, c in zip(boxes, confidences) if c >= confidence_threshold]
                    boxes = [b for b, c in filtered]
                    confidences = [c for b, c in filtered]
                else:
                    boxes, confidences = [], []
                
                # Get CSRNet estimate
                csrnet_count, density_map = self.predict_csrnet_count(frame)
                
                # Calculate FPS
                frame_time = time.time() - frame_start
                current_fps = 1.0 / frame_time if frame_time > 0 else 0
                self.fps_history.append(current_fps)
                if len(self.fps_history) > 30:
                    self.fps_history.pop(0)
                avg_fps = np.mean(self.fps_history)
                
                # Create visualization
                vis_frame = self.create_visualization(
                    frame, boxes, confidences, csrnet_count, density_map, avg_fps
                )
                
                # Optional: overlay density map
                if show_density_overlay and density_map is not None:
                    h, w = frame.shape[:2]
                    density_resized = cv2.resize(density_map, (w, h))
                    if density_resized.max() > 0:
                        density_normalized = density_resized / density_resized.max()
                        density_colored = cv2.applyColorMap(
                            (density_normalized * 255).astype(np.uint8), cv2.COLORMAP_JET
                        )
                        # Overlay on bottom part of frame
                        vis_bottom = vis_frame[160:, :, :]
                        vis_bottom[:] = cv2.addWeighted(vis_bottom, 0.7, density_colored, 0.3, 0)
                
                cv2.imshow('Person Detection & Counting', vis_frame)
                
                # Stats every 30 frames
                frame_count += 1
                if frame_count % 30 == 0:
                    yolo_str = f"YOLO: {len(boxes):3d}" if self.use_yolo else ""
                    csrnet_str = f"CSRNet: {int(csrnet_count):3d}" if self.use_csrnet else ""
                    print(f"📊 Frame {frame_count:4d} | {yolo_str} {csrnet_str} | FPS: {avg_fps:5.1f}")
                
                # Keyboard controls
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\n⏹️  Stopping...")
                    break
                elif key == ord('s'):
                    screenshot_count += 1
                    filename = f'detection_{screenshot_count}.png'
                    cv2.imwrite(filename, vis_frame)
                    print(f"📸 Saved: {filename}")
                elif key == ord('d'):
                    show_density_overlay = not show_density_overlay
                    print(f"🗺️  Density overlay: {'ON' if show_density_overlay else 'OFF'}")
                    
        except KeyboardInterrupt:
            print("\n⏹️  Interrupted")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            
            total_time = time.time() - start_time
            print(f"\n{'='*60}")
            print(f"📈 SESSION SUMMARY")
            print(f"{'='*60}")
            print(f"  Frames: {frame_count}")
            print(f"  Time: {total_time:.1f}s")
            print(f"  Avg FPS: {frame_count/total_time:.1f}")
            if screenshot_count:
                print(f"  Screenshots: {screenshot_count}")
            print(f"{'='*60}\n")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "="*60)
    print(" "*12 + "PERSON DETECTION & CROWD COUNTING")
    print("="*60)
    
    # Configuration
    CSRNET_MODEL = 'csrnet_partA_best.pth'  # Set to None to use YOLO only
    CAMERA_ID = 0  # 0 for webcam, 'video.mp4' for file
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    CONFIDENCE_THRESHOLD = 0.3  # Adjust detection sensitivity (0.1-0.9)
    SHOW_DENSITY = False  # Show density map overlay
    
    print(f"\n⚙️  Configuration:")
    print(f"  YOLO Detection: Enabled")
    print(f"  CSRNet Counting: {'Enabled' if CSRNET_MODEL else 'Disabled'}")
    print(f"  Device: {DEVICE}")
    print(f"  Camera: {CAMERA_ID}")
    print(f"  Confidence Threshold: {CONFIDENCE_THRESHOLD}")
    print("="*60 + "\n")
    
    # Create detector
    detector = HybridCrowdDetector(
        csrnet_path=CSRNET_MODEL,
        use_yolo=True,
        device=DEVICE
    )
    
    # Run
    detector.run(
        camera_id=CAMERA_ID,
        confidence_threshold=CONFIDENCE_THRESHOLD,
        show_density=SHOW_DENSITY
    )


if __name__ == '__main__':
    main()