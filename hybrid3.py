import cv2
import torch
import torch.nn as nn
import numpy as np
import time
from datetime import datetime
import winsound  # For Windows alarm sound (use 'pygame' or 'playsound' for cross-platform)

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
# ALARM SYSTEM
# =============================================================================

class AlarmSystem:
    """Manages crowd threshold alerts and alarms"""
    
    def __init__(self, threshold, warning_threshold=None, cooldown_seconds=5):
        """
        Args:
            threshold: Maximum allowed people before alarm triggers
            warning_threshold: Warning level (optional, default 80% of threshold)
            cooldown_seconds: Minimum seconds between alarm sounds
        """
        self.threshold = threshold
        self.warning_threshold = warning_threshold or int(threshold * 0.8)
        self.cooldown_seconds = cooldown_seconds
        
        self.alarm_triggered = False
        self.last_alarm_time = 0
        self.violation_count = 0
        self.alarm_log = []
        
    def check_threshold(self, count):
        """Check if count exceeds thresholds and return status"""
        current_time = time.time()
        status = "NORMAL"
        
        # Check if threshold exceeded
        if count >= self.threshold:
            status = "ALARM"
            
            # Trigger alarm with cooldown
            if current_time - self.last_alarm_time > self.cooldown_seconds:
                self.trigger_alarm(count)
                self.last_alarm_time = current_time
                self.violation_count += 1
            
            self.alarm_triggered = True
            
        elif count >= self.warning_threshold:
            status = "WARNING"
            self.alarm_triggered = False
            
        else:
            status = "NORMAL"
            self.alarm_triggered = False
        
        return status
    
    def trigger_alarm(self, count):
        """Sound the alarm and log the event"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Log the event
        self.alarm_log.append({
            'timestamp': timestamp,
            'count': count,
            'threshold': self.threshold
        })
        
        # Console alert
        print(f"\n{'🚨'*20}")
        print(f"⚠️  ALARM TRIGGERED! ⚠️")
        print(f"Timestamp: {timestamp}")
        print(f"Count: {count} people (Threshold: {self.threshold})")
        print(f"{'🚨'*20}\n")
        
        # Sound alarm (Windows)
        try:
            # Beep: frequency=2500Hz, duration=500ms
            winsound.Beep(2500, 500)
        except:
            # Fallback for non-Windows or if winsound fails
            print('\a')  # System beep
    
    def get_status_color(self, status):
        """Return BGR color for status"""
        if status == "ALARM":
            return (0, 0, 255)  # Red
        elif status == "WARNING":
            return (0, 165, 255)  # Orange
        else:
            return (0, 255, 0)  # Green
    
    def save_log(self, filename='alarm_log.txt'):
        """Save alarm log to file"""
        if self.alarm_log:
            with open(filename, 'w') as f:
                f.write("CROWD ALARM LOG\n")
                f.write("="*60 + "\n\n")
                for event in self.alarm_log:
                    f.write(f"[{event['timestamp']}] Count: {event['count']} (Threshold: {event['threshold']})\n")
            print(f"📝 Alarm log saved to {filename}")


# =============================================================================
# HYBRID DETECTOR (YOLO + CSRNet + ALARM)
# =============================================================================

class HybridCrowdDetector:
    """Combines YOLO for bounding boxes + CSRNet for density estimation + Alarm System"""
    
    def __init__(self, csrnet_path=None, use_yolo=True, device='cuda', 
                 alarm_threshold=None, use_alarm=True, fusion_method='max'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.use_yolo = use_yolo
        self.fusion_method = fusion_method  # 'max', 'average', 'yolo', 'csrnet'
        
        # Load YOLO for person detection
        if use_yolo:
            try:
                from ultralytics import YOLO
                print("Loading YOLOv8 for person detection...")
                self.yolo = YOLO('yolov8m.pt')
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
        
        # Initialize alarm system
        self.use_alarm = use_alarm and alarm_threshold is not None
        if self.use_alarm:
            self.alarm_system = AlarmSystem(
                threshold=alarm_threshold,
                warning_threshold=int(alarm_threshold * 0.8),
                cooldown_seconds=5
            )
            print(f"✓ Alarm system initialized (Threshold: {alarm_threshold} people)")
        
        self.fps_history = []
        
    def detect_people_yolo(self, frame):
        """Detect people using YOLO and return bounding boxes"""
        results = self.yolo(frame, verbose=False)
        
        boxes = []
        confidences = []
        
        for result in results:
            for box in result.boxes:
                if int(box.cls[0]) == 0:  # Class 0 is 'person'
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
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_normalized = frame_rgb.astype(np.float32) / 255.0
            frame_tensor = torch.from_numpy(frame_normalized).permute(2, 0, 1).unsqueeze(0).float()
            frame_tensor = frame_tensor.to(self.device)
            
            output = self.csrnet(frame_tensor)
            count = output.sum().item()
            density_map = output.squeeze().cpu().numpy()
            
        return count, density_map
    
    def fuse_counts(self, yolo_count, csrnet_count):
        """Combine YOLO and CSRNet counts into single prediction"""
        if self.fusion_method == 'max':
            # Use the higher count (safer for crowd control)
            return max(yolo_count, csrnet_count)
        elif self.fusion_method == 'average':
            # Average both counts
            return int((yolo_count + csrnet_count) / 2)
        elif self.fusion_method == 'yolo':
            # Trust YOLO only
            return yolo_count
        elif self.fusion_method == 'csrnet':
            # Trust CSRNet only
            return csrnet_count
        else:
            return max(yolo_count, csrnet_count)
    
    def create_visualization(self, frame, boxes, confidences, final_count, 
                           density_map, fps, alarm_status=None):
        """Create visualization with bounding boxes, unified count, and alarm status"""
        vis_frame = frame.copy()
        h, w = vis_frame.shape[:2]
        
        # Draw bounding boxes
        for box, conf in zip(boxes, confidences):
            x1, y1, x2, y2 = box
            
            # Color based on confidence
            if conf > 0.7:
                color = (0, 255, 0)  # Green
            elif conf > 0.5:
                color = (0, 255, 255)  # Yellow
            else:
                color = (0, 165, 255)  # Orange
            
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
            
            label = f"{conf:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(vis_frame, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            cv2.putText(vis_frame, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Create info panel
        panel_height = 180 if self.use_alarm else 140
        panel = np.zeros((panel_height, w, 3), dtype=np.uint8)
        
        y_offset = 35
        
        # Title
        cv2.putText(panel, "Real-time Person Detection", (15, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        y_offset += 45
        
        # UNIFIED COUNT - Large and prominent
        count_text = f"Total Count: {final_count} people"
        cv2.putText(panel, count_text, (15, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
        y_offset += 50
        
        # Alarm status
        if self.use_alarm and alarm_status:
            status_color = self.alarm_system.get_status_color(alarm_status)
            status_text = f"Status: {alarm_status}"
            
            # Add warning/alarm indicator
            if alarm_status == "ALARM":
                status_text += f" (Limit: {self.alarm_system.threshold})"
                # Flashing red background
                if int(time.time() * 2) % 2:  # Blink effect
                    cv2.rectangle(panel, (10, y_offset - 25), (w - 10, y_offset + 5), 
                                (0, 0, 255), -1)
                    cv2.putText(panel, status_text, (15, y_offset),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                else:
                    cv2.putText(panel, status_text, (15, y_offset),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.9, status_color, 2)
            elif alarm_status == "WARNING":
                status_text += f" (Limit: {self.alarm_system.threshold})"
                cv2.putText(panel, status_text, (15, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, status_color, 2)
            else:
                cv2.putText(panel, status_text, (15, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, status_color, 2)
        
        # FPS
        fps_text = f"FPS: {fps:.1f}"
        cv2.putText(panel, fps_text, (w - 150, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Controls
        cv2.putText(panel, "'q' quit | 's' screenshot | 'd' density | 'l' save log", 
                   (15, panel_height - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Combine panel with frame
        vis_frame = np.vstack([panel, vis_frame])
        
        return vis_frame
    
    def run(self, camera_id=0, confidence_threshold=0.3, show_density=False):
        """Run real-time detection with alarm system"""
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            print(f"❌ Error: Could not open camera/video {camera_id}")
            return
        
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"\n{'='*60}")
        print(f"📹 Camera: {frame_width}x{frame_height}")
        print(f"🎯 Detection Mode: {'YOLO + CSRNet' if (self.use_yolo and self.use_csrnet) else 'YOLO only' if self.use_yolo else 'CSRNet only'}")
        print(f"🔀 Fusion Method: {self.fusion_method.upper()}")
        if self.use_alarm:
            print(f"🚨 Alarm System: Enabled (Threshold: {self.alarm_system.threshold})")
            print(f"⚠️  Warning Level: {self.alarm_system.warning_threshold}")
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
                    filtered = [(b, c) for b, c in zip(boxes, confidences) if c >= confidence_threshold]
                    boxes = [b for b, c in filtered]
                    confidences = [c for b, c in filtered]
                    yolo_count = len(boxes)
                else:
                    boxes, confidences = [], []
                    yolo_count = 0
                
                # Get CSRNet estimate
                csrnet_count, density_map = self.predict_csrnet_count(frame)
                
                # FUSE into single count
                final_count = self.fuse_counts(yolo_count, int(csrnet_count))
                
                # Check alarm threshold with final count
                alarm_status = None
                if self.use_alarm:
                    alarm_status = self.alarm_system.check_threshold(final_count)
                
                # Calculate FPS
                frame_time = time.time() - frame_start
                current_fps = 1.0 / frame_time if frame_time > 0 else 0
                self.fps_history.append(current_fps)
                if len(self.fps_history) > 30:
                    self.fps_history.pop(0)
                avg_fps = np.mean(self.fps_history)
                
                # Create visualization with unified count
                vis_frame = self.create_visualization(
                    frame, boxes, confidences, final_count, density_map, 
                    avg_fps, alarm_status
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
                        panel_height = 180 if self.use_alarm else 140
                        vis_bottom = vis_frame[panel_height:, :, :]
                        vis_bottom[:] = cv2.addWeighted(vis_bottom, 0.7, density_colored, 0.3, 0)
                
                cv2.imshow('Person Detection & Counting', vis_frame)
                
                # Stats every 30 frames
                frame_count += 1
                if frame_count % 30 == 0:
                    alarm_str = f"| {alarm_status}" if self.use_alarm else ""
                    print(f"📊 Frame {frame_count:4d} | Count: {final_count:3d} {alarm_str} | FPS: {avg_fps:5.1f}")
                
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
                elif key == ord('l') and self.use_alarm:
                    self.alarm_system.save_log()
                    
        except KeyboardInterrupt:
            print("\n⏹️  Interrupted")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            
            # Save alarm log on exit
            if self.use_alarm:
                self.alarm_system.save_log()
            
            total_time = time.time() - start_time
            print(f"\n{'='*60}")
            print(f"📈 SESSION SUMMARY")
            print(f"{'='*60}")
            print(f"  Frames: {frame_count}")
            print(f"  Time: {total_time:.1f}s")
            print(f"  Avg FPS: {frame_count/total_time:.1f}")
            if screenshot_count:
                print(f"  Screenshots: {screenshot_count}")
            if self.use_alarm:
                print(f"  Alarm Triggers: {self.alarm_system.violation_count}")
            print(f"{'='*60}\n")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "="*60)
    print(" "*8 + "PERSON DETECTION & CROWD COUNTING WITH ALARM")
    print("="*60)
    
    # Configuration
    CSRNET_MODEL = 'csrnet_partA_best.pth'  # Set to None to use YOLO only
    CAMERA_ID = 0  # 0 for webcam, 'video.mp4' for file
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    CONFIDENCE_THRESHOLD = 0.3  # Adjust detection sensitivity (0.1-0.9)
    SHOW_DENSITY = False  # Show density map overlay
    
    # ALARM SETTINGS
    USE_ALARM = True  # Enable/disable alarm system
    ALARM_THRESHOLD = 10  # Maximum number of people before alarm triggers
    
    # FUSION METHOD: How to combine YOLO and CSRNet counts
    # Options: 'max' (safer), 'average', 'yolo' (YOLO only), 'csrnet' (CSRNet only)
    FUSION_METHOD = 'max'
    
    print(f"\n⚙️  Configuration:")
    print(f"  YOLO Detection: Enabled")
    print(f"  CSRNet Counting: {'Enabled' if CSRNET_MODEL else 'Disabled'}")
    print(f"  Fusion Method: {FUSION_METHOD.upper()}")
    print(f"  Device: {DEVICE}")
    print(f"  Camera: {CAMERA_ID}")
    print(f"  Confidence Threshold: {CONFIDENCE_THRESHOLD}")
    if USE_ALARM:
        print(f"  🚨 Alarm Threshold: {ALARM_THRESHOLD} people")
        print(f"  ⚠️  Warning at: {int(ALARM_THRESHOLD * 0.8)} people")
    print("="*60 + "\n")
    
    # Create detector with alarm
    detector = HybridCrowdDetector(
        csrnet_path=CSRNET_MODEL,
        use_yolo=True,
        device=DEVICE,
        alarm_threshold=ALARM_THRESHOLD if USE_ALARM else None,
        use_alarm=USE_ALARM,
        fusion_method=FUSION_METHOD
    )
    
    # Run
    detector.run(
        camera_id=CAMERA_ID,
        confidence_threshold=CONFIDENCE_THRESHOLD,
        show_density=SHOW_DENSITY
    )


if __name__ == '__main__':
    main()