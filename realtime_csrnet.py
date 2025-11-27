# Just change the visualization function in your existing code
# Replace the create_visualization function with this:

def create_visualization(self, frame, count, density_map, fps):
    """Clean view: Just camera feed with count overlay"""
    overlay = frame.copy()
    h, w = overlay.shape[:2]
    
    # Semi-transparent top panel for info
    panel_height = 120
    panel = overlay[:panel_height].copy()
    panel[:] = (0, 0, 0)
    overlay[:panel_height] = cv2.addWeighted(overlay[:panel_height], 0.4, panel, 0.6, 0)
    
    # Count (large, centered)
    count_text = f"People Count: {int(count)}"
    text_size = cv2.getTextSize(count_text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
    text_x = (w - text_size[0]) // 2
    cv2.putText(overlay, count_text, (text_x, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
    
    # FPS (top right)
    cv2.putText(overlay, f"FPS: {fps:.1f}", (w - 150, 100), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Instructions (bottom)
    cv2.putText(overlay, "Press 'q' to quit", (20, h - 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    return overlay