import cv2
import numpy as np
import pyrealsense2 as rs

# Global variables to store click information
clicked_point = None
picked_color_bgr = None
picked_color_rgb = None
current_frame = None

def get_pixel_color(event, x, y, flags, param):
    """
    Mouse callback function to get the color of the clicked pixel.
    """
    global clicked_point, picked_color_bgr, picked_color_rgb, current_frame

    if event == cv2.EVENT_LBUTTONDOWN and current_frame is not None:
        # Store the clicked point coordinates
        clicked_point = (x, y)
        
        # Get the BGR color of the clicked pixel
        if 0 <= y < current_frame.shape[0] and 0 <= x < current_frame.shape[1]:
            picked_color_bgr = current_frame[y, x].tolist()
            # Convert to RGB for display purposes
            picked_color_rgb = (picked_color_bgr[2], picked_color_bgr[1], picked_color_bgr[0])

# Initialize RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()

# Configure color stream
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

print("Starting RealSense camera...")
try:
    # Start streaming
    pipeline.start(config)
    
    # Create windows
    window_name = 'RealSense Color Picker - Click to pick color'
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, get_pixel_color)
    
    # Create additional window for color swatch
    cv2.namedWindow('Color Information')
    
    print("Instructions:")
    print(" - Click anywhere on the image to pick a color")
    print(" - Press 'c' to clear the current selection")
    print(" - Press 'q' or ESC to quit")
    
    # Main loop
    while True:
        # Wait for a coherent pair of frames
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue
        
        # Convert images to numpy arrays
        frame = np.asanyarray(color_frame.get_data())
        current_frame = frame.copy()
        
        # If a color has been picked, draw information
        if clicked_point is not None and picked_color_bgr is not None:
            # Draw a circle at the clicked point
            cv2.circle(frame, clicked_point, 8, (0, 0, 255), 2)  # Red circle outline
            cv2.circle(frame, clicked_point, 6, (255, 255, 255), 1)  # White inner circle
            
            # Display color information on the frame
            info_text = f"BGR: {picked_color_bgr}"
            cv2.putText(frame, info_text, (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
            cv2.putText(frame, info_text, (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            
            # Create color information display
            info_display = np.zeros((200, 400, 3), dtype=np.uint8)
            info_display[:] = picked_color_bgr
            
            # Display color values
            cv2.putText(info_display, f"B: {picked_color_bgr[0]}", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(info_display, f"G: {picked_color_bgr[1]}", (20, 80), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(info_display, f"R: {picked_color_bgr[2]}", (20, 120), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Display hex value
            hex_color = "#{:02x}{:02x}{:02x}".format(picked_color_bgr[2], 
                                                      picked_color_bgr[1], 
                                                      picked_color_bgr[0])
            cv2.putText(info_display, f"HEX: {hex_color}", (20, 160), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Show the color information display
            cv2.imshow('Color Information', info_display)
        
        # Display the frame
        cv2.imshow(window_name, frame)
        
        # Handle keyboard input
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q') or key == 27:  # q or ESC
            break
        elif key & 0xFF == ord('c'):  # Clear selection
            clicked_point = None
            picked_color_bgr = None
            # Clear the information display
            cv2.imshow('Color Information', np.zeros((200, 400, 3), dtype=np.uint8))

finally:
    # Stop streaming
    print("Stopping RealSense pipeline...")
    pipeline.stop()
    cv2.destroyAllWindows()