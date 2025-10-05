# Traffic Management System for Spark Robot

This system provides comprehensive traffic management capabilities for your Spark robot, including traffic light detection, pedestrian crossing management, and zebra crossing awareness.

## 🚦 Features

### 1. Traffic Light Control
- **Red Light Detection**: Automatically stops the robot
- **Green Light Detection**: Allows robot to proceed
- **Color Analysis**: Uses HSV color space for reliable detection

### 2. Children Crossing Management
- **Child Detection**: Detects children on zebra crossings
- **Automatic Stop**: Stops robot when children are detected
- **Safety Priority**: Highest priority in traffic decisions

### 3. Zebra Crossing Awareness
- **Crossing Detection**: Identifies zebra crossings
- **Speed Control**: Automatically slows down when approaching
- **Distance-Based Response**: Responds based on proximity

## 📁 Files

1. **`traffic_controller.py`** - Main traffic management system
2. **`semi_auto_traffic.launch`** - Launch file with traffic system
3. **`TRAFFIC_SYSTEM_README.md`** - This documentation

## 🚀 Quick Start

### Launch with Traffic Management
```bash
roslaunch semi_auto_match_24 semi_auto_traffic.launch enable_traffic:=true
```

### Launch with Both Traffic and Grasping
```bash
roslaunch semi_auto_match_24 semi_auto_traffic.launch enable_traffic:=true enable_grasping:=true
```

### Launch Traffic Only (No Grasping)
```bash
roslaunch semi_auto_match_24 semi_auto_traffic.launch enable_traffic:=true enable_grasping:=false
```

## 🎯 Traffic Scenarios

### Scenario 1: Traffic Light
**Materials Required**: Physical traffic light model with red/green LEDs

**Robot Behavior**:
- **Red Light**: Robot stops immediately
- **Green Light**: Robot continues moving
- **No Light**: Robot continues normal operation

### Scenario 2: Children Crossing
**Materials Required**: 
- Rope
- Paper card of children tied to rope
- Zebra crossing markings

**Robot Behavior**:
- **Child Detected on Crossing**: Robot stops immediately
- **No Child Visible**: Robot proceeds normally
- **Crossing Detected**: Robot slows down

### Scenario 3: Zebra Crossing Approach
**Materials Required**: Yellow/white paper strips on ground resembling zebra crossing

**Robot Behavior**:
- **Crossing at Distance**: Robot slows down
- **No Children Present**: Robot continues slowly
- **Children Present**: Robot stops completely

## ⚙️ Configuration

### Speed Settings
```python
self.robot_speed = 0.2      # Normal speed (m/s)
self.slow_speed = 0.05      # Slow speed for crossings (m/s)
self.stop_speed = 0.0       # Stop speed (m/s)
```

### Detection Thresholds
```python
self.confidence_threshold = 0.5           # YOLO detection confidence
self.stable_threshold = 5                 # Frames needed for stable detection
self.zebra_crossing_distance_threshold = 150  # Distance threshold in pixels
```

### Color Detection Thresholds
```python
red_ratio > 0.1    # Red light detection threshold
green_ratio > 0.1  # Green light detection threshold
```

## 🎮 Control System

### Priority System
1. **Traffic Lights** (Highest Priority)
2. **Children on Crossing** (High Priority)
3. **Zebra Crossing** (Medium Priority)
4. **Normal Movement** (Default)

### Robot States
- **MOVING**: Normal speed operation
- **SLOWING**: Reduced speed for crossings
- **STOPPED**: Complete stop for safety

## 📊 ROS Topics

### Published Topics
- **`/cmd_vel`**: Robot movement commands
- **`/traffic/status`**: Traffic system status
- **`/traffic/image_with_detections`**: Visualization image

### Subscribed Topics
- **`/camera/color/image_raw`**: Camera feed

### Status Message Format
```
state:MOVING:speed:0.2:red_light:false:green_light:true:child:false:zebra:false
```

## 🔧 Setup Instructions

### 1. Install Dependencies
```bash
# Run the YOLO installation script
chmod +x install_yolo.sh
./install_yolo.sh
```

### 2. Prepare Traffic Scenarios

#### Traffic Light Setup
- Create a physical traffic light model
- Use bright LEDs for red and green lights
- Ensure good contrast against background

#### Children Crossing Setup
- Create paper cutouts of children
- Tie to rope for movement simulation
- Place on zebra crossing area

#### Zebra Crossing Setup
- Use yellow/white tape or paper
- Create horizontal stripes on ground
- Ensure good visibility from camera

### 3. Camera Calibration
- Ensure camera is properly mounted
- Check camera focus and lighting
- Test detection ranges

## 🧪 Testing

### Test Traffic Light Detection
1. Place traffic light in camera view
2. Toggle between red and green
3. Observe robot stopping/starting behavior

### Test Children Crossing
1. Place child figure on zebra crossing
2. Move child in/out of crossing area
3. Verify robot stops when child is present

### Test Zebra Crossing
1. Approach zebra crossing slowly
2. Verify robot slows down
3. Test with/without children present

## 🐛 Troubleshooting

### Common Issues

1. **Traffic Light Not Detected**
   - Check lighting conditions
   - Ensure sufficient color contrast
   - Adjust detection thresholds

2. **Children Not Detected**
   - Verify YOLO model includes 'person' class
   - Check if child is in zebra crossing area
   - Adjust confidence threshold

3. **Robot Not Stopping**
   - Check `/cmd_vel` topic publishing
   - Verify traffic controller is running
   - Check for conflicting movement commands

4. **False Detections**
   - Adjust confidence thresholds
   - Improve lighting conditions
   - Add stable detection requirements

### Debug Commands
```bash
# Check traffic status
rostopic echo /traffic/status

# Monitor robot commands
rostopic echo /cmd_vel

# View detection image
rqt_image_view /traffic/image_with_detections
```

## 📈 Performance Optimization

### For Better Detection
- Use good lighting conditions
- Ensure clear contrast for traffic lights
- Position camera at appropriate height

### For Smoother Operation
- Adjust speed parameters based on robot capabilities
- Fine-tune detection thresholds
- Use stable detection requirements

## 🔮 Future Enhancements

Potential improvements:
- Custom traffic light detection model
- Multiple pedestrian tracking
- Intersection management
- Traffic sign recognition
- Emergency vehicle detection
- Weather-aware operation

## 🚨 Safety Notes

- Always supervise robot operation
- Test in safe, controlled environment
- Ensure emergency stop capability
- Verify detection accuracy before autonomous operation
- Keep human operator ready to take control

## 📞 Support

For issues or questions:
1. Check ROS logs: `roslog` or `rostopic echo /rosout`
2. Verify camera topic: `rostopic list | grep camera`
3. Test YOLO model: `python3 -c "from ultralytics import YOLO; YOLO('yolo11n.pt').predict('test_image.jpg')"`
4. Monitor traffic status: `rostopic echo /traffic/status`
