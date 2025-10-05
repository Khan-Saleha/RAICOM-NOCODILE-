#!/usr/bin/env python3
# -*- coding: utf-8 -*
 
import  os
import  sys
import  tty, termios
import cv2
import roslib
import rospy
from geometry_msgs.msg import Twist
from std_msgs.msg import String
 
# 全局变量
cmd = Twist()
pub = rospy.Publisher('cmd_vel', Twist, queue_size=1)
grasp_pub = rospy.Publisher('/grasp', String, queue_size=1)

# global can_grasp
# global can_release

# def grasp_status_cp(msg):
#     global can_release,can_grasp
#     # 物体抓取成功,让机器人回起始点
#     if msg.data=='1':
#         can_release=True
#     if msg.data=='0' or msg.data=='-1':
#         can_grasp=True
# grasp_status=rospy.Subscriber('/grasp_status', String, grasp_status_cp, queue_size=1)
def __init__(self):
        '''
        初始化参数
        x & y 代表水源的中心点位置信息
        x_prev & y_prev 代表前一轮记录的中心点
        found_count 代表识别中心点的次数
        self.found_*** 代表是否准确找到水源的中心点
        '''

        # 定义水源参数
        self.water_found_count = 0 ; self.found_water = False 
        self.water_x = 0 ; self.water_y = 0 ; self.water_x_prev = 0 ; self.water_y_prev = 0
        
        # 获取标定文件数据
        filename = os.environ['HOME'] + "/thefile.txt"
        with open(filename, 'r') as f:
            s = f.read()
        arr = s.split()
        self.x_kb = [float(arr[0]), float(arr[1])]
        self.y_kb = [float(arr[2]), float(arr[3])]
        rospy.logwarn('X axia k and b value: ' + str(self.x_kb))
        rospy.logwarn('X axia k and b value: ' + str(self.y_kb))

        # 发布机械臂位姿
        self.pub1 = rospy.Publisher('position_write_topic', position, queue_size=10)
        # 发布机械臂吸盘
        self.pub2 = rospy.Publisher('pump_topic', status, queue_size=1)
        # 发布TWist消息控制机器人底盘
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        # 订阅机械臂抓取指令
        self.sub2 = rospy.Subscriber('/grasp', String, self.grasp_cp, queue_size=1)
        # 发布机械臂恢复状态指令
        self.arm_status_pub = rospy.Publisher('/swiftpro_status_topic', status, queue_size=1)
        # 订阅摄像头话题,接收图像信息后跳转image_cb进行处理
        self.sub = rospy.Subscriber("/camera/color/image_raw", Image, self.image_cb, queue_size=1)


        # 发布信息让机械臂到指定位置
        self.arm_position_reset()
        pos = position()
        pos.x = 20
        pos.y = 150
        pos.z = 35
        self.pub1.publish(pos)

def image_cb(self, data):
        # 将 ROS image消息类型转换成opencv类型  
        try:
            cv_image1 = CvBridge().imgmsg_to_cv2(data, "bgr8")
            # print(cv_image1.shape)
        except CvBridgeError as e:
            print('error')
        # 由RGB颜色转换成HSV颜色空间
        cv_image2 = cv2.cvtColor(cv_image1, cv2.COLOR_BGR2HSV)
        # 蓝色物体颜色检测范围
        LowerBlue = np.array([95, 90, 80])
        UpperBlue = np.array([130, 255, 255])
        # 阈值处理
        mask = cv2.inRange(cv_image2, LowerBlue, UpperBlue)
        # 位运算，对图像进行掩膜处理
        cv_image3 = cv2.bitwise_and(cv_image2, cv_image2, mask=mask)
        # 取三维矩阵中第一维的所有数据
        cv_image4 = cv_image3[:, :, 0]
        # 均值滤波处理
        blurred = cv2.blur(cv_image4, (9, 9))
        # 简单阈值函数
        (_, thresh) = cv2.threshold(blurred, 90, 255, cv2.THRESH_BINARY)
        # 获取结构元素
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
        # 执行高级形态变换
        cv_image5 = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        # 腐蚀处理
        cv_image5 = cv2.erode(cv_image5, None, iterations=4)
        # 扩张处理
        cv_image5 = cv2.dilate(cv_image5, None, iterations=4) 
        # 检测轮廓
        contours, hier = cv2.findContours(cv_image5, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(cv_image1, contours, 0, (0, 255, 0), 2)
        # 显示图像
        

        # 根据检测的轮廓信息，找到其中距离最近的水源
        # len(contours) 识别到的水源的个数
        if len(contours) > 0:
            dis = []    
            dis_min = 640          
            # enumerate()为迭代的对象添加序列号
            for i, c in enumerate(contours):
                # 绘制水源图形
                rect = cv2.minAreaRect(c)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                # 计算水源中心点
                x_mid = (box[0][0] + box[2][0] + box[1][0] + box[3][0]) / 4
                y_mid = (box[0][1] + box[2][1] + box[1][1] + box[3][1]) / 4
                # w = math.sqrt((box[0][0] - box[1][0]) ** 2 + (box[0][1] - box[1][1]) ** 2)
                # h = math.sqrt((box[0][0] - box[3][0]) ** 2 + (box[0][1] - box[3][1]) ** 2)
                distance =  math.sqrt((x_mid-320)**2+(480-y_mid)**2)
                # 找出距离最近的水源
                dis.append(distance)
                if dis[i] < dis_min:    
                    dis_min = dis[i]
                    self.water_x = x_mid
                    self.water_y = y_mid
            
            # 基于有可能出现在移动或旋转过程中识别到水源的情况
            # 设定found_count参数判断水源有无移动
            # 当found_count次数达到30时，认为水源没有移动，可以进行定位           
            if self.water_found_count >= 30:
                # 将找到物体的参数设定为True
                self.found_water = True                
            else:
                # 判断识别的水源中心点位置有无移动
                if abs(self.water_x - self.water_x_prev) <= 4 and abs(self.water_y - self.water_y_prev) <= 4:
                    self.water_found_count = self.water_found_count + 1
                else:
                    # 一旦移动，归零
                    self.water_found_count = 0
        else:
            # 没有找到轮廓，判断水源移动参数归零
            self.water_found_count = 0

        self.water_x_prev = self.water_x
        self.water_y_prev = self.water_y
        
        cv2.circle(cv_image1, (int(self.water_x), int(self.water_y)), 5, (0, 0, 255), -1)

        cv2.imshow("contours",cv_image1)
        cv2.waitKey(1)

def keyboardLoop():
    rospy.init_node('teleop')
    #初始化监听键盘按钮时间间隔
    rate = rospy.Rate(rospy.get_param('~hz', 10))
 
    #速度变量
    # 慢速
    walk_vel_ = rospy.get_param('walk_vel', 0.1)
    # 快速
    run_vel_ = rospy.get_param('run_vel', 0.5)
    yaw_rate_ = rospy.get_param('yaw_rate', 0.5)
    yaw_rate_run_ = rospy.get_param('yaw_rate_run', 1.0)
    # walk_vel_前后速度
    max_tv = walk_vel_
    # yaw_rate_旋转速度
    max_rv = yaw_rate_
    # 参数初始化
    speed=0
    # global can_release,can_grasp
    # can_grasp=True
    # can_release=False
    x=0
    y=1
    print ("使用[WASD]控制机器人")
    print ("按[G]抓取水源")
    print ("按[H]放下物体到第一L")
    print ("按[Q]退出" )

    #读取按键循环
    while not rospy.is_shutdown():
        # linux下读取键盘按键
        fd = sys.stdin.fileno()
        turn =0
        old_settings = termios.tcgetattr(fd)
		#不产生回显效果
        old_settings[3] = old_settings[3] & ~termios.ICANON & ~termios.ECHO
        try :
            tty.setraw( fd )
            ch = sys.stdin.read( 1 )
        finally :
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        # ch代表获取的键盘按键
        
        if ch == 'x':
            while x==0:
                if (): # () = conditon received from dataset
                    speed = 0
                    turn = 0   
                else:
                    speed = 1
                    turn = 0
        if ch == 'c':
            while x==0:
                if (): # () = conditon received from dataset
                    while y >0 :
                        speed = y
                        turn = 0
                        y-=0.02  
                else:
                    speed = 1
                    turn = 0
        if ch == 'v':
            while x==0:
                if (): # () = conditon received from dataset
                    speed = 0
                    turn = 0   
                else:
                    speed = 1
                    turn = 0
                


        elif ch == 'h':
            # if can_release:
            msg=String()
            msg.data='1'
            grasp_pub.publish(msg)
            # can_release=False
            speed = 0
            turn = 0
        elif ch == 'w':
            max_tv = walk_vel_
            speed = 1
            turn = 0
        elif ch == 's':
            max_tv = walk_vel_
            speed = -1
            turn = 0
        elif ch == 'a':
            max_rv = yaw_rate_
            speed = 0
            turn = 1
        elif ch == 'd':
            max_rv = yaw_rate_
            speed = 0
            turn = -1
        elif ch == 'W':
            max_tv = run_vel_
            speed = 1
            turn = 0
        elif ch == 'S':
            max_tv = run_vel_
            speed = -1
            turn = 0
        elif ch == 'A':
            max_rv = yaw_rate_run_
            speed = 0
            turn = 1
        elif ch == 'D':
            max_rv = yaw_rate_run_
            speed = 0
            turn = -1
        elif ch == 'q':
            exit()
        else:
            max_tv = walk_vel_
            max_rv = yaw_rate_
            speed = 0
            turn = 0

        #发送消息
        cmd.linear.x = speed * max_tv
        cmd.angular.z = turn * max_rv
        pub.publish(cmd)
        rate.sleep()
		#停止机器人
        #stop_robot()
 
def stop_robot():
    cmd.linear.x = 0.0
    cmd.angular.z = 0.0
    pub.publish(cmd)
 
if __name__ == '__main__':
    try:
        keyboardLoop()
    except rospy.ROSInterruptException:
        pass

