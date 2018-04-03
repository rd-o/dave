#-*- coding: utf-8 -*-

import cv2
import time
import numpy as np
import math

def detect_lines(img):

    #gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    kernel_size = 5
    blur_gray = cv2.GaussianBlur(img,(kernel_size, kernel_size),0)

    low_threshold = 50
    high_threshold = 150
    edges = cv2.Canny(blur_gray, low_threshold, high_threshold)

    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 15  # minimum number of votes (intersections in Hough grid cell)
    #min_line_length = 50  # minimum number of pixels making up a line
    min_line_length = 100  # minimum number of pixels making up a line
    max_line_gap = 20  # maximum gap in pixels between connectable line segments
    line_image = np.copy(img) * 0  # creating a blank to draw lines on

    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
			min_line_length, max_line_gap)

    size=50
    #line_image = img
    if lines is not None:
        for line in lines:
            for x1,y1,x2,y2 in line:
                #cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),5)
                len1 = math.sqrt(pow(x1-x2,2)+pow(y1-y2,2))
                x4 = int(x2+(x2-x1)/len1*size)
                y4 = int(y2+(y2-y1)/len1*size)
                x3 = int(x1+(x1-x2)/len1*size)
                y3 = int(y1+(y1-y2)/len1*size)
                cv2.line(line_image,(x3,y3),(x4,y4),(255,0,0),5)

    return line_image
"""
sudo apt-get install python-opencv
sudo apt-get install python-matplotlib
"""

##################
DELAY = 0.02
USE_CAM = 0
IS_FOUND = 0

MORPH = 7
CANNY = 250
##################
# 420x600 oranı 105mmx150mm gerçek boyuttaki kağıt için
_width  = 600.0
_height = 420.0
_margin = 0.0
##################

if USE_CAM: video_capture = cv2.VideoCapture(0)

corners = np.array(
  [
    [[      _margin, _margin       ]],
    [[       _margin, _height + _margin  ]],
    [[ _width + _margin, _height + _margin  ]],
    [[ _width + _margin, _margin       ]],
  ]
)

pts_dst = np.array( corners, np.float32 )

while True :

  if USE_CAM :
    ret, rgb = video_capture.read()
  else :
    ret = 1
    rgb = cv2.imread( "test.jpg", 1 )

  if ( ret ):

    gray = cv2.cvtColor( rgb, cv2.COLOR_BGR2GRAY )

    gray = cv2.bilateralFilter( gray, 1, 10, 120 )

    edges  = cv2.Canny( gray, 10, CANNY )

    kernel = cv2.getStructuringElement( cv2.MORPH_RECT, ( MORPH, MORPH ) )

    closed = cv2.morphologyEx( edges, cv2.MORPH_CLOSE, kernel )

    lines = detect_lines(closed)
    cv2.namedWindow( 'lines' )
    cv2.imshow( 'lines', lines )

    _, contours, h = cv2.findContours( lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE )

    for cont in contours:

      # Küçük alanları pass geç
      if cv2.contourArea( cont ) > 500 :

        arc_len = cv2.arcLength( cont, True )

        approx = cv2.approxPolyDP( cont, 0.1 * arc_len, True )

        if ( len( approx ) == 4 ):
          print("found!")
          IS_FOUND = 1
           #M = cv2.moments( cont )
          #cX = int(M["m10"] / M["m00"])
          #cY = int(M["m01"] / M["m00"])
          #cv2.putText(rgb, "Center", (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

          pts_src = np.array( approx, np.float32 )

          h, status = cv2.findHomography( pts_src, pts_dst )
          out = cv2.warpPerspective( rgb, h, ( int( _width + _margin * 2 ), int( _height + _margin * 2 ) ) )

          cv2.drawContours( rgb, [approx], -1, ( 255, 0, 0 ), 2 )

        else : pass

    #cv2.imshow( 'closed', closed )
    #cv2.imshow( 'gray', gray )
    cv2.namedWindow( 'edges' )
    cv2.imshow( 'edges', edges )

    cv2.namedWindow( 'rgb', )
    cv2.imshow( 'rgb', rgb )

    if IS_FOUND :
      cv2.namedWindow( 'out', )
      cv2.imshow( 'out', out )

    if cv2.waitKey(27) & 0xFF == ord('q') :
      break

    if cv2.waitKey(99) & 0xFF == ord('c') :
      current = str( time.time() )
      cv2.imwrite( 'ocvi_' + current + '_edges.jpg', edges )
      cv2.imwrite( 'ocvi_' + current + '_gray.jpg', gray )
      cv2.imwrite( 'ocvi_' + current + '_org.jpg', rgb )
      print("Pictures saved")

    time.sleep( DELAY )

  else :
    print("Stopped")
    break

if USE_CAM : video_capture.release()
cv2.destroyAllWindows()

# end
