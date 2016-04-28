///////////////////////////////////////////////////////////////
This is a readme file for the GUI for hrl_manipulation_task
This gives a brief overview of the GUI implementation.
///////////////////////////////////////////////////////////////
Written by You Keun Kim
///////////////////////////////////////////////////////////////


GUI uses [assistive_teleop] package to make the web-based GUI.
Look at the [assistive_teleop/launch/hrl_manipulation_task_assistive.launch] to see the launch files.

To add new tabs, buttons or slider, see [assistive_teleop/www/hz_tab_interface.html]. Use this file to add tabs, buttons, sliders, etc.

After adding new tabs, buttons or sliders on [assistive_teleop/www/hz_tab_interface.html], two things must be done in order for added buttons and tabs to function.

1.) Adding new js file.
2.) Calling new js file in [hz_tab_interface.html] and [init.js].

Add a new js file in [assistive_teleop/www/js] folder. There are multiple examples of it in this folder.

After making js file, make sure [assistive_teleop/www/hz_tab_interface.html] is importing the new js file. Also, in [assistive_teleop/www/js/init.js] file, add new init function for the new js file.

Currently GUI uses following files.

man_Task.js: This file calls out function for the manipulation tab.

ad_slider.js: This file calls out function for slider in the manipulation tab. However, inactivation and activation of sliders are done in man_Task.js

////////////////////////////////////////////////////////////////////

Video Feedback

Video feedback file is in [hrl_manipulation_task/src/Kinect2Overlay.cpp] This file is based upon the head-confirmation program. Using openCV2, new overlay can be drawn on the image file.

After the change in Kinect2Overlay.cpp, modify file [assistive_teleop/www/js/video/mjpeg_client.js]. In self.cameraData, change 'Feedback'
 features to match the new overlay. (topic name, width, height,etc)

Installation
sudo apt-get install ros-indigo-pr2-common-action-msgs ros-indigo-rosbridge-server ros-indigo-mjpeg-server


