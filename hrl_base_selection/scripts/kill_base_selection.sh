echo "Killing nodes started by base selection"
rosnode kill /autobed/autobed_state_publisher
rosnode kill /ar_track_alvar
rosnode kill /autobed_occupied_server
rosnode kill /autobed_state_publisher_node
rosnode kill /base_selection
rosnode kill /find_autobed
rosnode kill /find_head
rosnode kill /sm_pr2_servoing
echo "Nodes killed"
