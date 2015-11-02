#!/bin/bash

kill -9 `ps aux | grep julius | awk '{print $2}'`

echo "After you see the message \"waiting client at 10500\","
echo "press enter again [Press Enter]"
read Dmy

exec julius_mft -C ./julius/julius.jconf -module &

read Dmy
#exec rosrun hrl_sound_localization receive.py | tee result.txt
exec python ../src/hrl_sound_localization/receive.py | tee result.txt
