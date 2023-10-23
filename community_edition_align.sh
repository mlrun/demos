cat enterprise_demos.txt | while read line
do
   echo removing enterprise $line
   rm -r $line
done

