cat enterprise_demos.txt | while read line
do
   echo rm -r $line
done