cat enterprise_demos.txt | while read line
do
   rm -r $line
done