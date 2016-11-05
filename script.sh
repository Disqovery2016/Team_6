rm feed.csv
rm file2 #remove file2 if it exists 
wget https://thingspeak.com/channels/179009/feed.csv
sed -i '1d' feed.csv #deleted the first line
touch file2 #created a file first so that it does not get overridden later
FILE=$1
while read line; do
echo "$line" | awk -F"," '{print $3","$4","$5}' > file2
done <$FILE
#now i have to give this file "file2" as input to my classifier
java -jar tcs.jar



