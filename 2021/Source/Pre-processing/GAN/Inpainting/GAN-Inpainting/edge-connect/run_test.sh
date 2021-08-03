# DIR='./checkpoints'
# URL='https://drive.google.com/uc?export=download&id=1IrlFQGTpdQYdPeZIEgGUaSFpbYtNpekA'

echo "Testing pre-trained models..."
# mkdir -p $DIR
# FILE="$(curl -sc /tmp/gcokie "${URL}" | grep -o '="uc-name.*</span>' | sed 's/.*">//;s/<.a> .*//')" 
# curl -Lb /tmp/gcokie "${URL}&confirm=$(awk '/_warning_/ {print $NF}' /tmp/gcokie)" -o "$DIR/${FILE}" 

# echo "Extracting pre-trained models..."

python3 test.py --checkpoints ./checkpoints/places2 --input ../testcase/input --mask ../testcase/mask --output ../testcase/results/edgeconnect

echo "Testing success."