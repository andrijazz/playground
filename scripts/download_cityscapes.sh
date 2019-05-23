#!/usr/bin/env bash

# store cookies
echo "setting up cookies..."

wget --keep-session-cookies --save-cookies=cookies.txt --post-data 'username=masacarica&password=masacarica&submit=Login' https://www.cityscapes-dataset.com/login/

# downloading ...
echo "downloading cityscapes..."

# left
wget --load-cookies cookies.txt --content-disposition https://www.cityscapes-dataset.com/file-handling/?packageID=3
# left rgb semantic
wget --load-cookies cookies.txt --content-disposition https://www.cityscapes-dataset.com/file-handling/?packageID=1

# right
wget --load-cookies cookies.txt --content-disposition https://www.cityscapes-dataset.com/file-handling/?packageID=5
# disparity
wget --load-cookies cookies.txt --content-disposition https://www.cityscapes-dataset.com/file-handling/?packageID=7

# unpacking
echo "unpacking cityscapes..."

unar -d leftImg8bit_trainvaltest.zip
unar -d gtFine_trainvaltest.zip
unar -d rightImg8bit_trainvaltest.zip
unar -d disparity_trainvaltest.zip
