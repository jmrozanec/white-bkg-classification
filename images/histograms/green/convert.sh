for file in *.jpg;
do
#	convert $file -colors 2 -colorspace Gray $file.copy
	convert $file -colorspace Gray -colors 2 $file.copy
#	mv $file.copy $file
done
